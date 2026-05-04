# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import subprocess
import numpy as np

import pytest
import sys

from hatchet import GraphFrame
from hatchet.readers.caliper_reader import CaliperReader
from hatchet.util.executable import which
from hatchet.external.console import ConsoleRenderer

caliperreader_avail = True
try:
    import caliperreader
except ImportError:
    caliperreader_avail = False

annotations = [
    "main",
    "LagrangeLeapFrog",
    "LagrangeElements",
    "ApplyMaterialPropertiesForElems",
    "EvalEOSForElems",
    "CalcEnergyForElems",
    "CalcPressureForElems",
    "CalcSoundSpeedForElems",
    "UpdateVolumesForElems",
    "CalcTimeConstraintsForElems",
    "CalcCourantConstraintForElems",
    "CalcHydroConstraintForElems",
    "TimeIncrement",
    "LagrangeNodal",
    "CalcForceForNodes",
    "CalcVolumeForceForElems",
    "IntegrateStressForElems",
    "CalcHourglassControlForElems",
    "CalcFBHourglassForceForElems",
    "CalcLagrangeElements",
    "CalcKinematicsForElems",
    "CalcQForElems",
    "CalcMonotonicQGradientsForElems",
    "CalcMonotonicQRegionForElems",
]


def test_graphframe(lulesh_caliper_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_caliper(str(lulesh_caliper_json))

    assert len(gf.dataframe.groupby("name")) == 24

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "node"):
            assert gf.dataframe[col].dtype == object

    # TODO: add tests to confirm values in dataframe


def test_read_lulesh_json(lulesh_caliper_json):
    """Sanity check the Caliper reader by examining a known input."""
    reader = CaliperReader(str(lulesh_caliper_json))
    reader.read_json_sections()

    assert len(reader.json_data) == 192
    assert len(reader.json_cols) == 4
    assert len(reader.json_cols_mdata) == 4
    assert len(reader.json_nodes) == 24

    reader.create_graph()
    assert all(an in reader.idx_to_label.values() for an in annotations)


def test_calc_pi_json(calc_pi_caliper_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_caliper(str(calc_pi_caliper_json))

    assert len(gf.dataframe.groupby("name")) == 100


@pytest.mark.skipif(not which("cali-query"), reason="needs cali-query to be in path")
def test_lulesh_cali(lulesh_caliper_cali):
    """Sanity check the Caliper reader ingesting a .cali file."""
    grouping_attribute = "function"
    default_metric = "sum(sum#time.duration),inclusive_sum(sum#time.duration)"
    query = "select function,%s group by %s format json-split" % (
        default_metric,
        grouping_attribute,
    )

    gf = GraphFrame.from_caliper(str(lulesh_caliper_cali), query)

    assert len(gf.dataframe.groupby("name")) == 18


@pytest.mark.skipif(not which("cali-query"), reason="needs cali-query to be in path")
def test_lulesh_json_stream(lulesh_caliper_cali):
    """Sanity check the Caliper reader ingesting a JSON string literal."""
    cali_query = which("cali-query")
    grouping_attribute = "function"
    default_metric = "sum(sum#time.duration),inclusive_sum(sum#time.duration)"
    query = "select function,%s group by %s format json-split" % (
        default_metric,
        grouping_attribute,
    )

    cali_json = subprocess.Popen(
        [cali_query, "-q", query, lulesh_caliper_cali], stdout=subprocess.PIPE
    )

    gf = GraphFrame.from_caliper(cali_json.stdout)

    assert len(gf.dataframe.groupby("name")) == 18


@pytest.mark.skipif(sys.version_info > (3, 8), reason="Temporarily let this test fail.")
def test_filter_squash_unify_caliper_data(lulesh_caliper_json):
    """Sanity test a GraphFrame object with known data."""
    gf1 = GraphFrame.from_caliper(str(lulesh_caliper_json))
    gf2 = GraphFrame.from_caliper(str(lulesh_caliper_json))

    assert gf1.graph is not gf2.graph

    gf1_index_names = gf1.dataframe.index.names
    gf2_index_names = gf2.dataframe.index.names

    gf1.dataframe.reset_index(inplace=True)
    gf2.dataframe.reset_index(inplace=True)

    # indexes are the same since we are reading in the same dataset
    assert all(gf1.dataframe["node"] == gf2.dataframe["node"])

    gf1.dataframe.set_index(gf1_index_names, inplace=True)
    gf2.dataframe.set_index(gf2_index_names, inplace=True)

    squash_gf1 = gf1.filter(lambda x: x["name"].startswith("Calc"))
    squash_gf2 = gf2.filter(lambda x: x["name"].startswith("Calc"))

    squash_gf1.unify(squash_gf2)

    assert squash_gf1.graph is squash_gf2.graph

    squash_gf1.dataframe.reset_index(inplace=True)
    squash_gf2.dataframe.reset_index(inplace=True)

    # Indexes should still be the same after unify. Sort indexes before comparing.
    assert all(squash_gf1.dataframe["node"] == squash_gf2.dataframe["node"])

    squash_gf1.dataframe.set_index(gf1_index_names, inplace=True)
    squash_gf2.dataframe.set_index(gf2_index_names, inplace=True)


def test_tree(lulesh_caliper_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_caliper(str(lulesh_caliper_json))

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="time",
        precision=3,
        name_column="name",
        expand_name=False,
        context_column="file",
        rank=0,
        thread=0,
        depth=10000,
        highlight_name=False,
        colormap="RdYlGn",
        invert_colormap=False,
    )
    assert "121489.000 main" in output
    assert "663.000 LagrangeElements" in output
    assert "21493.000 CalcTimeConstraintsForElems" in output

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="time (inc)",
        precision=3,
        name_column="name",
        expand_name=False,
        context_column="file",
        rank=0,
        thread=0,
        depth=10000,
        highlight_name=False,
        colormap="RdYlGn",
        invert_colormap=False,
    )
    assert "662712.000 EvalEOSForElems" in output
    assert "2895319.000 LagrangeNodal" in output


def test_graphframe_to_literal(lulesh_caliper_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_caliper(str(lulesh_caliper_json))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)


def test_graphframe_native_lulesh_from_file(lulesh_caliper_cali):
    """Sanity check the native Caliper reader by examining a known input."""

    gf = GraphFrame.from_caliperreader(str(lulesh_caliper_cali))

    assert len(gf.dataframe.groupby("name")) == 19
    assert "cali.caliper.version" in gf.metadata.keys()

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "node"):
            assert gf.dataframe[col].dtype == object


@pytest.mark.skipif(
    not caliperreader_avail, reason="needs caliper-reader package to be loaded"
)
def test_graphframe_native_lulesh_from_caliperreader(lulesh_caliper_cali):
    """Sanity check the native Caliper reader by examining a known input."""
    r = caliperreader.CaliperReader()
    r.read(lulesh_caliper_cali)

    gf = GraphFrame.from_caliperreader(r)

    assert len(gf.dataframe.groupby("name")) == 19
    assert "cali.caliper.version" in gf.metadata.keys()

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "node"):
            assert gf.dataframe[col].dtype == object


def test_inclusive_time_calculation(lulesh_caliper_json):
    """Validate calculate_inclusive_metrics() on known dataset containing per-rank data."""
    gf = GraphFrame.from_caliper(str(lulesh_caliper_json))

    # save original time (inc) column for correctness check
    gf.dataframe["orig_inc_time"] = gf.dataframe["time (inc)"]

    # remove original time (inc) column since it will be generated by calculate_inclusive_metrics()
    del gf.dataframe["time (inc)"]

    gf.calculate_inclusive_metrics()
    assert all(
        gf.dataframe["time (inc)"].values == gf.dataframe["orig_inc_time"].values
    )


def test_sw4_cuda_from_caliperreader(sw4_caliper_cuda_activity_profile_cali):
    gf = GraphFrame.from_caliperreader(sw4_caliper_cuda_activity_profile_cali)

    assert len(gf.graph) == 549
    assert all(
        metric in gf.dataframe.columns for metric in gf.exc_metrics + gf.inc_metrics
    )

    for col in gf.dataframe.columns:
        if col in ("#scale#cupti.activity.duration", "#scale#sum#cupti.host.duration"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in "rank":
            assert gf.dataframe[col].dtype == np.int64
        elif col in "name":
            assert gf.dataframe[col].dtype == object

    for col in gf.exc_metrics + gf.inc_metrics:
        assert col in gf.dataframe.columns


def test_sw4_cuda_summary_from_caliperreader(
    sw4_caliper_cuda_activity_profile_summary_cali,
):
    gf = GraphFrame.from_caliperreader(sw4_caliper_cuda_activity_profile_summary_cali)

    assert len(gf.graph) == 393
    assert all(
        metric in gf.dataframe.columns for metric in gf.exc_metrics + gf.inc_metrics
    )

    for col in gf.exc_metrics + gf.inc_metrics:
        assert col in gf.dataframe.columns
