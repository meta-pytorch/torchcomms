# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import os

from hatchet import GraphFrame
from hatchet.readers.hpctoolkit_reader import HPCToolkitReader
from hatchet.external.console import ConsoleRenderer

modules = [
    "cpi",
    "/collab/usr/global/tools/hpctoolkit/chaos_5_x86_64_ib/"
    "hpctoolkit-2017-03-16/lib/hpctoolkit/ext-libs/libmonitor.so.0.0.0",
    "/usr/local/tools/mvapich2-intel-debug-2.2/lib/libmpi.so.12.0.5",
    "/lib64/libc-2.12.so",
    "/usr/lib64/libpsm_infinipath.so.1.14",
]

src_files = [
    "./src/cpi.c",
    "<unknown file>",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpi/" "init/init.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpi/" "init/initthread.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/"
    "ch3/src/mpid_init.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/"
    "ch3/channels/psm/src/mpidi_calls.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/"
    "ch3/channels/psm/src/psm_entry.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpi/" "init/finalize.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/"
    "ch3/src/mpid_finalize.c",
    "/tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/"
    "ch3/channels/psm/src/psm_exit.c",
    "interp.c",
    "<unknown file>",
]

procedures = [
    "main",
    "<program root>",
    "MPI_Init",
    "pthread_create",
    "MPI_Finalize",
    "PMPI_Init",
    "MPIR_Init_thread",
    "MPID_Init",
    "MPIDI_CH3_Init",
    "MPIDI_CH3_Finalize",
    "psm_doinit",
    "PMPI_Finalize",
    "MPID_Finalize",
    "psm_dofinalize",
    "__GI_sched_yield",
    "<unknown procedure>",
]


def test_graphframe(data_dir, calc_pi_hpct_db):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

    assert len(gf.dataframe.groupby("module")) == 5
    assert len(gf.dataframe.groupby("file")) == 11
    assert len(gf.dataframe.groupby("name")) == 20

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank", "line"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "type", "file", "module", "node"):
            assert gf.dataframe[col].dtype == object

    # add tests to confirm values in dataframe
    df = pd.read_csv(str(os.path.join(data_dir, "hpctoolkit-cpi-graphframe.csv")))

    gf.dataframe.reset_index(inplace=True)
    df.reset_index(inplace=True)

    gf.dataframe.sort_values(by=["nid", "rank"], inplace=True)
    df.sort_values(by=["nid", "rank"], inplace=True)

    t1 = gf.dataframe["time"].values
    t2 = df["time"].values

    ti1 = gf.dataframe["time (inc)"].values
    ti2 = df["time (inc)"].values

    for v1, v2 in zip(t1, t2):
        assert v1 == v2

    for v1, v2 in zip(ti1, ti2):
        assert v1 == v2


def test_tree(calc_pi_hpct_db):
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

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
    assert "0.000 <program root> <unknown file>" in output
    assert (
        "0.000 198:MPIR_Init_thread /tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpi/init/initthread.c"
        in output
    )

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
    assert "17989.000 interp.c:0 interp.c" in output
    assert (
        "999238.000 230:psm_dofinalize /tmp/dpkg-mkdeb.gouoc49UG7/src/mvapich/src/build/../src/mpid/ch3/channels/psm/src/psm_exit.c"
        in output
    )


def test_read_calc_pi_database(calc_pi_hpct_db):
    """Sanity check the HPCT database reader by examining a known input."""
    reader = HPCToolkitReader(str(calc_pi_hpct_db))
    reader.fill_tables()

    assert len(reader.load_modules) == 5
    assert len(reader.src_files) == 12
    assert len(reader.procedure_names) == 16
    assert all(lm in reader.load_modules.values() for lm in modules)
    assert all(sf in reader.src_files.values() for sf in src_files)
    assert all(pr in reader.procedure_names.values() for pr in procedures)


def test_allgather(data_dir, osu_allgather_hpct_db):
    gf = GraphFrame.from_hpctoolkit(str(osu_allgather_hpct_db))

    assert len(gf.dataframe.groupby("module")) == 9
    assert len(gf.dataframe.groupby("file")) == 41
    assert len(gf.dataframe.groupby("name")) == 383
    assert len(gf.dataframe.groupby("type")) == 3

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank", "thread", "line"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "type", "file", "module", "node"):
            assert gf.dataframe[col].dtype == object

    # add tests to confirm values in dataframe
    df = pd.read_csv(str(os.path.join(data_dir, "hpctoolkit-allgather-graphframe.csv")))

    gf.dataframe.reset_index(inplace=True)
    df.reset_index(inplace=True)

    gf.dataframe.sort_values(by=["nid", "rank", "thread"], inplace=True)
    df.sort_values(by=["nid", "rank", "thread"], inplace=True)

    t1 = gf.dataframe["time"].values
    t2 = df["time"].values

    ti1 = gf.dataframe["time (inc)"].values
    ti2 = df["time (inc)"].values

    for v1, v2 in zip(t1, t2):
        assert v1 == v2

    for v1, v2 in zip(ti1, ti2):
        assert v1 == v2


def test_graphframe_to_literal(data_dir, calc_pi_hpct_db):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)


def test_graphframe_to_literal_with_threads(data_dir, osu_allgather_hpct_db):
    gf = GraphFrame.from_hpctoolkit(str(osu_allgather_hpct_db))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)


def test_inclusive_time_calculation(data_dir, calc_pi_hpct_db):
    """Validate calculate_inclusive_metrics() on known dataset."""
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

    # save original time (inc) column for correctness check
    gf.dataframe["orig_inc_time"] = gf.dataframe["time (inc)"]

    # remove original time (inc) column since it will be generated by calculate_inclusive_metrics()
    del gf.dataframe["time (inc)"]

    gf.calculate_inclusive_metrics()
    assert all(
        gf.dataframe["time (inc)"].values == gf.dataframe["orig_inc_time"].values
    )


# START OF TESTS FOR THE NEW FORMAT
def test_graphframe_v4(data_dir, calc_pi_hpct_v4_db):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_v4_db))

    for col in gf.dataframe.columns:
        if col in gf.inc_metrics or col in gf.exc_metrics:
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("line"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "node", "file", "module"):
            assert gf.dataframe[col].dtype == object

    # TODO: add tests to confirm values in dataframe


def test_tree_v4(calc_pi_hpct_v4_db):
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_v4_db))

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
    assert "0.088 main src/home/ocankur/apps/test/hatchet_cpi/cpi.c" in output
    assert (
        "0.058 PMPI_Reduce [libmpi.so.40.30.1] src/home/ocankur/apps/test/hatchet_cpi/cpi.c"
        in output
    )
    assert "0.029 MPI_Finalize src/home/ocankur/apps/test/hatchet_cpi/cpi.c" in output
    assert (
        "0.000 PMPI_Bcast [libmpi.so.40.30.1] src/home/ocankur/apps/test/hatchet_cpi/cpi.c"
        in output
    )

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="time (inc)",
        precision=3,
        name_column="name",
        expand_name=False,
        context_column="file",
        rank=3,
        thread=0,
        depth=10000,
        highlight_name=False,
        colormap="RdYlGn",
        invert_colormap=False,
    )
    assert "0.090 main src/home/ocankur/apps/test/hatchet_cpi/cpi.c" in output
    assert (
        "0.059 PMPI_Bcast [libmpi.so.40.30.1] src/home/ocankur/apps/test/hatchet_cpi/cpi.c"
        in output
    )
    assert (
        "0.000 PMPI_Reduce [libmpi.so.40.30.1] src/home/ocankur/apps/test/hatchet_cpi/cpi.c"
        in output
    )
    assert "0.030 MPI_Finalize src/home/ocankur/apps/test/hatchet_cpi/cpi.c" in output


# def test_inclusive_metric_calculation_new_format(data_dir, saxpy_mpi_cuda_newhpct_db):
#     """Validate calculate_inclusive_metrics() on known dataset."""
#     gf = GraphFrame.from_hpctoolkit(str(saxpy_mpi_cuda_newhpct_db))

#     metric = gf.exc_metrics[0]
#     # save original time (inc) column for correctness check
#     gf.dataframe["orig_inc_time"] = gf.dataframe[metric + " (inc)"]

#     # remove original time (inc) column since it will be generated by calculate_inclusive_metrics()
#     del gf.dataframe[metric + " (inc)"]

#     gf.calculate_inclusive_metrics()
#     assert all(
#         gf.dataframe[metric + " (inc)"].values == gf.dataframe["orig_inc_time"].values
#     )


def test_graphframe_to_literal__v4(data_dir, calc_pi_hpct_v4_db):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_v4_db))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)
