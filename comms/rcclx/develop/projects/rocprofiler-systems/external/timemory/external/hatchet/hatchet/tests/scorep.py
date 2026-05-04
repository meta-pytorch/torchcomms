# Copyright 2021-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from hatchet import GraphFrame
from hatchet.external.console import ConsoleRenderer

import pytest

# check if the pycubexr package is available
# without importing the package.
# there are other ways to do it for python 3+
import imp

pycubexr_avail = True
try:
    imp.find_module("pycubexr")
except ImportError:
    pycubexr_avail = False


procedures = [
    "cpi",
    "main",
    "MPI_Init",
    "MPI_Comm_size",
    "MPI_Comm_rank",
    "iteration",
    "MPI_Bcast",
    "MPI_Reduce",
    "iteration.cold.1",
    "MPI_Finalize",
]


@pytest.mark.skipif(not pycubexr_avail, reason="pycubexr package not available")
def test_graphframe(scorep_profile_cubex):
    gf = GraphFrame.from_scorep(str(scorep_profile_cubex))

    assert len(gf.dataframe.groupby("name")) == 10

    for col in gf.dataframe.columns:
        if col in (
            "max time (inc)",
            "min time (inc)",
            "time",
        ):
            assert gf.dataframe[col].dtype == np.float64
        elif col in (
            "hits",
            "end_line",
            "line",
            "visits",
            "bytes_sent",
            "bytes_received",
        ):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "file", "node"):
            assert gf.dataframe[col].dtype == object


@pytest.mark.skipif(not pycubexr_avail, reason="pycubexr package not available")
def test_tree(scorep_profile_cubex):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_scorep(str(scorep_profile_cubex))

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
    assert "5.056 cpi" in output
    assert "0.507 MPI_Init MPI" in output
    assert "4.539 iteration /p/lustre1/cankur1/test/scorep/cpi.c" in output

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="visits",
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
    assert "1.000 cpi" in output
    assert "100082.000 iteration /p/lustre1/cankur1/test/scorep/cpi.c" in output


@pytest.mark.skipif(not pycubexr_avail, reason="pycubexr package not available")
def test_graphframe_to_literal(scorep_profile_cubex):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_scorep(str(scorep_profile_cubex))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)
