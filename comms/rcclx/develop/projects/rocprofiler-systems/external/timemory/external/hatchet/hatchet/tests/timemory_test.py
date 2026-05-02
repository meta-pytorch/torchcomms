# Copyright 2020-2023 The Regents of the University of California, through
# Lawrence Berkeley National Laboratory, and other Hatchet Project Developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from hatchet import GraphFrame
from hatchet.external.console import ConsoleRenderer

import pytest

timemory_avail = True
try:
    import timemory
except ImportError:
    timemory_avail = False


@pytest.mark.skipif(not timemory_avail, reason="timemory package not available")
def test_graphframe(timemory_json_data):
    """Sanity test a GraphFrame object with known data."""
    from timemory.component import WallClock

    wc_s = WallClock.id()  # string identifier
    wc_v = WallClock.index()  # enumeration id
    gf = GraphFrame.from_timemory(timemory_json_data, [wc_s])

    assert len(gf.dataframe) == timemory.size([wc_v])[wc_v]

    for col in gf.dataframe.columns:
        if col in ("sum.wall_clock.inc", "sum.wall_clock"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("nid", "rank"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "node"):
            assert gf.dataframe[col].dtype == object


@pytest.mark.skipif(not timemory_avail, reason="timemory package not available")
def test_tree(timemory_json_data):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_timemory(timemory_json_data)

    print(gf.tree("sum.wall_clock"))

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="sum.wall_clock",
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

    print(output)

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf.graph.roots,
        gf.dataframe,
        metric_column="sum.wall_clock",
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

    print(output)


@pytest.mark.skipif(not timemory_avail, reason="timemory package not available")
def test_graphframe_to_literal(timemory_json_data):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_timemory(timemory_json_data)
    graph_literal = gf.to_literal()

    assert len(graph_literal) == len(gf.graph.roots)


@pytest.mark.skipif(not timemory_avail, reason="timemory package not available")
def test_default_metric(timemory_json_data):
    """Validation test for GraphFrame object using default metric field"""
    gf = GraphFrame.from_timemory(timemory_json_data)

    for func in ["tree", "to_dot", "to_flamegraph"]:
        lhs = "{}".format(getattr(gf, func)(gf.default_metric))
        rhs = "{}".format(getattr(gf, func)())
        assert lhs == rhs
