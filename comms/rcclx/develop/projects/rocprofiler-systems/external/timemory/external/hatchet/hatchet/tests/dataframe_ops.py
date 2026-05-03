# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from __future__ import division

from hatchet import GraphFrame


def test_filter(mock_graph_literal):
    """Test the filter operation with a foo-bar tree."""
    gf = GraphFrame.from_literal(mock_graph_literal)

    filtered_gf = gf.filter(lambda x: x["time"] > 5.0, squash=False)
    assert len(filtered_gf.dataframe) == 9
    assert all(time > 5.0 for time in filtered_gf.dataframe["time"])

    filtered_gf = gf.filter(lambda x: x["name"].startswith("g"), squash=False)
    assert len(filtered_gf.dataframe) == 7
    assert all(name.startswith("g") for name in filtered_gf.dataframe["name"])


def test_add(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1.add(gf2)

    assert gf3.graph == gf1.graph.union(gf2.graph)

    assert len(gf3.graph) == gf3.dataframe.shape[0]
    assert gf3.dataframe["time"].sum() == 330
    assert gf3.dataframe["time (inc)"].sum() == 1320

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3.add(gf4)
    assert gf5.graph == gf3.graph == gf4.graph


def test_sub(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1.sub(gf2)

    assert gf3.graph == gf1.graph.union(gf2.graph)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    for metric in gf3.exc_metrics + gf3.inc_metrics:
        assert gf3.dataframe[metric].sum() == 0

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3.sub(gf4)
    assert gf5.graph == gf3.graph == gf4.graph


def test_div(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1.div(gf2)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    assert gf3.graph == gf1.graph.union(gf2.graph)

    assert gf3.dataframe["time"].sum() == 21
    assert gf3.dataframe["time (inc)"].sum() == 24

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3.div(gf4)
    assert gf5.graph == gf3.graph == gf4.graph


def test_mul(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1.mul(gf2)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    assert gf3.graph == gf1.graph.union(gf2.graph)

    assert gf3.dataframe["time"].sum() == 1575
    assert gf3.dataframe["time (inc)"].sum() == 37900


def test_add_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1 + gf2

    assert gf3.graph == gf1.graph.union(gf2.graph)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    assert gf3.dataframe["time"].sum() == 330
    assert gf3.dataframe["time (inc)"].sum() == 1320

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3 + gf4
    assert gf5.graph == gf3.graph == gf4.graph

    gf6 = gf1 + gf2 + gf1
    assert gf6.dataframe["time"].sum() == 495

    gf7 = gf1 + gf2
    gf8 = gf7 + gf1
    assert gf8.graph == gf6.graph
    assert gf8.dataframe["time"].sum() == gf6.dataframe["time"].sum()


def test_sub_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1 - gf2

    assert gf3.graph == gf1.graph.union(gf2.graph)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    for metric in gf3.exc_metrics + gf3.inc_metrics:
        assert gf3.dataframe[metric].sum() == 0

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3.sub(gf4)

    assert gf5.graph == gf3.graph == gf4.graph

    gf6 = gf1 - gf2 - gf1
    assert gf6.dataframe["time"].sum() == -165

    gf7 = gf1 - gf2
    gf8 = gf7 - gf1
    assert gf8.graph == gf6.graph
    assert gf8.dataframe["time"].sum() == gf6.dataframe["time"].sum()


def test_div_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf3 = gf1 / gf2

    assert gf3.graph == gf1.graph.union(gf2.graph)
    assert len(gf3.graph) == gf3.dataframe.shape[0]

    assert gf3.dataframe["time"].sum() == 21
    assert gf3.dataframe["time (inc)"].sum() == 24

    gf4 = gf3.copy()
    assert gf4.graph is gf3.graph

    gf5 = gf3 / gf4 / gf3

    assert gf5.graph == gf3.graph == gf4.graph
    assert gf5.dataframe["time (inc)"].sum() == 24

    gf6 = gf3 / gf4
    gf7 = gf6 / gf3
    assert gf7.graph == gf5.graph
    assert gf7.dataframe["time"].sum() == gf5.dataframe["time"].sum()


def test_mul_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)
    gf3 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph is not gf3.graph

    gf4 = gf1 * gf2 * gf3

    assert gf4.graph == gf1.graph.union(gf2.graph.union(gf3.graph))
    assert len(gf4.graph) == gf4.dataframe.shape[0]

    assert gf4.dataframe["time"].sum() == 17625
    assert gf4.dataframe["time (inc)"].sum() == 3397500


def test_iadd_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf1 += gf2

    assert gf1.graph == gf1.graph.union(gf2.graph)
    assert len(gf1.graph) == gf1.dataframe.shape[0]

    assert gf1.dataframe["time"].sum() == 330
    assert gf1.dataframe["time (inc)"].sum() == 1320

    gf3 = gf1.copy()
    assert gf3.graph is gf1.graph

    gf3 += gf1 + gf2 + gf2

    assert gf3.graph == gf1.graph
    assert gf3.dataframe["time"].sum() == 990


def test_isub_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf1 -= gf2

    assert gf1.graph == gf1.graph.union(gf2.graph)
    assert len(gf1.graph) == gf1.dataframe.shape[0]

    for metric in gf1.exc_metrics + gf1.inc_metrics:
        assert gf1.dataframe[metric].sum() == 0

    gf3 = gf1.copy()
    assert gf3.graph is gf1.graph

    gf3 -= gf1

    assert gf3.graph == gf1.graph


def test_idiv_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf1 /= gf2

    assert gf1.graph == gf1.graph.union(gf2.graph)
    assert len(gf1.graph) == gf1.dataframe.shape[0]

    assert gf1.dataframe["time"].sum() == 21
    assert gf1.dataframe["time (inc)"].sum() == 24

    gf3 = gf1.copy()
    assert gf3.graph is gf1.graph

    gf3 /= gf1

    assert gf3.graph == gf1.graph


def test_imul_operator(mock_graph_literal):
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2 = GraphFrame.from_literal(mock_graph_literal)

    assert gf1.graph is not gf2.graph

    gf1 *= gf2

    assert gf1.graph == gf1.graph.union(gf2.graph)
    assert len(gf1.graph) == gf1.dataframe.shape[0]

    assert gf1.dataframe["time"].sum() == 1575
    assert gf1.dataframe["time (inc)"].sum() == 37900
