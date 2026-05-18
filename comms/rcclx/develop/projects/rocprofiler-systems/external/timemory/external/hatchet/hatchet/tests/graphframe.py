# -*- coding: utf-8 -*-

# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from __future__ import division

import os

import pytest

import numpy as np
import pandas as pd

from hatchet import GraphFrame, QueryMatcher
from hatchet.graphframe import InvalidFilter, EmptyFilter
from hatchet.frame import Frame
from hatchet.graph import Graph
from hatchet.node import Node
from hatchet.external.console import ConsoleRenderer


def test_copy(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    other = gf.copy()

    assert gf.graph is other.graph
    assert gf.dataframe is not other.dataframe
    assert gf.dataframe.equals(other.dataframe)
    assert gf.inc_metrics == other.inc_metrics
    assert gf.exc_metrics == other.exc_metrics
    assert gf.default_metric == other.default_metric
    assert gf.metadata == other.metadata


def test_deepcopy(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    other = gf.deepcopy()

    assert gf.graph == other.graph
    assert gf.dataframe is not other.dataframe
    assert gf.inc_metrics == other.inc_metrics
    assert gf.exc_metrics == other.exc_metrics
    assert gf.default_metric == other.default_metric
    assert gf.metadata == other.metadata


def test_drop_index_levels(calc_pi_hpct_db):
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    num_nodes = len(gf.graph)

    gf.drop_index_levels()
    num_rows = len(gf.dataframe.index)

    assert num_nodes == num_rows


def test_unify_hpctoolkit_data(calc_pi_hpct_db):
    gf1 = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    gf2 = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

    assert gf1.graph is not gf2.graph
    # indexes are the same since we are reading in the same dataset
    assert all(gf1.dataframe.index == gf2.dataframe.index)

    gf1.unify(gf2)

    assert gf1.graph is gf2.graph

    # Indexes should still be the same after unify. Sort indexes before comparing.
    gf1.dataframe.sort_index(inplace=True)
    gf2.dataframe.sort_index(inplace=True)
    assert all(gf1.dataframe.index == gf2.dataframe.index)


def test_invalid_constructor():
    # bad Graph
    with pytest.raises(ValueError):
        GraphFrame(None, None, None, None)

    # bad dataframe
    with pytest.raises(ValueError):
        GraphFrame(Graph([]), None, None, None)

    # dataframe has no "node" index
    with pytest.raises(ValueError):
        GraphFrame(Graph([]), pd.DataFrame(), None, None)


def test_from_lists():
    gf = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))
    gf.dataframe.reset_index(inplace=True, drop=False)

    assert list(gf.graph.traverse(attrs="name")) == ["a", "b", "c", "d", "e"]

    assert all(gf.dataframe["time"] == ([1.0] * 5))

    nodes = set(gf.graph.traverse())
    assert all(node in list(gf.dataframe["node"]) for node in nodes)


def test_calculate_inclusive_metrics():
    gf = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))
    (a, b, c, d, e) = gf.graph.traverse()

    # this is computed automatically by from_lists -- drop it for this test.
    del gf.dataframe["time (inc)"]

    gf.calculate_inclusive_metrics()
    assert gf.dataframe.loc[a, "time (inc)"] == 5
    assert gf.dataframe.loc[b, "time (inc)"] == 2
    assert gf.dataframe.loc[c, "time (inc)"] == 1
    assert gf.dataframe.loc[d, "time (inc)"] == 2
    assert gf.dataframe.loc[e, "time (inc)"] == 1


def test_calculate_exclusive_metrics(calc_pi_hpct_db):
    # test without multiindex.
    gf_drop = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    gf_drop.drop_index_levels()
    gf_drop2 = gf_drop.deepcopy()
    # create another column to set as a ground truth.
    gf_drop.dataframe["time (ground)"] = gf_drop.dataframe["time"]
    # drop "time" and remove from exc_metrics
    gf_drop.dataframe.drop(columns="time", axis=1, inplace=True)
    gf_drop.exc_metrics.remove("time")
    gf_drop.calculate_exclusive_metrics(columns="time (inc)")
    # time should be created
    assert list(gf_drop.dataframe.columns).count("time") == 1
    assert gf_drop.exc_metrics.count("time") == 1
    # check if new created "time" column matches with the grount truth
    assert gf_drop.dataframe["time"].equals(gf_drop.dataframe["time (ground)"])

    # create another column to set as a ground truth.
    gf_drop2.dataframe["time (ground)"] = gf_drop2.dataframe["time"]
    gf_drop2.calculate_exclusive_metrics(columns="time (inc)")
    # time should not be created again
    assert list(gf_drop2.dataframe.columns).count("time") == 1
    assert gf_drop2.exc_metrics.count("time") == 1
    # check if new created "time" column matches with the grount truth
    assert gf_drop2.dataframe["time"].equals(gf_drop2.dataframe["time (ground)"])

    # test 1D multiindex.
    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    # create another column to set as a ground truth.
    gf.dataframe["time (ground)"] = gf.dataframe["time"]
    # drop "time" and remove from exc_metrics
    gf.dataframe.drop(columns="time", axis=1, inplace=True)
    gf.exc_metrics.remove("time")
    gf.calculate_exclusive_metrics(columns="time (inc)")

    # check if new created "time" column matches with the grount truth
    assert gf.dataframe["time"].equals(gf.dataframe["time (ground)"])
    # time should be created
    assert list(gf.dataframe.columns).count("time") == 1
    assert gf.exc_metrics.count("time") == 1

    # test 2D multiindex.
    gf_dummy_2D = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    gf_dummy_2D.dataframe = gf_dummy_2D.dataframe.reset_index()
    gf_dummy_2D.dataframe["thread"] = 0
    gf_dummy_2D.dataframe.set_index(["node", "rank", "thread"], inplace=True)

    # create another column to set as a ground truth.
    gf_dummy_2D.dataframe["time (ground)"] = gf_dummy_2D.dataframe["time"]
    # drop "time" and remove from exc_metrics
    gf_dummy_2D.dataframe.drop(columns="time", axis=1, inplace=True)
    gf_dummy_2D.exc_metrics.remove("time")
    gf_dummy_2D.calculate_exclusive_metrics(columns="time (inc)")

    # check if new created "time" column matches with the grount truth
    assert gf_dummy_2D.dataframe["time"].equals(gf_dummy_2D.dataframe["time (ground)"])
    # time should be created
    assert list(gf_dummy_2D.dataframe.columns).count("time") == 1
    assert gf_dummy_2D.exc_metrics.count("time") == 1

    # test without giving parameters to create_exclusive_columns()
    gf_no_params = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    # create another column to set as a ground truth.
    gf_no_params.dataframe["time (ground)"] = gf_no_params.dataframe["time"]
    # drop "time" and remove from exc_metrics
    gf_no_params.dataframe.drop(columns="time", axis=1, inplace=True)
    gf_no_params.exc_metrics.remove("time")
    gf_no_params.calculate_exclusive_metrics()

    assert gf_no_params.dataframe["time"].equals(
        gf_no_params.dataframe["time (ground)"]
    )
    # time should be created
    assert list(gf_no_params.dataframe.columns).count("time") == 1
    assert gf_no_params.exc_metrics.count("time") == 1


def test_subtree_sum_value_error():
    gf = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))

    # in and out columns with different lengths
    with pytest.raises(ValueError):
        gf.subtree_sum(["time"], [])


def test_subtree_sum_inplace():
    gf = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))
    (a, b, c, d, e) = gf.graph.traverse()

    gf.subtree_sum(["time"])
    assert gf.dataframe.loc[a, "time"] == 5
    assert gf.dataframe.loc[b, "time"] == 2
    assert gf.dataframe.loc[c, "time"] == 1
    assert gf.dataframe.loc[d, "time"] == 2
    assert gf.dataframe.loc[e, "time"] == 1


def test_subtree_product():
    gf = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))
    (a, b, c, d, e) = gf.graph.traverse()

    gf.dataframe["time2"] = gf.dataframe["time"] * 2

    gf.subtree_sum(["time", "time2"], ["out", "out2"], function=np.prod)
    assert gf.dataframe.loc[a, "out"] == 1
    assert gf.dataframe.loc[b, "out"] == 1
    assert gf.dataframe.loc[c, "out"] == 1
    assert gf.dataframe.loc[d, "out"] == 1
    assert gf.dataframe.loc[e, "out"] == 1

    assert gf.dataframe.loc[a, "out2"] == 32
    assert gf.dataframe.loc[b, "out2"] == 4
    assert gf.dataframe.loc[c, "out2"] == 2
    assert gf.dataframe.loc[d, "out2"] == 4
    assert gf.dataframe.loc[e, "out2"] == 2


def check_filter_no_squash(gf, filter_func, num_rows):
    """Ensure filtering and squashing results in the right Graph and GraphFrame."""

    # sequential tests
    orig_graph = gf.graph.copy()
    filtered = gf.filter(filter_func, squash=False, num_procs=1)
    filtered.dataframe.reset_index(inplace=True)

    assert filtered.graph is gf.graph
    assert filtered.graph == orig_graph
    assert len(filtered.dataframe) == num_rows

    # parallel versions of the same test
    orig_graph = gf.graph.copy()
    filtered = gf.filter(filter_func, squash=False)
    filtered.dataframe.reset_index(inplace=True)

    assert filtered.graph is gf.graph
    assert filtered.graph == orig_graph
    assert len(filtered.dataframe) == num_rows


def check_filter_squash(gf, filter_func, expected_graph, expected_inc_time):
    """Ensure filtering and squashing results in the right Graph and GraphFrame."""

    # sequential tests
    filtered_squashed = gf.filter(filter_func, num_procs=1)
    index_names = filtered_squashed.dataframe.index.names
    filtered_squashed.dataframe.reset_index(inplace=True)

    assert filtered_squashed.graph is not gf.graph
    assert all(
        n in filtered_squashed.graph.traverse()
        for n in filtered_squashed.dataframe["node"]
    )
    filtered_squashed.dataframe.set_index(index_names, inplace=True)

    filtered_squashed.dataframe.reset_index(inplace=True, drop=False)
    assert filtered_squashed.graph == expected_graph
    assert len(filtered_squashed.dataframe.index) == len(expected_graph)
    filtered_squashed_node_names = list(expected_graph.traverse(attrs="name"))
    assert all(
        n.frame["name"] in filtered_squashed_node_names
        for n in filtered_squashed.dataframe["node"]
    )
    filtered_squashed.dataframe.set_index(index_names, inplace=True)

    # verify inclusive metrics at different nodes
    nodes = list(filtered_squashed.graph.traverse())
    assert len(nodes) == len(expected_inc_time)

    assert expected_inc_time == [
        filtered_squashed.dataframe.loc[node, "time (inc)"] for node in nodes
    ]

    # parallel versions
    filtered_squashed = gf.filter(filter_func)
    index_names = filtered_squashed.dataframe.index.names
    filtered_squashed.dataframe.reset_index(inplace=True)

    assert filtered_squashed.graph is not gf.graph
    assert all(
        n in filtered_squashed.graph.traverse()
        for n in filtered_squashed.dataframe["node"]
    )
    filtered_squashed.dataframe.set_index(index_names, inplace=True)

    filtered_squashed.dataframe.reset_index(inplace=True, drop=False)
    assert filtered_squashed.graph == expected_graph
    assert len(filtered_squashed.dataframe.index) == len(expected_graph)
    filtered_squashed_node_names = list(expected_graph.traverse(attrs="name"))
    assert all(
        n.frame["name"] in filtered_squashed_node_names
        for n in filtered_squashed.dataframe["node"]
    )
    filtered_squashed.dataframe.set_index(index_names, inplace=True)

    # verify inclusive metrics at different nodes
    nodes = list(filtered_squashed.graph.traverse())
    assert len(nodes) == len(expected_inc_time)

    assert expected_inc_time == [
        filtered_squashed.dataframe.loc[node, "time (inc)"] for node in nodes
    ]


def test_filter_squash():
    r"""Test squash on a simple tree with one root.

          a
         / \      remove bd     a
        b   d    ---------->   / \
       /      \               c   e
      c        e

    """
    check_filter_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "e"))),
        lambda row: row["node"].frame["name"] in ("a", "c", "e"),
        Graph.from_lists(("a", "c", "e")),
        [3, 1, 1],  # a, c, e
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "e"))),
        lambda row: row["node"].frame["name"] in ("a", "c", "e"),
        3,  # a, c, e
    )


def test_filter_squash_with_merge():
    r"""Test squash with a simple node merge.

          a
         / \      remove bd     a
        b   d    ---------->    |
       /      \                 c
      c        c

    Note that here, because b and d have been removed, a will have only
    one child called c, which will contain merged (summed) data from the
    original c rows.

    """
    check_filter_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "c"))),
        lambda row: row["node"].frame["name"] in ("a", "c"),
        Graph.from_lists(("a", "c")),
        [3, 2],  # a, c
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "c"))),
        lambda row: row["node"].frame["name"] in ("a", "c"),
        3,  # a, c, c
    )


def test_filter_squash_with_rootless_merge():
    r"""Test squash on a simple tree with several rootless node merges.

               a
          ___/ | \___     remove abcd
         b     c     d   ------------>  e f g
        /|\   /|\   /|\
       e f g e f g e f g

    Note that here, because b and d have been removed, a will have only
    one child called c, which will contain merged (summed) data from the
    original c rows.

    """
    check_filter_squash(
        GraphFrame.from_lists(
            ("a", ("b", "e", "f", "g"), ("c", "e", "f", "g"), ("d", "e", "f", "g"))
        ),
        lambda row: row["node"].frame["name"] not in ("a", "b", "c", "d"),
        Graph.from_lists(["e"], ["f"], ["g"]),
        [3, 3, 3],  # e, f, g
    )

    check_filter_no_squash(
        GraphFrame.from_lists(
            ("a", ("b", "e", "f", "g"), ("c", "e", "f", "g"), ("d", "e", "f", "g"))
        ),
        lambda row: row["node"].frame["name"] not in ("a", "b", "c", "d"),
        9,  # e, f, g, e, f, g, e, f, g
    )


def test_filter_squash_different_roots():
    r"""Test squash on a simple tree with one root but make multiple roots.

          a
         / \      remove a     b  d
        b   d    --------->   /    \
       /      \              c      e
      c        e

    """
    check_filter_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "e"))),
        lambda row: row["node"].frame["name"] != "a",
        Graph.from_lists(("b", "c"), ("d", "e")),
        [2, 1, 2, 1],  # b, c, d, e
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("a", ("b", "c"), ("d", "e"))),
        lambda row: row["node"].frame["name"] != "a",
        4,  # b, c, d, e
    )


def test_filter_squash_diamond():
    r"""Test that diamond edges are collapsed when squashing.

    Ensure we can handle the most basic DAG.

            a
           / \      remove bc     a
          b   c    ---------->    |
           \ /                    d
            d

    """
    d = Node(Frame(name="d"))
    check_filter_squash(
        GraphFrame.from_lists(("a", ("b", d), ("c", d))),
        lambda row: row["node"].frame["name"] not in ("b", "c"),
        Graph.from_lists(("a", "d")),
        [2, 1],  # a, d
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("a", ("b", d), ("c", d))),
        lambda row: row["node"].frame["name"] not in ("b", "c"),
        2,  # a, d
    )


def test_filter_squash_bunny():
    r"""Test squash on a complicated "bunny" shaped graph.

    This has multiple roots as well as multiple parents that themselves
    have parents.

          e   g
         / \ / \
        f   a   h    remove abc     e   g
           / \      ----------->   / \ / \
          b   c                   f   d   h
           \ /
            d

    """
    d = Node(Frame(name="d"))
    diamond = Node.from_lists(("a", ("b", d), ("c", d)))

    new_d = Node(Frame(name="d"))

    check_filter_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "b", "c"),
        Graph.from_lists(("e", new_d, "f"), ("g", new_d, "h")),
        [3, 1, 1, 3, 1],  # e, d, f, g, h
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "b", "c"),
        5,  # e, d, f, g, h
    )


def test_filter_squash_bunny_to_goat():
    r"""Test squash on a "bunny" shaped graph:

    This one is more complex because there are more transitive edges to
    maintain between the roots (e, g) and b and c.

          e   g                     e   g
         / \ / \                   /|\ /|\
        f   a   h    remove ac    f | b | h
           / \      ---------->     | | |
          b   c                      \|/
           \ /                        d
            d

    """
    d = Node(Frame(name="d"))
    diamond = Node.from_lists(("a", ("b", d), ("c", d)))

    new_d = Node(Frame(name="d"))
    new_b = Node.from_lists(("b", new_d))

    check_filter_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "c"),
        Graph.from_lists(("e", new_b, new_d, "f"), ("g", new_b, new_d, "h")),
        [4, 2, 1, 1, 4, 1],  # e, b, d, f, g, h
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "c"),
        6,  # e, b, d, f, g, h
    )


@pytest.mark.xfail(reason="Hatchet does not yet handle merging with parents properly.")
def test_filter_squash_bunny_to_goat_with_merge():
    r"""Test squash on a "bunny" shaped graph:

    This one is more complex because there are more transitive edges to
    maintain between the roots (e, g) and b and c.

          e   g
         / \ / \
        f   a   h    remove ac      e   g
           / \      ---------->    / \ / \
          b   c                   f   b   h
           \ /
            b

    """
    b = Node(Frame(name="b"))
    diamond = Node.from_lists(("a", ("b", b), ("c", b)))

    new_b = Node(Frame(name="b"))

    check_filter_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "c"),
        Graph.from_lists(("e", new_b, "f"), ("g", new_b, "h")),
        [4, 2, 1, 4, 1],  # e, b, f, g, h
    )

    check_filter_no_squash(
        GraphFrame.from_lists(("e", "f", diamond), ("g", diamond, "h")),
        lambda row: row["node"].frame["name"] not in ("a", "c"),
        5,  # e, b, f, g, h
    )


def test_filter_no_squash_mock_literal(mock_graph_literal):
    """Test the squash operation with a foo-bar tree."""
    gf = GraphFrame.from_literal(mock_graph_literal)
    nodes = list(gf.graph.traverse())
    assert not all(gf.dataframe.loc[nodes, "time"] > 5.0)
    filtered_gf = gf.filter(lambda x: x["time"] > 5.0, squash=False)
    assert filtered_gf.graph is gf.graph
    filtered_gf.dataframe.reset_index(drop=False, inplace=True)
    assert all(n in filtered_gf.graph.traverse() for n in filtered_gf.dataframe["node"])


def test_filter_squash_mock_literal(mock_graph_literal):
    """Test the squash operation with a foo-bar tree."""
    gf = GraphFrame.from_literal(mock_graph_literal)

    nodes = list(gf.graph.traverse())
    assert not all(gf.dataframe.loc[nodes, "time"] > 5.0)
    filtered_squashed_gf = gf.filter(lambda x: x["time"] > 5.0, squash=True)

    filtered_squashed_nodes = list(filtered_squashed_gf.graph.traverse())
    assert all(
        filtered_squashed_gf.dataframe.loc[filtered_squashed_nodes, "time"] > 5.0
    )
    assert len(filtered_squashed_gf.graph) == 7


def test_filter_no_squash_mock_literal_multi_subtree_merge(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    gf.drop_index_levels()
    filtlist = [1, 3, 7, 9, 21, 23]
    filtered_gf = gf.filter(lambda x: x["node"]._hatchet_nid in filtlist, squash=False)
    assert filtered_gf.graph is gf.graph
    filtered_gf.dataframe.reset_index(drop=False, inplace=True)
    assert all(n in filtered_gf.graph.traverse() for n in filtered_gf.dataframe["node"])


def test_filter_squash_mock_literal_multi_subtree_merge(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    gf.drop_index_levels()
    filtlist = [1, 3, 7, 9, 21, 23]
    filtered_squashed_gf = gf.filter(
        lambda x: x["node"]._hatchet_nid in filtlist, squash=True
    )
    filtered_squashed_nodes = list(filtered_squashed_gf.graph.traverse())
    assert len(filtered_squashed_gf.graph) == 2
    assert all(
        [
            n.frame.attrs["name"] == "bar" or n.frame.attrs["name"] == "grault"
            for n in filtered_squashed_nodes
        ]
    )


def test_filter_query_no_squash_high_level(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    path = [
        {"time (inc)": ">= 30.0"},
        (2, {"name": "[^b][a-z]+"}),
        ("*", {"name": "[^b][a-z]+"}),
        {"name": "gr[a-z]+"},
    ]
    filtered_gf = gf.filter(path, squash=False)
    assert filtered_gf.graph is gf.graph
    filtered_gf.dataframe.reset_index(drop=False, inplace=True)
    assert all(n in filtered_gf.graph.traverse() for n in filtered_gf.dataframe["node"])


def test_filter_query_squash_high_level(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    path = [
        {"time (inc)": ">= 30.0"},
        (2, {"name": "[^b][a-z]+"}),
        ("*", {"name": "[^b][a-z]+"}),
        {"name": "gr[a-z]+"},
    ]
    root = gf.graph.roots[0]
    match = list(
        set(
            [
                root,
                root.children[1],
                root.children[1].children[0],
                root.children[1].children[0].children[0],
                root.children[1].children[0].children[0].children[1],
            ]
        )
    )
    filtered_squashed_gf = gf.filter(path, squash=True)
    filtered_squashed_nodes = list(filtered_squashed_gf.graph.traverse())
    assert len(filtered_squashed_nodes) == len(match)
    assert (
        (
            filtered_squashed_gf.dataframe.loc[filtered_squashed_nodes, "time (inc)"]
            >= 30.0
        )
        | (
            ~filtered_squashed_gf.dataframe.loc[
                filtered_squashed_nodes, "name"
            ].str.startswith("b")
        )
        | (
            filtered_squashed_gf.dataframe.loc[
                filtered_squashed_nodes, "name"
            ].str.startswith("gr")
        )
    ).all()


def test_filter_query_no_squash_low_level(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

    def time_filt(row):
        return row["time (inc)"] >= 30.0

    def no_b_filt(row):
        return not row["name"].startswith("b")

    def gr_name_filt(row):
        return row["name"].startswith("gr")

    query = (
        QueryMatcher()
        .match(".", time_filt)
        .rel(2, no_b_filt)
        .rel("*", no_b_filt)
        .rel(".", gr_name_filt)
    )
    filtered_gf = gf.filter(query, squash=False)
    filtered_gf.dataframe.reset_index(drop=False, inplace=True)
    assert filtered_gf.graph is gf.graph
    assert all(n in filtered_gf.graph.traverse() for n in filtered_gf.dataframe["node"])


def test_filter_query_squash_low_level(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

    def time_filt(row):
        return row["time (inc)"] >= 30.0

    def no_b_filt(row):
        return not row["name"].startswith("b")

    def gr_name_filt(row):
        return row["name"].startswith("gr")

    query = (
        QueryMatcher()
        .match(".", time_filt)
        .rel(2, no_b_filt)
        .rel("*", no_b_filt)
        .rel(".", gr_name_filt)
    )
    root = gf.graph.roots[0]
    match = list(
        set(
            [
                root,
                root.children[1],
                root.children[1].children[0],
                root.children[1].children[0].children[0],
                root.children[1].children[0].children[0].children[1],
            ]
        )
    )
    filtered_squashed_gf = gf.filter(query, squash=True)
    filtered_squashed_nodes = list(filtered_squashed_gf.graph.traverse())
    assert len(filtered_squashed_nodes) == len(match)
    assert (
        (
            filtered_squashed_gf.dataframe.loc[filtered_squashed_nodes, "time (inc)"]
            >= 30.0
        )
        | (
            ~filtered_squashed_gf.dataframe.loc[
                filtered_squashed_nodes, "name"
            ].str.startswith("b")
        )
        | (
            filtered_squashed_gf.dataframe.loc[
                filtered_squashed_nodes, "name"
            ].str.startswith("gr")
        )
    ).all()


def test_filter_bad_argument(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    fake_filter = {"bad": "filter"}
    with pytest.raises(InvalidFilter):
        gf.filter(fake_filter, squash=False)


def test_filter_emtpy_graphframe(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    empty_filter = [
        {"name": "waldo"},
        "+",
        {"time (inc)": ">= 20.0"},
        "+",
        {"time (inc)": 7.5, "time": 7.5},
    ]
    with pytest.raises(EmptyFilter):
        gf.filter(empty_filter, squash=False)


def test_tree(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

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
    assert "0.000 foo" in output
    assert "10.000 waldo" in output
    assert "15.000 garply" in output

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
    assert "55.000 waldo" in output
    assert "15.000 garply" in output


def test_to_dot(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)
    output = gf.to_dot(metric="time")

    # do a simple edge check -- this isn't exhaustive
    for node in gf.graph.traverse():
        for child in node.children:
            assert '"%s" -> "%s"' % (node._hatchet_nid, child._hatchet_nid) in output


def test_unify_diff_graphs():
    gf1 = GraphFrame.from_lists(("a", ("b", "c"), ("d", "e")))
    gf2 = GraphFrame.from_lists(("a", ("b", "c", "d"), ("e", "f"), "g"))

    assert gf1.graph is not gf2.graph

    gf1.unify(gf2)
    assert gf1.graph is gf2.graph

    assert len(gf1.graph) == gf1.dataframe.shape[0]


def test_sub_decorator(small_mock1, small_mock2, small_mock3):
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)
    gf3 = GraphFrame.from_literal(small_mock3)

    assert len(gf1.graph) == 6
    assert len(gf2.graph) == 7

    gf4 = gf1 - gf2

    assert len(gf4.graph) == 8
    assert gf4.dataframe.loc[gf4.dataframe["_missing_node"] == 2].shape[0] == 2  # "R"
    assert gf4.dataframe.loc[gf4.dataframe["_missing_node"] == 1].shape[0] == 1  # "L"
    assert (
        gf4.dataframe.loc[gf4.dataframe["_missing_node"] == 0].shape[0] == 5
    )  # "" or same in both

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf4.graph.roots,
        gf4.dataframe,
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
    assert "0.000 C" in output
    assert "nan D ▶" in output
    assert "10.000 H ◀" in output

    gf5 = gf1 - gf3

    assert len(gf1.graph) == 6
    assert len(gf3.graph) == 4

    assert len(gf5.graph) == 6
    assert gf5.dataframe.loc[gf5.dataframe["_missing_node"] == 2].shape[0] == 0  # "R"
    assert gf5.dataframe.loc[gf5.dataframe["_missing_node"] == 1].shape[0] == 2  # "L"
    assert gf5.dataframe.loc[gf5.dataframe["_missing_node"] == 0].shape[0] == 4  # ""

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf5.graph.roots,
        gf5.dataframe,
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
    assert "15.000 A" in output
    assert "5.000 C ◀" in output
    assert "10.000 H ◀" in output


def test_div_decorator(small_mock1, small_mock2):
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)

    assert len(gf1.graph) == 6
    assert len(gf2.graph) == 7

    gf3 = gf1 / gf2

    assert len(gf3.graph) == 8
    assert gf3.dataframe.loc[gf3.dataframe["_missing_node"] == 2].shape[0] == 2  # "R"
    assert gf3.dataframe.loc[gf3.dataframe["_missing_node"] == 1].shape[0] == 1  # "L"
    assert gf3.dataframe.loc[gf3.dataframe["_missing_node"] == 0].shape[0] == 5  # ""

    output = ConsoleRenderer(unicode=True, color=False).render(
        gf3.graph.roots,
        gf3.dataframe,
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
    assert "1.000 C" in output
    assert "inf B" in output
    assert "nan D ▶" in output
    assert "10.000 H ◀" in output


def test_groupby_aggregate_simple(mock_dag_literal_module):
    r"""Test reindex on a simple graph:

          a                              main
         / \                             /  \
        b   e       groupby module     foo  bar
        |   |       -------------->     |    |
        c   f                         graz  baz

        Node   Module
         a     main
         b     foo
         c     graz
         e     bar
         f     baz

    """
    modules = ["main", "foo", "graz", "bar", "baz"]

    gf = GraphFrame.from_literal(mock_dag_literal_module)

    groupby_col = ["module"]
    agg_func = {"time (inc)": np.max, "time": np.max}
    out_gf = gf.groupby_aggregate(groupby_col, agg_func)

    assert all(m in out_gf.dataframe.name.values for m in modules)
    assert len(out_gf.graph) == len(modules)


def test_groupby_aggregate_complex(mock_dag_literal_module_complex):
    r"""Test reindex on a complex graph:

          a                              main
         / \                             /  \
        b   e       groupby module     foo  bar
        |           -------------->     |
        c                             graz
        |
        d

        Node   Module
         a     main
         b     foo
         c     graz
         d     graz
         e     bar

    """
    modules = ["main", "foo", "graz", "bar"]

    gf = GraphFrame.from_literal(mock_dag_literal_module_complex)

    groupby_col = ["module"]
    agg_func = {"time (inc)": np.sum, "time": np.sum}
    out_gf = gf.groupby_aggregate(groupby_col, agg_func)

    assert all(m in out_gf.dataframe.name.values for m in modules)
    assert len(out_gf.graph) == len(modules)


def test_groupby_aggregate_more_complex(mock_dag_literal_module_more_complex):
    r"""Test reindex on a more complex graph:

          a                              main
         / \                             /  \
        b   e       groupby module     foo--bar
        |   |       -------------->     |
        c   f                         graz
        |
        d

        Node   Module
         a     main
         b     foo
         c     graz
         d     graz
         e     bar
         f     foo

    """
    modules = ["main", "foo", "graz", "bar"]

    gf = GraphFrame.from_literal(mock_dag_literal_module_more_complex)

    groupby_col = ["module"]
    agg_func = {"time (inc)": np.sum, "time": np.sum}
    out_gf = gf.groupby_aggregate(groupby_col, agg_func)

    assert all(m in out_gf.dataframe.name.values for m in modules)
    assert len(out_gf.graph) == len(modules)


def test_depth(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

    nnodes_depth_2 = 0
    max_depth = 0

    # determine max depth in example graph
    # also, count number of nodes at depth 2
    for i, node in enumerate(gf.graph.traverse()):
        if node._depth > max_depth:
            max_depth = node._depth
        if node._depth == 2:
            nnodes_depth_2 += 1

    assert nnodes_depth_2 == 7
    assert max_depth == 5


def test_tree_deprecated_parameters(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

    with pytest.raises(ValueError):
        gf.tree(invert_colors=True)

    with pytest.raises(ValueError):
        gf.tree(name="name")

    with pytest.raises(TypeError):
        gf.tree(metric="time", metric_column="time")


def test_output_with_cycle_graphs():
    r"""Test three output modes on a graph with cycles,
        multiple parents and children.

        a --
       / \ /
      b   c
       \ /
        d
       / \
      e   f
    """

    dot_edges = [
        # d has two parents and two children
        '"1" -> "2";',
        '"5" -> "2";',
        '"2" -> "3";',
        '"2" -> "4";',
        # a -> c -> a cycle
        '"0" -> "5";',
        '"5" -> "0";',
    ]

    a = Node(Frame(name="a"))
    d = Node(Frame(name="d"))
    gf = GraphFrame.from_lists([a, ["b", [d]], ["c", [d, ["e"], ["f"]], [a]]])

    lit_list = gf.to_literal()
    treeout = gf.tree()
    dotout = gf.to_dot()

    # scan through litout produced dictionary for edges
    a_children = [n["frame"]["name"] for n in lit_list[0]["children"]]
    a_c_children = [n["frame"]["name"] for n in lit_list[0]["children"][1]["children"]]
    a_b_children = [n["frame"]["name"] for n in lit_list[0]["children"][0]["children"]]

    assert len(lit_list) == 1
    assert len(a_children) == 2

    # a -> (b,c)
    assert "b" in a_children
    assert "c" in a_children

    # a -> c -> a cycle
    assert "a" in a_c_children

    # d has two parents
    assert "d" in a_c_children
    assert "d" in a_b_children

    # check certain edges are in dot
    for edge in dot_edges:
        assert edge in dotout

    # removing header to prevent it being counted
    treeout = "\n".join(treeout.split("\n")[6:])

    # check that a certain number of occurences
    # of same node are in tree indicating multiple
    # edges
    assert treeout.count("a") == 2
    assert treeout.count("d") == 2
    assert treeout.count("e") == 1
    assert treeout.count("f") == 1


def test_show_metric_columns(mock_graph_literal):
    gf = GraphFrame.from_literal(mock_graph_literal)

    assert sorted(gf.show_metric_columns()) == sorted(["time", "time (inc)"])


def test_to_literal_node_ids():
    r"""Test to_literal and from_literal with ids on a graph with cycles,
        multiple parents and children.

        a --
       / \ /
      b   c
       \ /
        d
       / \
      e   f
    """

    a = Node(Frame(name="a"))
    d = Node(Frame(name="d"))
    gf = GraphFrame.from_lists([a, ["b", [d]], ["c", [d, ["e"], ["f"]], [a]]])
    lit_list = gf.to_literal()

    gf2 = gf.from_literal(lit_list)
    lit_list2 = gf2.to_literal()

    assert lit_list == lit_list2


def test_filter_squash_query_nan_and_inf_metric(small_mock1, small_mock2):
    """Use call path query language on a metric column containing both
    int/float, NaN and inf."""
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)

    gf3 = gf1 / gf2

    query_nan = [{"time": "== np.nan"}]
    filt_nan_gf3 = gf3.filter(query_nan, squash=True)

    assert len(filt_nan_gf3.graph.roots) == 2
    assert all(pd.isnull(time) for time in filt_nan_gf3.dataframe["time (inc)"])
    assert all(pd.isnull(time) for time in filt_nan_gf3.dataframe["time"])
    assert filt_nan_gf3.dataframe.shape[0] == 2
    assert sorted(filt_nan_gf3.dataframe["name"].values) == ["D", "G"]

    query_inf = [{"time": "== np.inf"}]
    filt_inf_gf3 = gf3.filter(query_inf, squash=True)

    assert len(filt_inf_gf3.graph.roots) == 1
    assert all(np.isinf(inc_time) for inc_time in filt_inf_gf3.dataframe["time (inc)"])
    assert all(np.isinf(exc_time) for exc_time in filt_inf_gf3.dataframe["time"])
    assert filt_inf_gf3.dataframe.shape[0] == 1
    assert filt_inf_gf3.dataframe["name"].values[0] == "B"


def test_filter_squash_query_metric_with_nan_and_inf(small_mock1, small_mock2):
    """Use call path query language to match nodes with NaN and inf metric values."""
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)

    gf3 = gf1 / gf2

    query = [{"time": ">= 1"}]
    filter_gf3 = gf3.filter(query, squash=True)

    assert len(filter_gf3.graph.roots) == 3
    assert filter_gf3.dataframe["time"].sum() == np.inf
    assert filter_gf3.dataframe["time (inc)"].sum() == np.inf
    assert filter_gf3.dataframe.shape[0] == 5


def test_filter_nan_and_inf(small_mock1, small_mock2):
    """Use lambda to filter for nodes with NaN and inf values."""
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)

    gf3 = gf1 / gf2

    filt_nan_gf3 = gf3.filter(lambda x: pd.isnull(x["time"]), squash=True)

    assert len(filt_nan_gf3.graph.roots) == 2
    assert all(pd.isnull(inc_time) for inc_time in filt_nan_gf3.dataframe["time (inc)"])
    assert all(pd.isnull(exc_time) for exc_time in filt_nan_gf3.dataframe["time"])
    assert filt_nan_gf3.dataframe.shape[0] == 2
    assert sorted(filt_nan_gf3.dataframe["name"].values) == ["D", "G"]

    filt_inf_gf3 = gf3.filter(lambda x: np.isinf(x["time"]), squash=True)

    assert len(filt_inf_gf3.graph.roots) == 1
    assert all(np.isinf(inc_time) for inc_time in filt_inf_gf3.dataframe["time (inc)"])
    assert all(np.isinf(exc_time) for exc_time in filt_inf_gf3.dataframe["time"])
    assert filt_inf_gf3.dataframe.shape[0] == 1
    assert filt_inf_gf3.dataframe["name"].values == "B"


def test_filter_with_nan_and_inf(small_mock1, small_mock2):
    """Use lambda to filter for metric containing int/float, NaN, and inf values."""
    gf1 = GraphFrame.from_literal(small_mock1)
    gf2 = GraphFrame.from_literal(small_mock2)

    gf3 = gf1 / gf2

    filter_gf3 = gf3.filter(lambda x: x["time"] > 5, squash=True)

    assert len(filter_gf3.graph.roots) == 2
    assert filter_gf3.dataframe["time"].sum() == np.inf
    assert filter_gf3.dataframe["time (inc)"].sum() == np.inf
    assert filter_gf3.dataframe.shape[0] == 2
    assert sorted(filter_gf3.dataframe["name"].values) == ["B", "H"]


def test_inc_metric_only(mock_graph_inc_metric_only):
    """Test graph with only an inclusive metric and no associated exclusive
    metric. A filter-squash should not change the inclusive metric values, and
    the list of exclusive metrics and inclusive metrics should stay the same.
    """
    gf = GraphFrame.from_literal(mock_graph_inc_metric_only)
    filt_gf = gf.filter(lambda x: x["time (inc)"] > 50, squash=True, num_procs=1)

    assert len(filt_gf.graph) == 3
    assert all(filt_gf.dataframe["name"].values == ["A", "E", "H"])
    assert all(filt_gf.dataframe["time (inc)"].values == [130, 55, 55])
    assert gf.inc_metrics == filt_gf.inc_metrics
    assert gf.exc_metrics == filt_gf.exc_metrics


def test_hdf_load_store(mock_graph_literal):
    if os.path.exists("test_gframe.hdf"):
        os.remove("test_gframe.hdf")
    gf_orig = GraphFrame.from_literal(mock_graph_literal)
    gf_orig.to_hdf("test_gframe.hdf", "test_key")
    gf_loaded = GraphFrame.from_hdf("test_gframe.hdf", key="test_key")

    assert gf_orig.dataframe.equals(gf_loaded.dataframe)
    assert gf_orig.graph == gf_loaded.graph

    if os.path.exists("test_gframe.hdf"):
        os.remove("test_gframe.hdf")


def test_update_metadata(mock_graph_literal):
    # initialize test and dummy graphframes
    gf_test = GraphFrame.from_literal(mock_graph_literal)
    gf_dummy = GraphFrame.from_literal(mock_graph_literal)

    # update metadata of dummy graphframe
    gf_dummy.metadata["num_processes"] = 1
    gf_dummy.metadata["num_threads"] = 1
    gf_dummy.metadata["input_hierarchy"] = "function"

    # initialize dict with same "input_hierarchy" metadata
    update_dict = {"input_hierarchy": "function"}

    # update test graphframe using function
    gf_test = gf_test.update_metadata(1, 1, update_dict)

    # check if both graphframes contain the same metadata
    assert gf_test.metadata == gf_dummy.metadata
