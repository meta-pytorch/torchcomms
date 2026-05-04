# Copyright 2020-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import json

import numpy as np

from hatchet import GraphFrame
from hatchet.external.console import ConsoleRenderer


def test_graphframe(hatchet_pyinstrument_json):
    """Sanity test a GraphFrame object with known data."""

    def test_children(child_dict, gf):
        df = gf.dataframe.loc[
            (gf.dataframe["name"] == child_dict["function"])
            & (gf.dataframe["file"] == child_dict["file_path_short"])
            & (gf.dataframe["line"] == child_dict["line_no"])
            & (gf.dataframe["time (inc)"] == child_dict["time"])
            & (gf.dataframe["is_application_code"] == child_dict["is_application_code"])
        ]

        assert len(df.index) == 1
        assert df.index[0].frame["name"] == child_dict["function"]

        child_number = 0
        if "children" in child_dict:
            for child in child_dict["children"]:
                # to calculate the number of children
                child_number += 1
                test_children(child, gf)

        # number of children should be the same.
        assert len(df.index[0].children) == child_number

    # create a graphframe using from_pyinstrument
    gf = GraphFrame.from_pyinstrument(str(hatchet_pyinstrument_json))

    graph_dict = []
    # read directly from the input file to compare it with the graphframe.
    with open(str(hatchet_pyinstrument_json)) as pyinstrument_json:
        graph_dict = json.load(pyinstrument_json)

        # roots should be the same
        assert graph_dict["root_frame"]["function"] == gf.graph.roots[0].frame["name"]

        child_number = 0
        if "children" in graph_dict["root_frame"]:
            for child in graph_dict["root_frame"]["children"]:
                # to calculate the number of children
                child_number += 1
                test_children(child, gf)

        # number of children should be the same.
        assert len(gf.graph.roots[0].children) == child_number

    assert len(gf.dataframe.groupby("name")) == 44

    gf.dataframe.reset_index(inplace=True)

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("line"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("is_application_code"):
            assert gf.dataframe[col].dtype == bool
        elif col in ("name", "type", "file", "node"):
            assert gf.dataframe[col].dtype == object


def test_tree(hatchet_pyinstrument_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_pyinstrument(str(hatchet_pyinstrument_json))

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
    assert "0.000 <module> examples.py" in output
    assert "0.025 read hatchet/readers/caliper_reader.py" in output

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
    assert "0.478 <module> examples.py" in output
    assert "0.063 from_caliper_json hatchet/graphframe.py" in output


def test_graphframe_to_literal(hatchet_pyinstrument_json):
    """Sanity test a GraphFrame object with known data."""
    gf = GraphFrame.from_pyinstrument(str(hatchet_pyinstrument_json))
    graph_literal = gf.to_literal()

    gf2 = GraphFrame.from_literal(graph_literal)

    assert len(gf.graph) == len(gf2.graph)
