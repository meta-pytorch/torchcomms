# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import re

from hatchet import GraphFrame
from hatchet.external.console import ConsoleRenderer


def test_graphframe(hatchet_cycle_pstats):
    gf = GraphFrame.from_cprofile(str(hatchet_cycle_pstats))

    assert len(gf.dataframe.groupby("file")) == 4
    assert len(gf.dataframe.groupby("name")) == 9

    gf.dataframe.reset_index(inplace=True)

    for col in gf.dataframe.columns:
        if col in ("time (inc)", "time"):
            assert gf.dataframe[col].dtype == np.float64
        elif col in ("line", "numcalls", "nativecalls"):
            assert gf.dataframe[col].dtype == np.int64
        elif col in ("name", "type", "file", "module", "node"):
            assert gf.dataframe[col].dtype == object


def test_tree(hatchet_cycle_pstats):
    gf = GraphFrame.from_cprofile(str(hatchet_cycle_pstats))

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
    assert "g pstats_reader_test.py" in output
    assert "<method 'disable' ...Profiler' objects> ~" in output

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
    assert "f pstats_reader_test.py" in output
    assert re.match("(.|\n)*recursive(.|\n)*recursive", output)
