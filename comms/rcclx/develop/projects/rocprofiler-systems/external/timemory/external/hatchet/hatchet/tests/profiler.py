# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import hatchet as ht
import pstats
import os
from hatchet.util.profiler import Profiler

import pytest
import sys


decorated_calls = [
    "copy",
    "deepcopy",
    "drop_index_levels",
    "filter",
    "squash",
    "tree",
    "to_dot",
    "__add__",
    "__imul__",
    "__isub__",
    "traverse",
    "copy",
    "__len__",
]


def f():
    """Dummy function for profiling"""
    for i in range(1000):
        for j in range(1000):
            i * j


def test_profiler():
    """Test the profiler start/stop, ensure that f() has been profiled"""
    prf = Profiler()

    prf.start()
    f()
    prf.stop()

    sts = pstats.Stats(prf._prf)
    assert "stats" in sts.__dict__

    fn_names = []
    for stat in sts.__dict__["stats"]:
        fn_names.append(stat[2])

    assert "f" in fn_names

    assert isinstance(prf.__str__(), str)
    assert "(f)" in prf.__str__()

    os.remove(prf._output + ".pstats")


def test_write_file():
    """Test that write_to_file, writes a profile to the correct file."""
    prf = Profiler()

    prf.start()
    f()
    prf.stop()

    prf.write_to_file("test.pstats")

    assert os.path.exists("test.pstats")

    sts = pstats.Stats("test.pstats")
    fn_names = []
    for stat in sts.__dict__["stats"]:
        fn_names.append(stat[2])

    assert "f" in fn_names

    os.remove("test.pstats")
    os.remove(prf._output + ".pstats")


@pytest.mark.skipif(sys.version_info > (3, 8), reason="Temporarily allow this to fail.")
def test_profiling_calc_pi(calc_pi_hpct_db):
    """Test debug wrapper as called from hpctoolkit."""
    prf = Profiler()
    output_file = prf._output + ".pstats"

    prf.start()
    gf = ht.GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

    gf.copy()
    gf2 = gf.deepcopy()
    gf.tree()
    gf.to_dot()

    gf3 = gf + gf2
    gf *= gf2
    gf3 -= gf2

    gf.graph.traverse()
    gf.graph.copy()
    len(gf.graph)

    gf2 = gf.filter(lambda x: x["time"] > 0.01)

    gf2.squash()
    gf.drop_index_levels()

    prf.stop()

    assert os.path.exists(output_file)

    sts = pstats.Stats(output_file)

    fn_names = []
    for stat in sts.__dict__["stats"]:
        fn_names.append(stat[2])

    for call in decorated_calls:
        assert call in fn_names

    os.remove(output_file)
