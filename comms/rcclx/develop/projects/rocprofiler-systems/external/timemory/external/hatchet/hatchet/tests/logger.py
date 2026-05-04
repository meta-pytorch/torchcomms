# Copyright 2022-2023 The University of Arizona and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import json
import glob
import pytest

from hatchet import GraphFrame
from hatchet.util.logger import Logger


def test_no_logging_by_default(calc_pi_hpct_db, lulesh_caliper_json):
    # test HPCTOOLKIT
    GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    assert Logger._active is False

    # test Caliper
    GraphFrame.from_caliper(str(lulesh_caliper_json))
    assert Logger._active is False


@pytest.mark.xfail(reason="Allow this to fail until global configuration PR is merged.")
def test_output(calc_pi_hpct_db):
    functions = [
        "from_hpctoolkit",
        "copy",
        "deepcopy",
        "filter",
        "squash",
        "to_dot",
        "to_flamegraph",
        "to_literal",
        "add",
        "sub",
        "mul",
    ]
    log_lines = []

    gf = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    assert Logger._active is True

    gf2 = gf.copy()
    gf3 = gf2.deepcopy()
    filtered = gf3.filter(lambda x: x["time"] > 0.1, squash=False)
    filtered.squash()
    gf.to_dot()
    gf.to_flamegraph()
    gf.to_literal()

    gf3 = gf + gf2
    gf3 = gf - gf2
    gf3 = gf * gf2

    path = "hatchet_*.log"
    files = glob.glob(path)
    assert len(files) == 1

    with open(files[0], "r") as f:
        line = f.readline()
        while line:
            log_lines.append(json.loads(line))
            line = f.readline()

    for line in log_lines:
        assert line["function"] in functions

    Logger.set_inactive()
    os.remove(files[0])
