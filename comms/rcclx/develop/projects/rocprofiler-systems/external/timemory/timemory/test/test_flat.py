#!@PYTHON_EXECUTABLE@
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from __future__ import absolute_import

__author__ = "Muhammad Haseeb"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Muhammad Haseeb"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

try:
    import mpi4py  # noqa: F401
    from mpi4py import MPI  # noqa: F401
except ImportError:
    pass

import os
import time
import json
import random
import unittest
import timemory as tim
from timemory.profiler import profile, config
from timemory.bundle import marker

# --------------------------- test setup variables ----------------------------------- #


# --------------------------- helper functions ----------------------------------------- #
# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n - 1) + fibonacci(n - 2))


def fib(n, instr):
    if instr is True:
        with marker(components=["wall_clock", "current_peak_rss"], key="fib"):
            return n if n < 2 else (fib(n - 1, True) + fib(n - 2, False))
    else:
        return n if n < 2 else (fibonacci(n - 1) + fibonacci(n - 2))


# sleep for n millisec
def do_sleep(n):
    time.sleep(n * 1e-3)


# cpu utilization for n millisec
def consume(n):
    # Python 3.7 has time_ns()
    try:
        # get current time in nsec
        now = time.time_ns()
        # try until time point
        while time.time_ns() < (now + (n * 1e6)):
            pass
    except AttributeError:
        now = 1000 * time.time()
        # try until time point
        while (1000 * time.time()) < (now + n):
            pass


# get a random entry
def random_entry(vect):
    random.seed(int(1.0e9 * time.time()))
    num = int(random.uniform(0, len(vect) - 1))
    return vect[num]


# get auto_tuple config
def get_config(items=["wall_clock"]):
    return [getattr(tim.component, x) for x in items]


# -------------------------- Thottle Tests set ---------------------------------------- #
# Flat tests class
class TimemoryFlatTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        # set up environment variables
        tim.settings.verbose = 1
        tim.settings.debug = False
        tim.settings.json_output = True
        tim.settings.mpi_thread = False
        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False

    def setUp(self):
        tim.settings.flat_profile = True
        config.include_internal = True

    def tearDown(self):
        tim.settings.flat_profile = False
        config.include_internal = False

    # Tear down class: finalize
    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test profiler_depth
    def test_parse(self):
        """parse"""

        tim.settings.flat_profile = True
        self.assertTrue(tim.settings.flat_profile)

        tim.settings.flat_profile = False
        self.assertFalse(tim.settings.flat_profile)

        os.environ["TIMEMORY_FLAT_PROFILE"] = "ON"
        tim.settings.parse()

        ret = os.environ.get("TIMEMORY_FLAT_PROFILE")

        self.assertTrue(ret)
        self.assertTrue(tim.settings.flat_profile)

        # reset
        tim.settings.flat_profile = False
        os.environ["TIMEMORY_FLAT_PROFILE"] = "OFF"

    # ---------------------------------------------------------------------------------- #
    # test profiler_depth
    def test_no_flat(self):
        """not_flat"""
        n = 10
        tim.settings.flat_profile = False
        with marker(
            components=["wall_clock", "cpu_clock"], key=self.shortDescription()
        ):
            fib(n, True)

        print(
            "{}\n".format(
                json.dumps(tim.get()["timemory"]["wall_clock"], indent=4)
            )
        )
        # inspect data
        data = tim.get()["timemory"]["wall_clock"]["ranks"][0]["graph"]
        self.assertEqual(data[-1]["depth"], n)

    # ---------------------------------------------------------------------------------- #
    # test profiler_depth
    def test_flat(self):
        """flat"""
        n = 10
        with marker(
            components=["wall_clock", "cpu_clock"], key=self.shortDescription()
        ):
            with profile(
                components=["wall_clock", "cpu_clock"],
                flat=True,
                timeline=False,
            ):
                fibonacci(n)

        print(
            "{}\n".format(
                json.dumps(tim.get()["timemory"]["wall_clock"], indent=4)
            )
        )
        # inspect data
        data = tim.get()["timemory"]["wall_clock"]["ranks"][0]["graph"]
        self.assertEqual(data[-1]["depth"], 0)


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
