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

import time
import unittest
import timemory as tim
from timemory import settings as settings
from timemory import component as comp


# --------------------------- tolerance variables -------------------------------------- #
# deltas for assert
deltatimer = 0.015
deltautil = 2.5


# --------------------------- helper functions ----------------------------------------- #
# check availability of a component
def check_available(component):
    return getattr(component, "available")


# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n - 1) + fibonacci(n - 2))


# cpu utilization for n milliseconds
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


# -------------------------- Timing Tests set ---------------------------------------- #
# Timing tests class
class TimemoryTimingTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        settings.timing_units = "sec"
        settings.timing_precision = 9
        settings.json_output = True
        tim.init()  # need args? timemory is inited as soon as you import timemory
        settings.dart_output = True
        settings.dart_count = True
        settings.dart_count = 1
        settings.banner = False

    # tear down class: finalize
    @classmethod
    def tearDownClass(self):
        # tim.finalize()
        # tim.dmp.finalize()
        pass

    # ---------------------------------------------------------------------------------- #
    # test wall timer
    def test_wall_clock(self):
        """wall_timer"""
        if not comp.WallClock.available:
            raise unittest.SkipTest("[{}] not available".format("wall_clock"))

        obj = comp.WallClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test monotonic timer
    def test_monotonic_clock(self):
        """monotonic_timer"""
        if not comp.MonotonicClock.available:
            raise unittest.SkipTest(
                "[{}] not available".format("monotonic_clock")
            )

        obj = comp.MonotonicClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test monotonic timer raw
    def test_monotonic_raw_clock(self):
        """monotonic_raw_timer"""
        if not comp.MonotonicRawClock.available:
            raise unittest.SkipTest(
                "[{}] not available".format("monotonic_raw_clock")
            )

        obj = comp.MonotonicRawClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test system timer
    def test_system_clock(self):
        """system_timer"""
        if not comp.SysClock.available:
            raise unittest.SkipTest("[{}] not available".format("sys_clock"))

        obj = comp.SysClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test user timer
    def test_user_clock(self):
        """user_timer"""
        if not comp.UserClock.available:
            raise unittest.SkipTest("[{}] not available".format("user_clock"))

        obj = comp.UserClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test cpu timer
    def test_cpu_clock(self):
        """cpu_timer"""
        if not comp.CpuClock.available:
            raise unittest.SkipTest("[{}] not available".format("cpu_clock"))

        obj = comp.CpuClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test cpu utilization
    def test_cpu_util(self):
        """cpu_utilization"""
        if not comp.CpuUtil.available:
            raise unittest.SkipTest("[{}] not available".format("cpu_util"))

        obj = comp.CpuUtil(self.shortDescription())
        obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get(), delta=deltautil)

    # ---------------------------------------------------------------------------------- #
    # test thread cpu clock
    def test_thread_cpu_clock(self):
        """thread_cpu_timer"""
        if not comp.ThreadCpuClock.available:
            raise unittest.SkipTest(
                "[{}] not available".format("thread_cpu_clock")
            )

        obj = comp.ThreadCpuClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test thread cpu utilization
    def test_thread_cpu_util(self):
        """thread_cpu_utilization"""
        if not comp.ThreadCpuUtil.available:
            raise unittest.SkipTest(
                "[{}] not available".format("thread_cpu_util")
            )

        obj = comp.ThreadCpuUtil(self.shortDescription())
        obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get(), delta=deltautil)

    # ---------------------------------------------------------------------------------- #
    # test process_cpu_clock
    def test_process_cpu_clock(self):
        """process_cpu_timer"""
        if not comp.ProcessCpuClock.available:
            raise unittest.SkipTest(
                "[{}] not available".format("process_cpu_clock")
            )

        obj = comp.ProcessCpuClock(self.shortDescription())
        obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get(), delta=deltatimer)

    # ---------------------------------------------------------------------------------- #
    # test process cpu utilization
    def test_process_cpu_util(self):
        """process_cpu_utilization"""
        if not comp.ProcessCpuUtil.available:
            raise unittest.SkipTest(
                "[{}] not available".format("process_cpu_util")
            )

        obj = comp.ProcessCpuUtil(self.shortDescription())
        obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get(), delta=deltautil)


# ----------------------------- main test runner -------------------------------------- #
# test runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
