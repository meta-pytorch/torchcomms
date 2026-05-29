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
import unittest
import threading
import inspect
import timemory as tim
from timemory import settings as settings

# --------------------------- test setup variables ----------------------------------- #


# --------------------------- helper functions ----------------------------------------- #


# check availability of a component
def check_available(component):
    return inspect.isclass(component)


# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n - 1) + fibonacci(n - 2))


# sleep for n nanosec
def do_sleep(n):
    time.sleep(n * 1e-9)


# cpu utilization for n nanosec
def consume(n):
    # Python 3.7 has time_ns()
    try:
        # get current time in nsec
        now = time.time_ns()
        # try until time point
        while time.time_ns() < (now + n):
            pass
    except AttributeError:
        now = 1e9 * time.time()
        # try until time point
        while (1e9 * time.time()) < (now + n):
            pass


# get auto_tuple config
def get_config(items=["wall_clock"]):
    return [getattr(tim.component, x) for x in items]


# -------------------------- Thottle Tests set ---------------------------------------- #
# Throttle tests class
class TimemoryThrottleTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        # set up environment variables
        os.environ["TIMEMORY_VERBOSE"] = "1"
        os.environ["TIMEMORY_COLLAPSE_THREADS"] = "OFF"

        settings.parse()

        settings.verbose = 1
        settings.debug = False
        settings.json_output = True
        settings.mpi_thread = False
        settings.file_output = True
        settings.dart_output = True
        settings.dart_count = 1
        settings.banner = False

        tim.trace.init("wall_clock", False, "throttle_tests")

        self.nthreads = 1

    # Tear down class: finalize
    @classmethod
    def tearDownClass(self):
        # unset environment variables
        del os.environ["TIMEMORY_VERBOSE"]
        del os.environ["TIMEMORY_COLLAPSE_THREADS"]
        pass

    # ---------------------------------------------------------------------------------- #
    # test expect_true
    def test_expect_true(self):
        """expect_true"""
        settings.debug = False
        n = 2 * settings.throttle_count
        tim.trace.push("true")
        for i in range(n):
            tim.trace.push(self.shortDescription())
            tim.trace.pop(self.shortDescription())
        tim.trace.pop("true")

        self.assertTrue(tim.trace.is_throttled(self.shortDescription()))

    # ---------------------------------------------------------------------------------- #
    # test expect_false
    def test_expect_false(self):
        """expect_false"""
        settings.debug = False
        n = 2 * settings.throttle_count
        v = 2 * settings.throttle_value

        for i in range(n):
            tim.trace.push(self.shortDescription())
            consume(v)
            tim.trace.pop(self.shortDescription())

        self.assertFalse(tim.trace.is_throttled(self.shortDescription()))

    # ---------------------------------------------------------------------------------- #
    def test_region_serial(self):
        """region_serial"""
        settings.debug = False

        def _run(name):
            tim.region.push("rsthread")
            n = 8 * settings.throttle_count
            for i in range(n):
                tim.region.push(name)
                tim.region.pop(name)

            tim.region.pop("rsthread")
            # self.assertTrue(tim.trace.is_throttled(name))

        for i in range(self.nthreads):
            _run(self.shortDescription())

        # print(tim.trace.is_throttled(self.shortDescription()))
        # print(tim.trace.is_throttled("thread"))

    # ---------------------------------------------------------------------------------- #
    # test region_multithreaded
    def test_region_multithreaded(self):
        """region_multithreaded"""
        settings.debug = False

        def _run(name):
            tim.region.push("rthread")
            n = 8 * settings.throttle_count
            for i in range(n):
                tim.region.push(name)
                tim.region.pop(name)

            tim.region.pop("rthread")
            # check assertion
            self.assertTrue(tim.trace.is_throttled(name))

        threads = []

        for i in range(self.nthreads):
            thd = threading.Thread(target=_run, args=(self.shortDescription(),))
            thd.start()
            threads.append(thd)

        for itr in threads:
            itr.join()

    # ---------------------------------------------------------------------------------- #
    # test multithreaded
    def test_multithreaded(self):
        """multithreaded"""
        settings.debug = False

        # _run function
        def _run(name, idx):
            _name = "{}_{}".format(name, idx)
            _hash = tim.add_hash_id(_name)
            n = 2 * settings.throttle_count
            v = 2 * settings.throttle_value
            if idx % 2 == 1:
                for i in range(n):
                    tim.trace.push(_hash)
                    consume(v)
                    tim.trace.pop(_hash)
            else:
                for i in range(n):
                    tim.trace.push(_hash)
                    tim.trace.pop(_hash)

            is_throttled = tim.trace.is_throttled(_name)
            _answer = False if (idx % 2 == 1) else True
            print("thread " + str(idx) + " throttling: " + str(is_throttled))
            self.assertTrue(is_throttled == _answer)

        # thread handles
        threads = []

        # all threads
        tim.trace.push(self.shortDescription())

        # make new threads
        for i in range(self.nthreads):
            thd = threading.Thread(
                target=_run, args=(self.shortDescription(), i)
            )
            thd.start()
            threads.append(thd)

        # wait for join
        for itr in threads:
            itr.join()

        tim.trace.pop(self.shortDescription())


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
