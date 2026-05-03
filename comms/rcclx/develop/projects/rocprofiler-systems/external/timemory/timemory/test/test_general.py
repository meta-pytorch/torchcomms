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

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
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


# --------------------------- General Tests set ---------------------------------------- #
# General tests class
class TimemoryGeneralTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # set up environment variables
        tim.settings.verbose = 1
        tim.settings.debug = False
        tim.settings.tree_output = False
        tim.settings.text_output = True
        tim.settings.cout_output = False
        tim.settings.json_output = True
        tim.settings.flamegraph_output = False
        tim.settings.mpi_thread = False
        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test handling hash identifier access
    def test_hashing(self):
        """hashing"""

        fib_hash = tim.get_hash("fibonacci")
        fib_val = tim.add_hash_id("fibonacci")

        with tim.util.marker(["wall_clock"], key="fibonacci"):
            fibonacci(20)

        fib_id = tim.get_hash_identifier(fib_val)

        self.assertEqual(fib_hash, fib_val)
        self.assertEqual(fib_id, "fibonacci")

        foobar_hash = tim.get_hash("foobar")
        foobar_val = tim.add_hash_id("foobar")
        foobar_id = tim.get_hash_identifier(foobar_val)

        self.assertEqual(foobar_hash, foobar_val)
        self.assertEqual(foobar_id, "foobar")

        print(f"fib_hash    : {fib_hash}")
        print(f"fib_val     : {fib_val}")
        print(f"fib_id      : {fib_id}")
        print(f"foobar_hash : {foobar_hash}")
        print(f"foobar_val  : {foobar_val}")
        print(f"foobar_id   : {foobar_id}")


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
