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

""" @file test/__main__.py
Run all timemory unittests
"""

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
import sys
import unittest
import timemory as tim


# discover and run all timemory unittests in the current directory
def run_all_tests():
    # auto discover unittests from test_*.py files into the timemory test suite
    timTestSuite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern="test*.py",
    )

    # print the loaded tests
    print(
        "============= Loaded Tests =============\n\n {}\n".format(timTestSuite)
    )

    # create a results object to store test results
    result = unittest.TestResult()

    # enable stdout buffer
    result.buffer = True

    # run all tests in timTestSuite, use result object to store results
    print("\n============= Tests Stdout =============\n")
    # run the tests
    timTestSuite.run(result)

    # finalize tracing
    tim.trace.finalize()

    # finalize timemory
    tim.finalize()

    # print the results
    print("\n============= Results =============\n")
    print("{}\n".format(result))
    return result


# run all tests
if __name__ == "__main__":
    tim.initialize([__file__])
    man = tim.manager()
    result = run_all_tests()
    man.write_ctest_notes("./python-testing")
    tim.finalize()
    if result.errors is not None:
        nerr = len(result.errors)
        sys.exit(nerr)
    else:
        sys.exit(0)
