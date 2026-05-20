#!@PYTHON_EXECUTABLE@
#
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
import os
import sys

from timemory.analyze import embedded_analyze


def try_analyze():
    debug = os.environ.get("TIMEMORY_DEBUG", False)
    verbose = int(os.environ.get("TIMEMORY_VERBOSE", 0))
    verbose += 4 if debug else 0
    if verbose > 3:
        print("Plotting with args: {}".format(sys.argv))
    embedded_analyze(
        sys.argv[1:],
        data=[],
        call_exit=True,
        verbose=os.environ.get("TIMEMORY_VERBOSE", verbose),
        ranks=[],
    )


if __name__ == "__main__":
    try_analyze()
