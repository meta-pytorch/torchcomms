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
import mmap
import numpy as np
import unittest
import random
import tempfile
import timemory as tim
from timemory import component as comp

# --------------------------- test setup variables ----------------------------------- #
niter = 20
nelements = 0.95 * (mmap.PAGESIZE * 500)
memory_unit = (tim.units.kilobyte, "KiB")

peak = comp.PeakRss("peak_rss")
curr = comp.PageRss("curr_rss")
rb = comp.ReadBytes("read_bytes")
wb = comp.WrittenBytes("written_bytes")
current_peak = comp.CurrentPeakRss("curr_peak_rss")

sizeof_int64_t = np.ones(1, np.int64).itemsize
sizeof_double = np.ones(1, np.float64).itemsize

tot_size = nelements * sizeof_int64_t / memory_unit[0]
tot_rw = nelements * sizeof_double / memory_unit[0]

peak_tolerance = 5 * tim.units.megabyte
curr_tolerance = 5 * tim.units.megabyte
byte_tolerance = tot_rw


# --------------------------- helper functions ----------------------------------------- #
# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n - 1) + fibonacci(n - 2))


# get a random entry
def random_entry(vect):
    random.seed(int(1.0e9 * time.time()))
    num = int(random.uniform(0, len(vect) - 1))
    return vect[num]


# allocate memory
def allocate():
    peak.reset()
    curr.reset()
    current_peak.reset()

    curr.start()
    peak.start()
    current_peak.start()

    v = np.full(int(nelements), 15, dtype=np.int64)
    ret = fibonacci(0)
    nfib = random_entry(v)

    for i in range(niter):
        nfib = random_entry(v)
        ret += fibonacci(nfib)

    if ret < 0:
        print("fibonacci({}) * {} = {}\n".format(nfib, niter, ret))

    current_peak.stop()
    curr.stop()
    peak.stop()


# read and write to file
def read_write():
    rb.start()
    wb.start()

    ofs = tempfile.NamedTemporaryFile(delete=False)
    fname = ofs.name

    writeArray = np.array(range(int(nelements)))
    writeArray = writeArray.astype(np.float64)

    # make a new bytes array
    writeByteArray = bytearray(writeArray)

    # write the data to file
    ofs.write(writeByteArray)

    # close the file
    ofs.close()

    # read the data from file
    data = np.fromfile(fname, dtype=np.float64)

    rb.stop()
    wb.stop()

    return data


# print_info for peak, page rss
def print_info(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get())
    print("[" + name + "]>  expected :", expected, memory_unit[1])


# print_info for wb and rb
def print_info_wb_rb(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get()[0])
    print("[" + name + "]>  expected :", expected, memory_unit[1])


# print_info for current peak
def print_info_curr_peak(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get()[1] - obj.get()[0])
    print("[" + name + "]>  expected :", expected, memory_unit[1])


# --------------------------- RUsage Tests set ---------------------------------------- #
# RSS usage tests class
class TimemoryRUsageTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        tim.settings.verbose = 0
        tim.settings.debug = False
        tim.settings.json_output = True
        tim.settings.precision = 9
        tim.settings.memory_units = memory_unit[1]
        tim.settings.mpi_thread = False
        tim.settings.file_output = True

        # perform allocation once here
        allocate()

        # perform read write once here
        read_write()

        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False
        tim.settings.dart_type = "peak_rss"

    # tear down class: finalize
    @classmethod
    def tearDownClass(self):
        # timemory finalize
        # tim.finalize()
        # tim.dmp.finalize()
        pass

    # ---------------------------------------------------------------------------------- #
    # test peak rss
    def test_peak_rss(self):
        """peak rss"""
        if not comp.PeakRss.available:
            raise unittest.SkipTest("[{}] not available".format("peak_rss"))

        print_info(self.shortDescription(), peak, tot_size)
        self.assertAlmostEqual(tot_size, peak.get(), delta=peak_tolerance)

    # ---------------------------------------------------------------------------------- #
    # test page rss
    def test_page_rss(self):
        """page rss"""
        if not comp.PageRss.available:
            raise unittest.SkipTest("[{}] not available".format("page_rss"))

        print_info(self.shortDescription(), curr, tot_size)
        self.assertAlmostEqual(tot_size, curr.get(), delta=curr_tolerance)

    # ---------------------------------------------------------------------------------- #
    # test read_bytes
    def test_read_bytes(self):
        """read bytes"""
        if not comp.ReadBytes.available:
            raise unittest.SkipTest("[{}] not available".format("read_bytes"))

        print_info_wb_rb(self.shortDescription(), rb, tot_rw)
        self.assertAlmostEqual(tot_rw, rb.get()[0], delta=byte_tolerance)

    # ---------------------------------------------------------------------------------- #
    # test write_bytes
    def test_written_bytes(self):
        """written bytes"""
        if not comp.WrittenBytes.available:
            raise unittest.SkipTest(
                "[{}] not available".format("written_bytes")
            )

        print_info_wb_rb(self.shortDescription(), wb, tot_rw)
        self.assertAlmostEqual(tot_rw, wb.get()[0], delta=byte_tolerance)

    # ---------------------------------------------------------------------------------- #
    # test write_bytes
    def test_current_peak_rss(self):
        """current peak rss"""
        if not comp.CurrentPeakRss.available:
            raise unittest.SkipTest(
                "[{}] not available".format("current_peak_rss")
            )

        print_info_curr_peak(self.shortDescription(), current_peak, tot_size)
        self.assertAlmostEqual(
            tot_size,
            current_peak.get()[1] - current_peak.get()[0],
            delta=peak_tolerance,
        )


# ----------------------------- main test runner -------------------------------------- #
# test runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
