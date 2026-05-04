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

import os
import unittest
import numpy as np
import timemory as tim
from timemory.bundle import marker

rank = 0
try:
    import mpi4py  # noqa: F401
    from mpi4py import MPI  # noqa: F401

    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    pass

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Muhammad Haseeb"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


# --------------------------- helper functions ----------------------------------------- #
# compute fibonacci
def fibonacci(n, with_arg=True):
    mode = "full" if not with_arg else "defer"
    with marker(
        ["wall_clock", "peak_rss", "current_peak_rss"],
        "fib({})".format(n) if with_arg else "",
        mode=mode,
    ):
        arr = np.ones([100, 100], dtype=float)  # noqa: F841
        return (
            n
            if n < 2
            else (fibonacci(n - 1, with_arg) + fibonacci(n - 2, with_arg))
        )


# --------------------------- Hatchet Tests set ---------------------------------------- #
# Hatchet tests class
class TimemoryHatchetTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # set up environment variables
        tim.settings.verbose = 1
        tim.settings.debug = False
        tim.settings.tree_output = True
        tim.settings.text_output = True
        tim.settings.cout_output = False
        tim.settings.json_output = True
        tim.settings.flamegraph_output = False
        tim.settings.mpi_thread = False
        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False
        tim.settings.parse()

        # generate data
        fibonacci(15, False)
        fibonacci(8)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test handling wall-clock data from hatchet
    def test_hatchet_1D_data(self):
        """hatchet_1D_data"""

        try:
            import hatchet as ht
        except ImportError:
            return

        data = tim.get(hierarchy=True, components=["wall_clock"])
        gf = ht.GraphFrame.from_timemory(data)

        if rank == 0:
            print(gf.dataframe)
            print(gf.tree("sum.wall_clock"))
            print(gf.tree("sum.wall_clock.inc"))

    # ---------------------------------------------------------------------------------- #
    # test handling multi-dimensional data from hatchet
    def test_hatchet_2D_data(self):
        """hatchet_2d_data"""

        try:
            import hatchet as ht
        except ImportError:
            return

        from timemory.component import CurrentPeakRss

        data = tim.get(hierarchy=True, components=[CurrentPeakRss.id()])
        gf = ht.GraphFrame.from_timemory(data)

        if rank == 0:
            print(gf.dataframe)
            print(gf.tree("sum.start-peak-rss"))
            print(gf.tree("sum.stop-peak-rss.inc"))

        self.assertTrue(gf.tree("sum.start-peak-rss.inc") is not None)
        self.assertTrue(gf.tree("sum.stop-peak-rss") is not None)

    def test_hatchet_analyze(self):
        """test_hatchet_analyze"""

        try:
            import hatchet as ht  # noqa: F401
        except ImportError:
            return

        from timemory.analyze import embedded_analyze
        from timemory import settings
        import timemory

        outpath = os.path.join(
            settings.output_path,
            "analysis",
        )

        args = ["--expression", "x > 0"] + (
            f"-f dot flamegraph tree -o {outpath} --per-thread --per-rank "
            + "--select peak_rss --search fibonacci -e"
        ).split()

        if rank == 0:
            print(f"arguments: {args}")

        embedded_analyze(
            args,
            data=[timemory.get(hierarchy=True)],
        )


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
