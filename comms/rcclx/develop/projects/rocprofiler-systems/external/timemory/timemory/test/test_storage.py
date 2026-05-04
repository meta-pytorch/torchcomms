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

import unittest
import timemory as tim
from timemory.bundle import marker


# --------------------------- helper functions ----------------------------------------- #
# compute fibonacci
def fibonacci(n, with_arg=True):
    with marker(
        ["wall_clock", "peak_rss", "current_peak_rss"],
        "fib({})".format(n) if with_arg else "",
        mode="full" if not with_arg else "defer",
    ):
        return (
            n
            if n < 2
            else (fibonacci(n - 1, with_arg) + fibonacci(n - 2, with_arg))
        )


# --------------------------- Storage Tests set ---------------------------------------- #
# Storage tests class
class TimemoryStorageTests(unittest.TestCase):
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
        fibonacci(5, False)
        # fibonacci(8)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test handling wall-clock data from storage
    def test_storage_tree_data(self):
        """storage_tree_output"""

        data = []
        data.append(
            "{}".format(tim.get(hierarchy=True, components=["wall_clock"]))
        )
        data.append(
            "{}".format(
                tim.storage.WallClockStorage.dumps(
                    tim.storage.WallClockStorage.dmp_get_tree()
                )
            )
        )

        self.assertEqual(data[0], data[1])

    # ---------------------------------------------------------------------------------- #
    # test handling wall-clock data from storage
    def test_storage_flat_data(self):
        """storage_flat_output"""

        data = []
        data.append(
            "{}".format(tim.get(hierarchy=False, components=["wall_clock"]))
        )
        data.append(
            "{}".format(
                tim.storage.WallClockStorage.dumps(
                    tim.storage.WallClockStorage.dmp_get()
                )
            )
        )

        self.assertEqual(data[0], data[1])

    # ---------------------------------------------------------------------------------- #
    # test metadata storage
    def test_metadata_storage(self):
        """metadata_storage"""

        existing = tim.manager.get_metadata()
        existing["info"]["FOO"] = "bar"
        existing["info"]["BAR"] = [1.0, 2.0, 3.0]
        tim.manager.add_metadata("FOO", "bar")
        tim.manager.add_metadata("BAR", [1.0, 2.0, 3.0])

        self.assertDictEqual(existing, tim.manager.get_metadata())
        print("timemory metadata info:")
        for key, item in tim.manager.get_metadata()["info"].items():
            print(f"    {key} : {item}")


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
