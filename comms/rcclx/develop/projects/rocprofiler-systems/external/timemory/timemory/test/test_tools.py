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

import json
import unittest
import timemory as tim


# --------------------------- Hatchet Tests set ---------------------------------------- #
# Hatchet tests class
class TimemoryToolsTests(unittest.TestCase):
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
        tim.settings.memory_units = "MB"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test handling wall-clock data from hatchet
    def test_mallocp(self):
        """mallocp"""

        try:
            import numpy as np
            import gc
        except ImportError:
            return

        if not tim.component.is_available("malloc_gotcha"):
            return

        from timemory.component import MallocGotcha
        from timemory.storage import MallocGotchaStorage

        _idx = tim.start_mallocp()
        _arr = np.ones([1000, 1000], dtype=np.float64)
        _sum = np.sum(_arr)
        del _arr
        gc.collect()
        _idx = tim.stop_mallocp(_idx)

        self.assertTrue(_sum == 1000 * 1000)
        self.assertTrue(_idx == 0)

        _json = tim.get(components=[MallocGotcha.id()])
        _data = MallocGotchaStorage.get()
        _alloc = 0.0
        _dealloc = 0.0

        print("{}\n".format(json.dumps(_json, indent=4)))

        for itr in _data:
            if "malloc" in itr.prefix() or "calloc" in itr.prefix():
                _alloc += itr.data().get_raw()
            if "free" in itr.prefix():
                _dealloc += itr.data().get_raw()

        _minsz = 1000 * 1000 * 8
        print(f"Minimum size     : {_minsz}")
        print(f"Allocated size   : {_alloc}")
        print(f"Deallocated size : {_alloc}")

        self.assertTrue(_alloc > _minsz)

        unit_v = _json["timemory"]["malloc_gotcha"]["unit_value"]
        unit_r = _json["timemory"]["malloc_gotcha"]["unit_repr"]
        real_v = MallocGotcha.unit()
        real_r = MallocGotcha.display_unit().upper()

        self.assertEqual(unit_v, real_v)
        self.assertEqual(unit_r, real_r)


# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    tim.initialize([__file__])
    run()
