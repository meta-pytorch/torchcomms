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

from __future__ import absolute_import

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

from functools import wraps

try:
    from ..libs.libpytimemory import (
        start_function_wrappers as _start_function_wrappers,
    )
    from ..libs.libpytimemory import (
        stop_function_wrappers as _stop_function_wrappers,
    )

    available = True
except ImportError as e:
    import os

    _debug = os.environ.get("TIMEMORY_DEBUG", "").lower()
    if _debug in ["y", "yes", "1", "on", "true", "t"]:
        import sys

        sys.stderr.write(f"{e}\n")

    def _start_function_wrappers(*_args, **_kwargs):
        return None

    def _stop_function_wrappers(*_args, **_kwargs):
        return None

    available = False

__all__ = ["function_wrappers", "available"]


class function_wrappers(object):
    """A decorator or context-manager for dynamic function wrappers (Linux-only).
    These dynamic function wrappers either re-write the global offset table so that
    timemory components can extract their arguments and return values (e.g. wrap around
    malloc) or they enable the built-in callback API provided by the library itself
    (e.g. OMPT for OpenMP).

    Valid inputs are currently any of the strings following the tool name:
    - timemory-mpip: mpi, mpip
    - timemory-ompt: ompt, openmp
    - timemory-ncclp: nccl, ncclp
    - timemory-mallocp: malloc, mallocp, memory

    Example:

    .. highlight:: python
    .. code-block:: python

        @function_wrappers("mpi", "nccl", ompt=False)
        def foo():
            pass

        def bar():
            with function_wrappers(mpi=True, memory=True):
                pass
    """

    def __init__(self, *_args, **_kwargs):
        def _start_functor():
            return _start_function_wrappers(*_args, **_kwargs)

        def _stop_functor(_idx):
            _ret = _stop_function_wrappers(_idx)
            return None if sum(_ret) == 0 else _ret

        self._start = _start_functor
        self._stop = _stop_functor
        self._idx = None

    def start(self):
        self._idx = self._start()

    def stop(self):
        self._idx = self._stop(self._idx)

    def is_running(self):
        return self._idx is not None

    def __call__(self, func):
        """
        Decorator
        """

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            self._idx = self._start()
            _ret = func(*args, **kwargs)
            self._idx = self._stop(self._idx)
            return _ret

        return function_wrapper

    def __enter__(self, *args, **kwargs):
        """
        Context manager
        """
        self._idx = self._start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._idx = self._stop(self._idx)
        if (
            exc_type is not None
            and exc_value is not None
            and exc_traceback is not None
        ):
            import traceback

            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5
            )
