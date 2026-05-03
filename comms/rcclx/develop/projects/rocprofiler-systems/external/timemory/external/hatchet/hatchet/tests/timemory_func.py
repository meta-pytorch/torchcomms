# Copyright 2020-2023 The Regents of the University of California, through
# Lawrence Berkeley National Laboratory, and other Hatchet Project Developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

timemory_avail = True
try:
    from timemory.profiler import Profiler
    from timemory.trace import Tracer
    from timemory.component import WallClock, CpuClock
except ImportError:
    timemory_avail = False

    class FakeComponent:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def id():
            pass

    class FakeProfiler:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            pass

    WallClock = FakeComponent
    CpuClock = FakeComponent
    Profiler = FakeProfiler
    Tracer = FakeProfiler


ncnt = 0
components = [WallClock.id(), CpuClock.id()]


def eval_func(arr, tol):
    """Dummy tolerance-checking function for profiling which introduces some arbitrary branching"""
    global ncnt
    max = np.max(arr)
    if ncnt % 3 != 0:
        avg = np.mean(arr)
    else:
        avg = max
    ncnt = ncnt + 1
    return True if avg < tol and max < tol else False


@Profiler(components)
def prof_func(arr, tol):
    """Dummy function for profiling"""
    while not eval_func(arr, tol):
        arr = arr - np.power(arr, 3)


@Tracer(components)
def trace_func(arr, tol):
    """Dummy function for tracing"""
    while not eval_func(arr, tol):
        arr = arr - np.power(arr, 3)
