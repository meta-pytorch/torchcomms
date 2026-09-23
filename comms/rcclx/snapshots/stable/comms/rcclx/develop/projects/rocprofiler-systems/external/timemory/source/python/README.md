# Python Bindings

The timemory python interface is generated via the PyBind11 library. The combination of these two template-based
libraries provides a feature-rich interface which combines the flexibility of python with the performance of C++.

## Description

The python interface provides several pre-configured scoped components in bundles and profiler sub-packages which can be used as decorators or context-managers. The timemory settings be controlled either directly via the `settings` sub-package or by using environment variables. The `component` subpackage contains the individual components that can be used for custom instrumentation. Further, the `hardware_counters` sub-package provides API interface to accessing hardware counters (requires PAPI and/or CUDA support). The `mpi` subpackage provides bindings to timemory's MPI support. The generic data `plotting` (such as plotting instrumentation graphs) and `roofline` plotting are available in subsequent sub-packages.

## Contents

```bash
$ python -c "import timemory; help(timemory)"

Help on package timemory:

NAME
    timemory

PACKAGE CONTENTS
    api (package)
    bundle (package)
    common
    component (package)
    ert (package)
    hardware_counters (package)
    libs (package)
    mpi (package)
    mpi_support (package)
    notebook (package)
    options
    plotting (package)
    profiler (package)
    region (package)
    roofline (package)
    settings (package)
    signals
    test (package)
    trace (package)
    units
    util (package)

SUBMODULES
    scope

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        timemory.libs.auto_timer
        timemory.libs.component_bundle
        timemory.libs.manager
        timemory.libs.rss_usage
        timemory.libs.settings
        timemory.libs.timer

    class auto_timer(...)
    class component_bundle(...)
    class manager(...)
    class rss_usage(...)
    class settings(...)
    class timer(...)

FUNCTIONS
    FILE = file(back=2, only_basename=True, use_dirname=False, noquotes=True)
        Returns the file name

    FUNC = func(back=2)
        Returns the function name

    LINE = line(back=1)
        Returns the line number

    disable(...)
        disable() -> None

        Disable timemory

    disable_signal_detection(...)
        disable_signal_detection() -> None

        Enable signal detection

    enable(...)
        enable() -> None

        Enable timemory

    enable_signal_detection(...)
        enable_signal_detection(signal_list: list = []) -> None

        Enable signal detection

    enabled(...)
        enabled() -> bool

        Return if timemory is enabled or disabled

    finalize(...)
        finalize() -> None

        Finalize timemory (generate output) -- important to call if using MPI

    has_mpi_support(...)
        has_mpi_support() -> bool

        Return if the timemory library has MPI support

    init = initialize(...)
        initialize(argv: list = [], prefix: str = 'timemory-', suffix: str = '-output') -> None

        Initialize timemory

    initialize(...)
        initialize(argv: list = [], prefix: str = 'timemory-', suffix: str = '-output') -> None

        Initialize timemory

    is_enabled(...)
        is_enabled() -> bool

        Return if timemory is enabled or disabled

    report(...)
        report(filename: str = '') -> None

        Print the data

    set_rusage_children(...)
        set_rusage_children() -> None

        Set the rusage to record child processes

    set_rusage_self(...)
        set_rusage_self() -> None

        Set the rusage to record child processes

    timemory_finalize(...)
        timemory_finalize() -> None

        Finalize timemory (generate output) -- important to call if using MPI

    timemory_init(...)
        timemory_init(argv: list = [], prefix: str = 'timemory-', suffix: str = '-output') -> None

        Initialize timemory

    toggle(...)
        toggle(on: bool = True) -> None

        Enable/disable timemory

DATA
    __all__ = ['version_info', 'build_info', 'version', 'libs', '...
    __copyright__ = 'Copyright 2020, The Regents of the University of Cali...
    __email__ = 'jrmadsen@lbl.gov'
    __license__ = 'MIT'
    __maintainer__ = 'Jonathan Madsen'
    __status__ = 'Development'
    __warningregistry__ = {'version': 0, ("the imp module is deprecated in...
    build_info = {'build_type': 'RelWithDebInfo', 'compiler': '/opt/local/...
    version = '3.2.0'
    version_info = (3, 2, 0)

VERSION
    3.2.0

AUTHOR
    Jonathan Madsen

CREDITS
    ['Jonathan Madsen']

FILE
    /.../timemory/__init__.py
```

## Initialization and Finalization

- `timemory.init(...)` is called when the package is imported
  - Although it is maybe potentially usually useful to initialize with the filename
- It is highly recommended to call `timemory.finalize()` explicitly before the application terminates

```python
import timemory

# ... use timemory components, decorators, bundles, etc. ...

if __name__ == "__main__":
    # optional
    timemory.init([__file__])

    # ... etc. ...

    timemory.finalize()
```

## Settings

Timemory settings can be directly modified in Python or may be configured via enviroment variables.

### Direct Modification

```python
import timemory

# set verbose output to 1
timemory.settings.verbose = 1
# disable timemory debug prints
timemory.settings.debug = False
# set output data format output to json
timemory.settings.json_output = True
# disable mpi_thread mode
timemory.settings.mpi_thread  = False
# enable timemory dart output
timemory.settings.dart_output = True
timemory.settings.dart_count = 1
# disable timemory banner
timemory.settings.banner = False
```

### Environment Variables

```python
import os

# enable timemory flat profile mode
os.environ["TIMEMORY_FLAT_PROFILE"] = "ON"
# enable timemory timeline profile mode
os.environ["TIMEMORY_TIMELINE_PROFILE"] = "ON"
# parse the environment variable updates within timemory
timemory.settings.parse()
```

## Decorators

```python
import timemory
import timemory.util.marker as marker, auto_timer

@marker(["trip_count", "peak_rss"])
def fibonacci_instrumented(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)

@auto_timer()
def fibonacci_timed(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)
```

## Context Managers

```python
import timemory
from timemory.util import marker

def main():
    with marker(["wall_clock", "peak_rss"], flat=False, timeline=False):
        ans = fibonacci(n=3)
```

## Function Profiler

### Profiler Example

```python
#!/usr/bin/env python

import numpy as np
import timemory

from timemory.profiler import Profiler
from timemory.profiler import Config as ProfilerConfig

def eval_func(arr, tol):
    """Dummy tolerance-checking function"""
    max = np.max(arr)
    avg = np.mean(arr)
    return True if avg < tol and max < tol else False

@Profiler(["wall_clock", "cpu_clock"])
def profile_func(arr, tol):
    """Dummy function for profiling"""
    while not eval_func(arr, tol):
        arr = arr - np.power(arr, 3)

if __name__ == "__main__":

    ProfilerConfig.only_filenames = [__file__, "_methods.py"]
    profile_func(np.random.rand(100, 100), 1.0e-2)

    timemory.finalize()
```

### Profiler Output

```console
$ python ./doc-profiler.py

[cpu]|0> Outputting 'timemory-doc-profiler-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-doc-profiler-output/cpu.tree.json'...
[cpu]|0> Outputting 'timemory-doc-profiler-output/cpu.json'...
[cpu]|0> Outputting 'timemory-doc-profiler-output/cpu.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------|
|                                            TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                            |
|------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> profile_func/doc-profiler.py:15            |      1 |      0 | cpu    | sec    |  1.610 |  1.610 |  1.610 |  1.610 |  0.000 |   57.1 |
| >>> |_eval_func/doc-profiler.py:9              |   4994 |      1 | cpu    | sec    |  0.690 |  0.000 |  0.000 |  0.000 |  0.001 |   50.7 |
| >>>   |__mean/_methods.py:134                  |   4994 |      2 | cpu    | sec    |  0.340 |  0.000 |  0.000 |  0.000 |  0.001 |   55.9 |
| >>>     |__count_reduce_items/_methods.py:50   |   4994 |      3 | cpu    | sec    |  0.040 |  0.000 |  0.000 |  0.000 |  0.000 |   75.0 |
| >>>       |__count_reduce_items/_methods.py:50 |   4994 |      4 | cpu    | sec    |  0.010 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |__mean/_methods.py:134                |  24970 |      3 | cpu    | sec    |  0.110 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-doc-profiler-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-doc-profiler-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-doc-profiler-output/wall.json'...
[wall]|0> Outputting 'timemory-doc-profiler-output/wall.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------|
|                                                 REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> profile_func/doc-profiler.py:15            |      1 |      0 | wall   | sec    |  1.614 |  1.614 |  1.614 |  1.614 |  0.000 |   56.9 |
| >>> |_eval_func/doc-profiler.py:9              |   4994 |      1 | wall   | sec    |  0.697 |  0.000 |  0.000 |  0.000 |  0.000 |   39.0 |
| >>>   |__mean/_methods.py:134                  |   4994 |      2 | wall   | sec    |  0.425 |  0.000 |  0.000 |  0.000 |  0.000 |   51.5 |
| >>>     |__count_reduce_items/_methods.py:50   |   4994 |      3 | wall   | sec    |  0.084 |  0.000 |  0.000 |  0.000 |  0.000 |   76.2 |
| >>>       |__count_reduce_items/_methods.py:50 |   4994 |      4 | wall   | sec    |  0.020 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |__mean/_methods.py:134                |  24970 |      3 | wall   | sec    |  0.122 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------|
```

## Tracing Profiler

### Tracer Example

```python
#!/usr/bin/env python

import numpy as np
import timemory

from timemory.trace import Tracer
from timemory.trace import Config as TracerConfig

def eval_func(arr, tol):
    """Dummy tolerance-checking function"""
    max = np.max(arr)
    avg = np.mean(arr)
    return True if avg < tol and max < tol else False

@Tracer(["wall_clock", "cpu_clock"])
def trace_func(arr, tol):
    """Dummy function for tracing"""
    while not eval_func(arr, tol):
        arr = arr - np.power(arr, 3)

if __name__ == "__main__":

    TracerConfig.only_filenames = [__file__, "_methods.py"]
    trace_func(np.random.rand(100, 100), 1.0e-2)

    timemory.finalize()
```

### Tracer Output

```console
$ python ./doc-tracer.py

[cpu]|0> Outputting 'timemory-doc-tracer-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-doc-tracer-output/cpu.tree.json'...
[cpu]|0> Outputting 'timemory-doc-tracer-output/cpu.json'...
[cpu]|0> Outputting 'timemory-doc-tracer-output/cpu.txt'...

# ... report for cpu-clock ...

[wall]|0> Outputting 'timemory-doc-tracer-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-doc-tracer-output/wall.json'...
[wall]|0> Outputting 'timemory-doc-tracer-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-doc-tracer-output/wall.txt'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                    REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                        LABEL                                                         | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> @Tracer(["wall_clock", "cpu_clock"])                                               [trace_func/doc-tracer.py:15] |        |        |        |        |        |        |        |        |        |        |
| >>> def trace_func(arr, tol):                                                          [trace_func/doc-tracer.py:16] |      1 |      0 | wall   | sec    |  2.044 |  2.044 |  2.044 |  2.044 |  0.000 |  100.0 |
| >>>     while not eval_func(arr, tol):                                                 [trace_func/doc-tracer.py:18] |   4994 |      0 | wall   | sec    |  0.027 |  0.000 |  0.027 |  0.027 |  0.000 |  100.0 |
| >>>         arr = arr - np.power(arr, 3)                                               [trace_func/doc-tracer.py:19] |   4993 |      0 | wall   | sec    |  0.894 |  0.000 |  0.894 |  0.894 |  0.000 |  100.0 |
| >>> def eval_func(arr, tol):                                                           [trace_func/doc-tracer.py:09] |   4994 |      0 | wall   | sec    |  1.080 |  0.000 |  1.080 |  1.080 |  0.000 |  100.0 |
| >>>     max = np.max(arr)                                                              [trace_func/doc-tracer.py:11] |   4994 |      0 | wall   | sec    |  0.023 |  0.000 |  0.023 |  0.023 |  0.000 |  100.0 |
| >>>     avg = np.mean(arr)                                                             [trace_func/doc-tracer.py:12] |   4994 |      0 | wall   | sec    |  0.027 |  0.000 |  0.027 |  0.027 |  0.000 |  100.0 |
| >>>     return True if avg < tol and max < tol else False                              [trace_func/doc-tracer.py:13] |   4994 |      0 | wall   | sec    |  0.022 |  0.000 |  0.022 |  0.022 |  0.000 |  100.0 |
| >>> def _mean(a, axis=None, dtype=None, out=None, keepdims=False):                           [_mean/_methods.py:134] |   4994 |      0 | wall   | sec    |  0.799 |  0.000 |  0.799 |  0.799 |  0.000 |  100.0 |
| >>>     arr = asanyarray(a)                                                                  [_mean/_methods.py:135] |   4994 |      0 | wall   | sec    |  0.023 |  0.000 |  0.023 |  0.023 |  0.000 |  100.0 |
| >>>     is_float16_result = False                                                            [_mean/_methods.py:137] |   4994 |      0 | wall   | sec    |  0.022 |  0.000 |  0.022 |  0.022 |  0.000 |  100.0 |
| >>>     rcount = _count_reduce_items(arr, axis)                                              [_mean/_methods.py:138] |   4994 |      0 | wall   | sec    |  0.022 |  0.000 |  0.022 |  0.022 |  0.000 |  100.0 |
| >>>     if rcount == 0:                                                                      [_mean/_methods.py:140] |   4994 |      0 | wall   | sec    |  0.021 |  0.000 |  0.021 |  0.021 |  0.000 |  100.0 |
| >>>         warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)              [_mean/_methods.py:141] |        |        |        |        |        |        |        |        |        |        |
| >>>     if dtype is None:                                                                    [_mean/_methods.py:144] |   4994 |      0 | wall   | sec    |  0.021 |  0.000 |  0.021 |  0.021 |  0.000 |  100.0 |
| >>>         if issubclass(arr.dtype.type, (nt.integer, nt.bool_)):                           [_mean/_methods.py:145] |   4994 |      0 | wall   | sec    |  0.025 |  0.000 |  0.025 |  0.025 |  0.000 |  100.0 |
| >>>             dtype = mu.dtype('f8')                                                       [_mean/_methods.py:146] |        |        |        |        |        |        |        |        |        |        |
| >>>         elif issubclass(arr.dtype.type, nt.float16):                                     [_mean/_methods.py:147] |   4994 |      0 | wall   | sec    |  0.023 |  0.000 |  0.023 |  0.023 |  0.000 |  100.0 |
| >>>             dtype = mu.dtype('f4')                                                       [_mean/_methods.py:148] |        |        |        |        |        |        |        |        |        |        |
| >>>             is_float16_result = True                                                     [_mean/_methods.py:149] |        |        |        |        |        |        |        |        |        |        |
| >>>     ret = umr_sum(arr, axis, dtype, out, keepdims)                                       [_mean/_methods.py:151] |   4994 |      0 | wall   | sec    |  0.056 |  0.000 |  0.056 |  0.056 |  0.000 |  100.0 |
| >>>     if isinstance(ret, mu.ndarray):                                                      [_mean/_methods.py:152] |   4994 |      0 | wall   | sec    |  0.026 |  0.000 |  0.026 |  0.026 |  0.000 |  100.0 |
| >>>         ret = um.true_divide(                                                            [_mean/_methods.py:153] |        |        |        |        |        |        |        |        |        |        |
| >>>                 ret, rcount, out=ret, casting='unsafe', subok=False)                     [_mean/_methods.py:154] |        |        |        |        |        |        |        |        |        |        |
| >>>         if is_float16_result and out is None:                                            [_mean/_methods.py:155] |        |        |        |        |        |        |        |        |        |        |
| >>>             ret = arr.dtype.type(ret)                                                    [_mean/_methods.py:156] |        |        |        |        |        |        |        |        |        |        |
| >>>     elif hasattr(ret, 'dtype'):                                                          [_mean/_methods.py:157] |   4994 |      0 | wall   | sec    |  0.023 |  0.000 |  0.023 |  0.023 |  0.000 |  100.0 |
| >>>         if is_float16_result:                                                            [_mean/_methods.py:158] |   4994 |      0 | wall   | sec    |  0.021 |  0.000 |  0.021 |  0.021 |  0.000 |  100.0 |
| >>>             ret = arr.dtype.type(ret / rcount)                                           [_mean/_methods.py:159] |        |        |        |        |        |        |        |        |        |        |
| >>>         else:                                                                            [_mean/_methods.py:160] |        |        |        |        |        |        |        |        |        |        |
| >>>             ret = ret.dtype.type(ret / rcount)                                           [_mean/_methods.py:161] |   4994 |      0 | wall   | sec    |  0.028 |  0.000 |  0.028 |  0.028 |  0.000 |  100.0 |
| >>>     else:                                                                                [_mean/_methods.py:162] |        |        |        |        |        |        |        |        |        |        |
| >>>         ret = ret / rcount                                                               [_mean/_methods.py:163] |        |        |        |        |        |        |        |        |        |        |
| >>>     return ret                                                                           [_mean/_methods.py:165] |   4994 |      0 | wall   | sec    |  0.021 |  0.000 |  0.021 |  0.021 |  0.000 |  100.0 |
| >>> def _count_reduce_items(arr, axis):                                                      [_mean/_methods.py:050] |   4994 |      0 | wall   | sec    |  0.304 |  0.000 |  0.304 |  0.304 |  0.000 |  100.0 |
| >>>     if axis is None:                                                                     [_mean/_methods.py:051] |   4994 |      0 | wall   | sec    |  0.021 |  0.000 |  0.021 |  0.021 |  0.000 |  100.0 |
| >>>         axis = tuple(range(arr.ndim))                                                    [_mean/_methods.py:052] |   4994 |      0 | wall   | sec    |  0.027 |  0.000 |  0.027 |  0.027 |  0.000 |  100.0 |
| >>>     if not isinstance(axis, tuple):                                                      [_mean/_methods.py:053] |   4994 |      0 | wall   | sec    |  0.022 |  0.000 |  0.022 |  0.022 |  0.000 |  100.0 |
| >>>         axis = (axis,)                                                                   [_mean/_methods.py:054] |        |        |        |        |        |        |        |        |        |        |
| >>>     items = 1                                                                            [_mean/_methods.py:055] |   4994 |      0 | wall   | sec    |  0.020 |  0.000 |  0.020 |  0.020 |  0.000 |  100.0 |
| >>>     for ax in axis:                                                                      [_mean/_methods.py:056] |  14982 |      0 | wall   | sec    |  0.060 |  0.000 |  0.060 |  0.060 |  0.000 |  100.0 |
| >>>         items *= arr.shape[ax]                                                           [_mean/_methods.py:057] |   9988 |      0 | wall   | sec    |  0.042 |  0.000 |  0.042 |  0.042 |  0.000 |  100.0 |
| >>>     return items                                                                         [_mean/_methods.py:058] |   4994 |      0 | wall   | sec    |  0.020 |  0.000 |  0.020 |  0.020 |  0.000 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```

## Library Bindings

This is a sample of the python bindings to the timemory library interface available to C, C++, and Fortran.
Using these libraries calls in python code which calls C/C++/Fortran code also instrumented with
timemory will enable the same set of performance analysis components to span all the languages.

```python
import timemory

# initialize
timemory.trace.init("wall_clock", False, "tracing")

# insert a trace
timemory.trace.push("consume_time")
ans = work_milliseconds(1000)
timemory.trace.pop("consume_time")

# insert another trace point
timemory.trace.push("sleeping")
ans = sleep_milliseconds(1000)
timemory.trace.pop("sleeping")

# insert a region
timemory.region.push("work_region")
for i in range(10):
    ans = work_milliseconds(1000)
timemory.region.pop("work_region")

# finalize
timemory.trace.finalize()
```

## Individual Components

Each individual component is available as a stand-alone Python class.
The intention of making these individual components available is such that
tools can use these types to create custom tools. Thus, unless `push()` and `pop()`
are called on these objects, instances of these classes
will not store any data in the timemory call-graph (when applicable).

```python
import time
import timemory
from  timemory.component import WallClock

# instantiate wall clock component
wc = WallClock("wall-time")
# start the clock
wc.start()
# sleep
time.sleep(2)
# stop the clock
wc.stop()
# get data
result = wc.get()
#finalize timemory
timemory.finalize()
```

## Storage

Timemory allows direct access to the aforementioned call-graph storage.
The call-graph storage is the accumulated data for components which occur through
push and pop operations. These push and pop operations are implicitly
performed via decorators, context-managers, the function profiler, and the
trace profiler and may be explicitly performed when using individual components.
Internally, timemory stores call-graph entries in a minimal representation
and then packs the results together with more information when the data is requested.
Thus, one should not expect manipulation of the data provided by these routines to
be propagated to timemory internally.

Call-graph storage is available in two different layouts. The first represents
the call-graph results for each process as a one-dimensional array where the hierarchy
is represented through indentation of the string identifiers, a depth field, and
an array of hash values. The second represents the call-graph entries as a nested
data structure where each entry has a value and list of children.

### Storage Example

```python
#!/usr/bin/env python

import time
import timemory
from timemory.util import marker
from timemory.component import WallClock
from timemory.storage import WallClockStorage

@marker(["wall_clock"])
def foo():
    """Generate some data for timemory"""
    time.sleep(1)

    wc = WallClock("bar")
    for i in range(0, 10):
        # push every even iteration
        if i % 2 == 0:
            wc.push()
        wc.start()
        time.sleep(0.1 * (i + 1))
        wc.stop()
        # pop every odd iteration
        if i % 2 == 1:
            wc.pop()


def print_result():
    """
    Print the call-graph storage via the flat layout
    """

    print("\n#{}#\n# Storage Result".format("-" * 40))

    indent = "  "
    for itr in WallClockStorage.get():
        print("#{}#".format("-" * 40))
        print("{}{:20} : {}".format(indent, "Thread id", itr.tid()))
        print("{}{:20} : {}".format(indent, "Process id", itr.pid()))
        print("{}{:20} : {}".format(indent, "Depth", itr.depth()))
        print("{}{:20} : {}".format(indent, "Hash", itr.hash()))
        print("{}{:20} : {}".format(indent, "Rolling hash", itr.rolling_hash()))
        print("{}{:20} : {}".format(indent, "Prefix", itr.prefix()))
        print("{}{:20} : {}".format(indent, "Hierarchy", itr.hierarchy()))
        print("{}{:20} : {}".format(indent, "Data object", itr.data()))
        print("{}{:20} : {}".format(indent, "Statistics", itr.stats()))


def print_tree(data=None, depth=0):
    """
    Print the call-graph storage via the nested layout
    """

    if data is None:
        print("\n#{}#\n# Storage Tree".format("-" * 40))
        data = WallClockStorage.get_tree()

    def print_value(itr, indent):
        print("{}{:20} : {}".format(indent, "Thread id", itr.tid()))
        print("{}{:20} : {}".format(indent, "Process id", itr.pid()))
        print("{}{:20} : {}".format(indent, "Depth", itr.depth()))
        print("{}{:20} : {}".format(indent, "Hash", itr.hash()))
        print("{}{:20} : {}".format(indent, "Inclusive data", itr.inclusive().data()))
        print("{}{:20} : {}".format(indent, "Inclusive stat", itr.inclusive().stats()))
        print("{}{:20} : {}".format(indent, "Exclusive data", itr.exclusive().data()))
        print("{}{:20} : {}".format(indent, "Exclusive stat", itr.exclusive().stats()))

    indent = "  " * depth
    for itr in data:
        print("{}#{}#".format(indent, "-" * 40))
        print_value(itr.value(), indent)
        print_tree(itr.children(), depth + 1)


if __name__ == "__main__":
    # disable automatic output
    timemory.settings.auto_output = False

    foo()
    print_result()
    print_tree()
```

### Storage Output

```console

#----------------------------------------#
# Storage Result
#----------------------------------------#
  Thread id            : 0
  Process id           : 4385
  Depth                : 0
  Hash                 : 9631199822919835227
  Rolling hash         : 9631199822919835227
  Prefix               : >>> foo
  Hierarchy            : [9631199822919835227]
  Data object          :    6.534 sec wall
  Statistics           : [sum: 6.53361] [min: 6.53361] [max: 6.53361] [sqr: 42.6881] [count: 1]
#----------------------------------------#
  Thread id            : 0
  Process id           : 4385
  Depth                : 1
  Hash                 : 11474628671133349553
  Rolling hash         : 2659084420343633164
  Prefix               : >>> |_bar
  Hierarchy            : [9631199822919835227, 11474628671133349553]
  Data object          :    5.531 sec wall
  Statistics           : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]

#----------------------------------------#
# Storage Tree
#----------------------------------------#
Thread id            : {0}
Process id           : {4385}
Depth                : -1
Hash                 : 0
Prefix               : unknown-hash=0
Inclusive data       :    0.000 sec wall
Inclusive stat       : [sum: 0] [min: 0] [max: 0] [sqr: 0] [count: 0]
Exclusive data       :   -6.534 sec wall
Exclusive stat       : [sum: 0] [min: 0] [max: 0] [sqr: 0] [count: 0]
  #----------------------------------------#
  Thread id            : {0}
  Process id           : {4385}
  Depth                : 0
  Hash                 : 9631199822919835227
  Prefix               : foo
  Inclusive data       :    6.534 sec wall
  Inclusive stat       : [sum: 6.53361] [min: 6.53361] [max: 6.53361] [sqr: 42.6881] [count: 1]
  Exclusive data       :    1.002 sec wall
  Exclusive stat       : [sum: 1.00246] [min: 6.53361] [max: 6.53361] [sqr: 34.9765] [count: 1]
    #----------------------------------------#
    Thread id            : {0}
    Process id           : {4385}
    Depth                : 1
    Hash                 : 11474628671133349553
    Prefix               : bar
    Inclusive data       :    5.531 sec wall
    Inclusive stat       : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]
    Exclusive data       :    5.531 sec wall
    Exclusive stat       : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]
```

Note the first entry of storage tree has a negative depth and hash of zero. Nodes such of these
are "dummy" nodes which timemory keeps internally as bookmarks for root nodes and thread-forks
(parent call-graph location when a child thread was initialized or returned to "sea-level").
These entries can be discarded and a member function `is_dummy()` exists to help identify these nodes, e.g.:

```python
    for itr in data:
        if itr.value().is_dummy():
            print_tree(itr.children(), depth)
        else:
            print("{}#{}#".format(indent, "-" * 40))
            print_value(itr.value(), indent)
            print_tree(itr.children(), depth + 1)
```

Future versions of timemory may eliminate these reporting these nodes.
