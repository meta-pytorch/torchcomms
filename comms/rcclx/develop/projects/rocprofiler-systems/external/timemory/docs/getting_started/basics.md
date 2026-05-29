# Getting Started

## Dynamic Instrumentation

Dynamic instrumentation (instrumentation without any modifications to the source code) is only
available on Linux and requires the Dyninst package. It is recommended to install Dyninst
via Spack. For further information, please see documentation on
[timemory-run](../tools/timemory-run/README.md).

## Initialization and Finalization

Initialization is optional. The main benefits of initialization are, by default,
timemory will encode the name of the executable into the output folder, e.g.
`./foo ...` + initialization will generate output in `timemory-foo-output/`.
Finalization is somewhat optional but highly recommended because when data is
deleted at the termination of an application can be non-deterministic which
can lead to segmentation faults. When MPI or UPC++ is being used, finalization
should occur before these libraries are finalized and initialization can happen
before or after but initialization afterwards is recommended. Multiple calls
to initialization are permitted but multiple calls to finalization is not.

## Basic API

### C++ Toolkit Initialization and Finalization

```cpp
#include <timemory/timemory.hpp>

int main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    // ...
    tim::timemory_finalize();
    return 0;
}
```

### C / C++ Library Initialization and Finalization

```cpp
#include <timemory/library.h>

int main(int argc, char** argv)
{
    timemory_init_library(argc, argv);
    // ...
    timemory_finalize_library();
}
```

### Python Initialization and Finalization

```python
import timemory

if __name__ == "__main__":
    timemory.init([__file__])
    # ...
    timemory.finalize()
```

### C / C++ Library Interface

```cpp
#include <timemory/library.h>

void foo();
void bar();

int main(int argc, char** argv)
{
    // initialization
    timemory_init_library(argc, argv);

    // set the default set of components to collect
    timemory_set_default("wall_clock, cpu_clock");

    // start recording the current components
    timemory_push_region("main");

    // add peak resident-set size to current components
    timemory_add_components("peak_rss");

    foo();
    bar();

    // remove peak resident-set size to current components
    timemory_remove_components("peak_rss");

    // set cpu-utilization as the only set of components
    timemory_push_components("cpu_util");

    foo();

    // restore the previous configuration of components
    timemory_pop_components();

    // pause collection
    timemory_pause();

    bar();

    // resume collection
    timemory_resume();

    // end the main region
    timemory_pop_region("main");

    // end timemory collection, generate output
    timemory_finalize();

    return 0;
}

void foo()
{
    uint64_t idx;
    timemory_begin_record("foo", &idx);
    // ...
    timemory_end_record(idx);
}

void bar()
{
    uint64_t idx = timemory_get_begin_record("bar");
    // ...
    timemory_end_record(idx);
}
```

```cpp
timemory_push_components("papi_vector");
timemory_push_region("MY_REGION_OF_INTEREST");
//
// do something in region of interest...
//
timemory_pop_region("MY_REGION_OF_INTEREST");
```

### Fortran

```fortran
call timemory_push_components("papi_vector")
call timemory_push_region("MY_REGION_OF_INTEREST")
!
! do something in region of interest...
!
call timemory_pop_region("MY_REGION_OF_INTEREST")
```

### C++ Toolkit Interface

```cpp
using hwcounters_t = tim::component_tuple<tim::component::papi_vector>;

hwcounters_t roi("MY_REGION_OF_INTEREST");
roi.start();
//
// do something in region of interest...
//
roi.stop();
// access to data
auto hwc = roi.get();
// print
std::cout << hwc << '\n';
```

Or encoding the PAPI enumeration types explicitly:

```cpp
using hwcounters_t = tim::component_tuple<
    tim::component::papi_tuple<PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM>>;

hwcounters_t roi("MY_REGION_OF_INTEREST");
roi.start();
//
// do something in region of interest...
//
roi.stop();
// access to data
auto hwc = roi.get();
// print
std::cout << hwc << '\n';
```

### Python Context Manager

```python
from timemory.util import marker

with marker(["papi_vector"], key="MY_REGION_OF_INTEREST"):
    #
    # do something in region of interest...
    #
```

> Note: `from timemory.profiler import profile` and `from timemory.trace import trace` can replace `marker` above

### Python Decorator

```python
from timemory.util import marker

@marker(["papi_vector"])
def my_function_of_interest():
    #
    # do something in region of interest...
    #
```

> Note: `from timemory.profiler import profile` and `from timemory.trace import trace` can replace `marker` above

### Python Component Class

```python
from timemory.component import PapiVector

hwc = PapiVector("MY_REGION_OF_INTEREST")  # label when printing
hwc.start()
#
# do something in region of interest...
#
hwc.stop()
l1_tcm, l2_tcm, l3_tcm = hwc.get()  # get values
print("{}".format(hwc))             # print with label and value(s)
```

### C Enumeration Interface

```cpp
void* roi = TIMEMORY_BLANK_MARKER("MY_REGION_OF_INTEREST", PAPI_VECTOR);
//
// do something in region of interest...
//
FREE_TIMEMORY_MARKER(roi)
```

### Python

Timemory provides an extensive suite of Python utilities. Users are encouraged to
use the built-in `help(...)` manual pages from the Python interpreter for the
most extensive details. The `timemory.util` submodule provides decorators and
context managers to generic bundles of components. The `timemory.profiler`
submodule provides an implementation which instruments every Python interpreter
call in the scope of the profiling instance. In general, components can be
specified through lists/tuples of strings (use `timemory-avail -s` to see the string IDs of
the components) or the `timemory.component` enumeration values. Timemory
also provides stand-alone Python classes for each component in `timemory.components`
(note the `"s"` at the end). The stand-alone Python classes behave slightly
differently in that they do not implicitly interact with the persistent timemory
storage classes which track call-stack hierarchy and therefore require a
an invocation of `push()` before `start()` is invoked and an invocation of
`pop()` after `stop()` is invoked in order to show up correctly in the call-stack
tracing. In the absence of a `push()` and `pop()` operation, these classes
map to the underlying invocation of the tool with no overhead.

## Environment Controls

The vast majority of the environment variables can be viewed using the `timemory-avail` executable with the `-S/--settings` option.
Additionally, the `<OUTPUT_PATH>/metadata.json` file will record all the environment variables during the simulation. In particular,
some dynamically generated environment variables for components and variadic bundlers appear in the `metadata.json` file
and not in the `timemory-avail` output.
