# timemory

## Timing + Memory + Hardware Counter Utilities for C / C++ / CUDA / Python

[![Build Status](https://travis-ci.org/NERSC/timemory.svg?branch=master)](https://travis-ci.org/NERSC/timemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1/branch/master?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory/branch/master)
[![codecov](https://codecov.io/gh/NERSC/timemory/branch/master/graph/badge.svg)](https://codecov.io/gh/NERSC/timemory)

[timemory on GitHub (Source code)](https://github.com/NERSC/timemory)

[timemory General Documentation (ReadTheDocs)](https://timemory.readthedocs.io)

[timemory Source Code Documentation (Doxygen)](https://timemory.readthedocs.io/en/latest/doxygen-docs/)

[timemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

[timemory Tutorials](https://github.com/NERSC/timemory-tutorials)

- [ECP 2021 Tutorial Day 1 (YouTube)](https://www.youtube.com/watch?v=K1Pazcw7zVo)

- [ECP 2021 Tutorial Day 2 (YouTube)](https://www.youtube.com/watch?v=-zIpZDiwrmI)

[timemory Wiki](https://github.com/NERSC/timemory/wiki)

|             |                                                   |
| ----------- | ------------------------------------------------- |
| GitHub      | `git clone https://github.com/NERSC/timemory.git` |
| PyPi        | `pip install timemory`                            |
| Spack       | `spack install timemory`                          |
| conda-forge | `conda install -c conda-forge timemory`           |
|             | [![Conda Recipe](https://img.shields.io/badge/recipe-timemory-green.svg) ![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/timemory.svg) ![Conda Version](https://img.shields.io/conda/vn/conda-forge/timemory.svg) ![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/timemory.svg)](https://anaconda.org/conda-forge/timemory) |

## Purpose

The goal of timemory is to create an open-source performance measurement and analyis package
with modular and reusable components which can be used to adapt to any existing C/C++
performance measurement and analysis API and is arbitrarily extendable by users within their
application.
Timemory is not just another profiling tool, it is a profling _toolkit_ which streamlines building custom
profiling tools through modularity and then utilizes the toolkit to provides several pre-built tools.

In other words, timemory provides many pre-built tools, libraries, and interfaces but, due to it's modularity,
codes can re-use only individual pieces -- such as the classes for measuring different timing intervals, memory usage,
and hardware counters -- without the timemory "runtime management".

## Building and Installing

Timemory uses a standard CMake installation.
Several installation examples can be found in the [Wiki](https://github.com/NERSC/timemory/wiki/Installation-Examples). See the [installation documentation](https://timemory.readthedocs.io/en/develop/installation.html) for detailed information on the CMake options.

## Documentation

The full documentation is available at [timemory.readthedocs.io](https://timemory.readthedocs.io).
Detailed source documentation is provided in the [doygen](https://timemory.readthedocs.io/en/latest/doxygen-docs/)
section of the full documentation.
Tutorials are available in the [github.com/NERSC/timemory-tutorials](https://github.com/NERSC/timemory-tutorials).

## Overview

__*The primary objective of the timemory is the development of a common framework for binding together software
monitoring code (i.e. performance analysis, debugging, logging) into a compact and highly-efficient interface.*__

Timemory arose out of the need for a universal adapator kit for the various APIs provided several existing tools
and a straight-forward and intuitive method for creating new tools. Timemory makes it possible to bundle
together deterministic performance measurements, statistical performance
measurements (i.e. sampling), debug messages, data logging, and data validation into the same interface for
custom application-specific software monitoring interfaces, easily building tools like `time`,
`netstat`, instrumentation profilers, sampling profilers, and writing implementations for MPI-P, MPI-T, OMPT,
KokkosP, etc. Furthermore, timemory can forward its markers to several third-party profilers such as
[LIKWID](https://github.com/RRZE-HPC/likwid), [Caliper](https://github.com/LLNL/Caliper),
[TAU](https://www.cs.uoregon.edu/research/tau/home.php), [gperftools](https://github.com/gperftools/gperftools),
[Perfetto](https://perfetto.dev/docs/), VTune, Allinea-MAP, CrayPAT, Nsight-Systems, Nsight-Compute, and NVProf.

Timemory provides a front-end [C/C++/Fortran API](https://timemory.readthedocs.io/en/develop/api/library.html)
and [Python API](https://timemory.readthedocs.io/en/develop/api/python.html) which allows arbitrary selection
of 50+ different components from timers to hardware counters to interfaces with third-party tools. This is all
built generically from the toolkit API with type-safe bundles of tools such as:
`component_tuple<wall_clock, papi_vector, nvtx_marker, user_bundle>`
where `wall_clock` is a wall-clock timer,
`papi_vector` is a handle for hardware counters,
`nvxt_marker` creates notations in the NVIDIA CUDA profilers, and
`user_bundle` is a generic component which downstream users can insert more components into at runtime.

Performance measurement components written with timemory are arbitrarily scalable up to any number of threads and
processes and fully support intermixing different measurements at different locations within the program -- this
uniquely enables timemory to be deployed to collect performance data at scale in HPC because highly detailed collection can
occur at specific locations within the program where ubiquitous collection would simulatenously degrade performance
significantly and require a prohibitive amount of memory.

Timemory can be used as a backend to bundle instrumentation and sampling tools together, support serialization to JSON/XML,
and provide statistics among other uses. It can also be utilized as a front-end to invoke
custom instrumentation and sampling tools. Timemory uses the abstract term "component" for a structure
which encapsulates some performance analysis operation. The structure might encapsulate function
calls to another tool, record timestamps for timing, log values provided by the application,
provide a operator for replacing a function in the code dynamically, audit the incoming arguments
and/or outgoing return value from function, or just provide stubs which can be overloaded by the linker.

### Visualization and Analysis

The native output format of timemory is JSON and text; other output formats such as XML are also supported.
The text format is intended to be human readable. The JSON data
is intended for analysis and comes in two flavors: hierarchical and flat. Basic plotting capabilities are
available via `timemory-plotting` but users are highly encouraged to use [hatchet](https://github.com/hatchet/hatchet)
for analyzing the heirarchical JSON data in pandas dataframes. [Hatchet](https://github.com/hatchet/hatchet) supports
filtering, unions, addition, subtractions, output to `dot` and flamegraph formats, and an interactive Jupyter notebook.
At present, timemory supports 45+ metric types for analysis in Hatchet.

### Categories

There are 4 primary categories in timemory: components, operations, bundlers, and storage. Components provide
the specifics of how to perform a particular behavior, operations provide the scaffold for requesting that
a component perform an operation in complex scenarios, bundlers group components into a single generic handle,
and storage manages data collection over the lifetime of the application. When all four categories are combined,
timemory effectively resembles a standard performance analysis tool which passively collects data and provides
reports and analysis at the termination of the application. Timemory, however, makes it _very easy_ to subtract
storage from the equation and, in doing so, transforms timemory into a toolkit for customized data collection.

1. __*Components*__
   - Individual classes which encapsulate one or more measurement, analysis, logging, or third-party library action(s)
   - Any data specific to one instance of performing the action is stored within the instance of the class
   - Any configuration data specific to that type is typically stored within static member functions which return a reference to the configuration data
   - These classes are designed to support direct usage within other tools, libraries, etc.
   - Examples include:
     - `tim::component::wall_clock` : a simple wall-clock timer
     - `tim::component::vtune_profiler` : a simple component which turns the VTune Profiler on and off (when VTune is actively profiling application)
     - `tim::component::data_tracker_integer` : associates an integer values with a label as the application executes (e.g. number of loop iterations used somewhere)
     - `tim::component::papi_vector` : uses the PAPI library to collect hardware-counters values
     - `tim::component::user_bundle` : encapsulates an array of components which the user can dynamically manipulate during runtime
2. __*Operations*__
   - Templated classes whose primary purpose is to provide the implementation for performing some action on a component, e.g. `tim::operation::start<wall_clock>` will attempt to call the `start()` member function on a `wall_clock` component instance
   - Default implementations generally have one or two public functions: a constructor and/or a function call operator
     - These generally accept any/all arguments and use SFINAE to determine whether the operation can be performed with or without the given arguments (i.e. does `wall_clock` have a `store(int)` function? `store()`?)
   - Operations are (generally) not directly utilized by the user and are typically optimized out of the binary
   - Examples include:
     - `tim::operation::start` : instruct a component to start collection
     - `tim::operation::sample` : instruct a component to take individual measurement
     - `tim::operation::derive` : extra data from other components if it is available
3. __*Bundlers*__
   - Provide a generic handle for multiple components
   - Member functions generally accept any/all arguments and use operations classes to correctly to handle differences between different capabilities of the components it is bundling
   - Examples include:
     - `tim::auto_tuple`
     - `tim::component_tuple`
     - `tim::component_list`
     - `tim::lightweight_tuple`
   - Various flavors provide different implicit behaviors and allocate memory differently
     - `auto_tuple` starts all components when constructed and stops all components when destructed whereas `component_tuple` requires an explicit start
     - `component_tuple` allocates all components on the stack and components are "always on" whereas `component_list` allocates components on the heap and thus components can be activated/deactivated at runtime
     - `lightweight_tuple` does not implicitly perform any expensive actions, such as call-stack tracking in "Storage"
4. __*Storage*__
   - Provides persistent storage for multiple instances of components over the lifetime of a thread in the application
   - Responsible for maintaining the hierarchy and order of component measurements, i.e. call-stack tracking
   - Responsible for combining component data from multiple threads and/or processes and outputting the results

> NOTE: `tim::lightweight_tuple` is the recommended bundle for those seeking to use timemory as a toolkit for implementing custom tools and interfaces

## Features

- C++ Template API
    - Modular and fully-customizable
    - Adheres to C++ standard template library paradigm of "you don't pay for what you don't use"
    - Simplifies and facilitates creation and implementation of performance measurement tools
        - Create your own instrumentation profiler
        - Create your own instrumentation library
        - Create your own sampling profiler
        - Create your own sampling library
        - Create your own execution wrappers
        - Supplement timemory-provided tools with your own custom component(s)
        - Thread-safe data aggregation
        - Aggregate collection over multiple processes (MPI and UPC++ support)
        - Serialization to text, JSON, XML
    - Components are composable with other components
    - Variadic component bundlers which maintain complete type-safety
        - Components can be bundled together into a single handle without abstractions
    - Components can store data in any valid C++ data type
    - Components can return data in any valid C++ data type
- C / C++ / CUDA / Fortran Library API
    - Straight-forward collection of functions and macros for creating built-in performance analysis to your code
    - Component collection can be arbitrarily inter-mixed
        - E.g. collect "A" and "B" in one region, "A" and "C" in another region
    - Component collection can be dynamically manipulated at runtime
        - E.g. add/remove "A" at any point, on any thread, on any process
- Python API
    - Decorators and context-managers for functions or regions in code
    - Python function profiling
    - Python line-by-line profiling
    - Every component in `timemory-avail` is provided as a stand-alone Python class
        - Provide low-overhead measurements for building your own Python profiling tools
- Python Analysis via [pandas](https://pandas.pydata.org/)
- Command-line Tools
    - [timemory-avail](source/tools/timemory-avail/README.md)
        - Provides available components, settings, and hardware counters
        - Quick API reference tool
    - [timem](source/tools/timem/README.md) (UNIX)
        - Extended version of UNIX `time` command-line tool that includes additional information on memory usage, context switches, and hardware counters
        - Support collecting hardware counters (Linux-only, requires PAPI)
    - [timemory-run](source/tools/timemory-run/README.md) (Linux)
        - Dynamic instrumentation profiling tool
        - Supports runtime instrumentation and binary re-writing
    - [timemory-nvml](source/tools/timemory-nvml/README.md)
        - Data collection similar to `nvidia-smi`
    - `timemory-python-profiler`
        - Python function profiler supporting all timemory components
        - `from timemory.profiler import Profile`
    - `timemory-python-trace`
        - Python line-by-line profiler supporting all timemory components
        - `from timemory.trace import Trace`
- Instrumentation Libraries
    - [timemory-mpip](source/tools/timemory-mpip/README.md): MPI Profiling Library (Linux-only)
    - [timemory-ncclp](source/tools/timemory-ncclp/README.md): NCCL Profiling Library (Linux-only)
    - [timemory-ompt](source/tools/timemory-ompt/README.md): OpenMP Profiling Library
    - [timemory-compiler-instrument](source/tools/timemory-compiler-instrument/README.md): Compiler instrumentation Library
    - [kokkos-connector](source/tools/kokkos-connector/README.md): Kokkos Profiling Libraries

## Samples

Various macros are defined for C in [source/timemory/compat/timemory_c.h](source/timemory/timemory.h)
and [source/timemory/variadic/macros.hpp](source/timemory/variadic/macros.hpp). Numerous samples of
their usage can be found in the examples.

### Sample C++ Template API

```cpp
#include "timemory/timemory.hpp"

namespace comp = tim::component;
using namespace tim;

// specific set of components
using specific_t = component_tuple<comp::wall_clock, comp::cpu_clock>;
using generic_t  = component_tuple<comp::user_global_bundle>;

int
main(int argc, char** argv)
{
    // configure default settings
    settings::flat_profile() = true;
    settings::timing_units() = "msec";

    // initialize with cmd-line
    timemory_init(argc, argv);

    // add argparse support
    timemory_argparse(&argc, &argv);

    // create a region "main"
    specific_t m{ "main" };
    m.start();
    m.stop();

    // pause and resume collection globally
    settings::enabled() = false;
    specific_t h{ "hidden" };
    h.start().stop();
    settings::enabled() = true;

    // Add peak_rss component to specific_t
    mpl::push_back_t<specific_t, comp::peak_rss> wprss{ "with peak_rss" };

    // create region collecting only peak_rss
    component_tuple<comp::peak_rss> oprss{ "only peak_rss" };

    // convert component_tuple to a type that starts/stops upon construction/destruction
    {
        scope::config _scope{};
        if(true)  _scope += scope::flat{};
        if(false) _scope += scope::timeline{};
        convert_t<specific_t, auto_tuple<>> scoped{ "scoped start/stop + flat", _scope };
        // will yield auto_tuple<comp::wall_clock, comp::cpu_clock>
    }

    // configure the generic bundle via set of strings
    runtime::configure<comp::user_global_bundle>({ "wall_clock", "peak_rss" });
    // configure the generic bundle via set of enumeration ids
    runtime::configure<comp::user_global_bundle>({ TIMEMORY_WALL_CLOCK, TIMEMORY_CPU_CLOCK });
    // configure the generic bundle via component instances
    comp::user_global_bundle::configure<comp::page_rss, comp::papi_vector>();

    generic_t g{ "generic", quirk::config<quirk::auto_start>{} };
    g.stop();

    // Output the results
    timemory_finalize();
    return 0;
}
```

### Sample C / C++ Library API

```cpp
#include "timemory/library.h"
#include "timemory/timemory.h"

int
main(int argc, char** argv)
{
    // configure settings
    int overwrite       = 0;
    int update_settings = 1;
    // default to flat-profile
    timemory_set_environ("TIMEMORY_FLAT_PROFILE", "ON", overwrite, update_settings);
    // force timing units
    overwrite = 1;
    timemory_set_environ("TIMEMORY_TIMING_UNITS", "msec", overwrite, update_settings);

    // initialize with cmd-line
    timemory_init_library(argc, argv);

    // check if inited, init with name
    if(!timemory_library_is_initialized())
        timemory_named_init_library("ex-c");

    // define the default set of components
    timemory_set_default("wall_clock, cpu_clock");

    // create a region "main"
    timemory_push_region("main");
    timemory_pop_region("main");

    // pause and resume collection globally
    timemory_pause();
    timemory_push_region("hidden");
    timemory_pop_region("hidden");
    timemory_resume();

    // Add/remove component(s) to the current set of components
    timemory_add_components("peak_rss");
    timemory_remove_components("peak_rss");

    // get an identifier for a region and end it
    uint64_t idx = timemory_get_begin_record("indexed");
    timemory_end_record(idx);

    // assign an existing identifier for a region
    timemory_begin_record("indexed/2", &idx);
    timemory_end_record(idx);

    // create region collecting a specific set of data
    timemory_begin_record_enum("enum", &idx, TIMEMORY_PEAK_RSS, TIMEMORY_COMPONENTS_END);
    timemory_end_record(idx);

    timemory_begin_record_types("types", &idx, "peak_rss");
    timemory_end_record(idx);

    // replace current set of components and then restore previous set
    timemory_push_components("page_rss");
    timemory_pop_components();

    timemory_push_components_enum(2, TIMEMORY_WALL_CLOCK, TIMEMORY_CPU_CLOCK);
    timemory_pop_components();

    // Output the results
    timemory_finalize_library();
    return 0;
}
```

### Sample Fortran API

```fortran
program fortran_example
    use timemory
    use iso_c_binding, only : C_INT64_T
    implicit none
    integer(C_INT64_T) :: idx

    ! initialize with explicit name
    call timemory_init_library("ex-fortran")

    ! initialize with name extracted from get_command_argument(0, ...)
    ! call timemory_init_library("")

    ! define the default set of components
    call timemory_set_default("wall_clock, cpu_clock")

    ! Start region "main"
    call timemory_push_region("main")

    ! Add peak_rss to the current set of components
    call timemory_add_components("peak_rss")

    ! Nested region "inner" nested under "main"
    call timemory_push_region("inner")

    ! End the "inner" region
    call timemory_pop_region("inner")

    ! remove peak_rss
    call timemory_remove_components("peak_rss")

    ! begin a region and get an identifier
    idx = timemory_get_begin_record("indexed")

    ! replace current set of components
    call timemory_push_components("page_rss")

    ! Nested region "inner" with only page_rss components
    call timemory_push_region("inner (pushed)")

    ! Stop "inner" region with only page_rss components
    call timemory_pop_region("inner (pushed)")

    ! restore previous set of components
    call timemory_pop_components()

    ! end the "indexed" region
    call timemory_end_record(idx)

    ! End "main"
    call timemory_pop_region("main")

    ! Output the results
    call timemory_finalize_library()

end program fortran_example
```

### Sample Python API

#### Decorator

```python
from timemory.bundle import marker

@marker(["cpu_clock", "peak_rss"])
def foo():
    pass
```

#### Context Manager

```python
from timemory.profiler import profile

def bar():
    with profile(["wall_clock", "cpu_util"]):
        foo()
```

#### Individual Components

```python
from timemory.component import WallClock

def spam():

    wc = WallClock("spam")
    wc.start()

    bar()

    wc.stop()
    data = wc.get()
    print(data)
```

#### Argparse Support

```python
import argparse

parser = argparse.ArgumentParser("example")
# ...
timemory.add_arguments(parser)

args = parser.parse_args()
```

#### Component Storage

```python
from timemory.storage import WallClockStorage

# data for current rank
data = WallClockStorage.get()
# combined data on rank zero but all ranks must call it
dmp_data = WallClockStorage.dmp_get()
```

## Versioning

Timemory originated as a very simple tool for recording timing and memory measurements (hence the name) in C, C++, and Python and only supported
three modes prior to the 3.0.0 release: a fixed set of timers, a pair of memory measurements, and the combination of the two.
__Prior to the 3.0.0 release, timemory was almost completely rewritten from scratch__ with the sole exceptions of some C/C++ macro, e.g.
`TIMEMORY_AUTO_TIMER`, and some Python decorators and context-manager, e.g. `timemory.util.auto_timer`, whose behavior were
able to be fully replicated in the new release. Thus, while it may appear that timemory is a mature project at v3.0+, it
is essentially still in it's first major release.

## Citing timemory

To reference timemory in a publication, please cite the following paper:

- Madsen, J.R. et al. (2020) Timemory: Modular Performance Analysis for HPC. In: Sadayappan P., Chamberlain B., Juckeland G., Ltaief H. (eds) High Performance Computing. ISC High Performance 2020. Lecture Notes in Computer Science, vol 12151. Springer, Cham

## Additional Information

For more information, refer to the [documentation](https://timemory.readthedocs.io/en/latest/).
