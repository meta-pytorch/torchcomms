# Components

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2

   components/gotcha
   components/roofline
   components/custom_components
```

## Overview

This is an overview of the components available in timemory. For detailed info on the
member functions, etc. please refer to the [Doxygen](doxygen).

The component documentation below is categorized into some general subsections and then sorted alphabetically.
In general, which member function are present are not that important as long
as you use the variadic component bundlers -- these handle ignoring
trying to call `start()` on a component is the component does not have
a `start()` member function but is bundled alongside other components
which do (that the `start()` was intended for).

## Component Basics

Timemory components are C++ structs (class which defaults to `public` instead of `private`) which
define a single collection instance, e.g. the `wall_clock` component is written as a simple class
with two 64-bit integers with `start()` and `stop()` member functions.

```cpp
// This "component" is for conceptual demonstration only
// It is not intended to be copy+pasted
struct wall_clock
{
    int64_t m_value = 0;
    int64_t m_accum = 0;

    void start();
    void stop();
};
```

The `start()` member function which records a timestamp
and assigns it to one of the integers temporarily, the `stop()` member function
which records another timestamp, computes the difference and then assigns the difference
to the first integer and adds the difference to the second integer.

```cpp
void wall_clock::start()
{
    m_value = get_timestamp();
}

void wall_clock::stop()
{
    // compute difference b/t when start and stop were called
    m_value = (get_timestamp() - m_value);
    // accumulate the difference
    m_accum += m_value;
}
```

Thus, after `start()` and `stop()` is invoked twice on the object:

```cpp
wall_clock foo;

foo.start();
sleep(1); // sleep for 1 second
foo.stop();

foo.start();
sleep(1); // sleep for 1 second
foo.stop();
```

The first integer (`m_value`) represents the _most recent_ timing interval of 1 second
and the second integer (`m_accum`) represents the _accumulated_ timing interval totaling 2 seconds.
This design not only encapsulates how to take the measurement, but also provides it's own
data storage model. With this design, timemory measurements naturally support asynchronous
data collection. Additionally, as part of the design for generating the call-graph,
call-graphs are accumulated locally on each thread and on each process and merged at
the termination of the thread or process. This allows parallel data to be collection
free from synchronization overheads. On the worker threads, there is a concept of being
at "sea-level" -- the call-graphs relative position based on the base-line of the
primary thread in the application. When a worker thread is at sea-level, it reads the
position of the call-graph on the primary thread and creates a copy of that entry
in it's call-graph, ensuring that when merged into the primary thread at the end,
the accumulated call-graph across all threads is inserted into the appropriate
location. This approach has been found to produce the fewest number of artifacts.

In general, components do not need to conform to a specific interface. This is
relatively unique approach. Most performance analysis which allow user extensions
use callbacks and dynamic polymorphism to integrate the user extensions into their
workflow. It should be noted that there is nothing preventing a component from creating a
similar system but timemory is designed to query the presence of member function _names_ for
feature detection and adapts accordingly to the overloads of that function name and
it's return type. This is all possible due to the template-based design which makes
extensive use of variadic functions to accept any arguments at a high-level and
SFINAE to decide at compile-time which function to invoke (if a function is invoked at all).
For example:

- component A can contain these member functions:
    - `void start()`
    - `int get()`
    - `void set_prefix(const char*)`
- component B can contains these member functions:
    - `void start()`
    - `void start(cudaStream_t)`
    - `double get()`
- component C can contain these member functions:
    - `void start()`
    - `void set_prefix(const std::string&)`

And for a given bundle `component_tuple<A, B, C> obj`:

- When `obj` is created, a string identifer, instance of a `source_location` struct, or a hash is required
    - This is the label for the measurement
    - If a string is passed, `obj` generates the hash and adds the hash and the string to a hash-map if it didn't previously exist
    - `A::set_prefix(const char*)` will be invoked with the underlying `const char*` from the string that the hash maps to in the hash-map
    - `C::set_prefix(const std::string&)` will be invoked with string that the hash maps to in the hash-map
    - It will be detected that `B` does not have a member function named `set_prefix` and no member function will be invoked
- Invoking `obj.start()` calls the following member functions on instances of A, B, and C:
    - `A::start()`
    - `B::start()`
    - `C::start()`
- Invoking `obj.start(cudaStream_t)` calls the following member functions on instances of A, B, and C:
    - `A::start()`
    - `B::start(cudaStream_t)`
    - `C::start()`
- Invoking `obj.get()`:
    - Returns `std::tuple<int, double>` because it detects the two return types from A and B and the lack of `get()` member function in component C.

This design makes has several benefits and one downside in particular. The benefits
are that timemory: (1) makes it extremely easy to create a unified interface between two
or more components which different interfaces/capabilities, (2) invoking
the different interfaces is efficient since no feature detection logic is required at
runtime, and (3) components define their own interface.

With respect to #2, consider the two more traditional implementations. If callbacks are
used, a function pointer exists and a component which does not implement a feature
will either have a null function pointer (requiring a check at runtime time) or the
tool will implement an array of function pointers with an unknown size at compile-time.
In the latter case, this will require heap allocations (which are expensive operations) and
in both cases, the loop of the function pointers will likely be quite ineffienct
since function pointers have a very high probability of thrashing the instruction cache.
If dynamic polymorphism is used, then virtual table look-ups are required
during every iteration. In the timemory approach, none of these additional overheads
are present and there isn't even a loop -- the bundle either expands into a direct call to the
member function without any abstractions or nothing.

With respect to #1 and #3, this has some interesting implications with regard to a
universal instrumentation interface and is discussed in the following section and
the [CONTRIBUTING.md](CONTRIBUTING.md) documentation.

The aforementioned downside is that the byproduct of all this flexibility and adaption
to custom interfaces by each component is that directly using the template interface
can take quite a long time to compile.

## Component Metadata

```eval_rst
.. doxygenstruct:: tim::component::enumerator
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::metadata
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::properties
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::static_properties
   :members:
   :undoc-members:
```

## Timing Components

```eval_rst
.. doxygenstruct:: tim::component::cpu_clock

.. doxygenstruct:: tim::component::cpu_util

.. doxygenstruct:: tim::component::kernel_mode_time

.. doxygenstruct:: tim::component::monotonic_clock

.. doxygenstruct:: tim::component::monotonic_raw_clock

.. doxygenstruct:: tim::component::process_cpu_clock

.. doxygenstruct:: tim::component::process_cpu_util

.. doxygenstruct:: tim::component::system_clock

.. doxygenstruct:: tim::component::thread_cpu_clock

.. doxygenstruct:: tim::component::thread_cpu_util

.. doxygenstruct:: tim::component::user_clock

.. doxygenstruct:: tim::component::user_mode_time

.. doxygenstruct:: tim::component::wall_clock
```

## Resource Usage Components

```eval_rst
.. doxygenstruct:: tim::component::current_peak_rss

.. doxygenstruct:: tim::component::num_io_in

.. doxygenstruct:: tim::component::num_io_out

.. doxygenstruct:: tim::component::num_major_page_faults

.. doxygenstruct:: tim::component::num_minor_page_faults

.. doxygenstruct:: tim::component::page_rss

.. doxygenstruct:: tim::component::peak_rss

.. doxygenstruct:: tim::component::priority_context_switch

.. doxygenstruct:: tim::component::virtual_memory

.. doxygenstruct:: tim::component::voluntary_context_switch
```

## I/O Components

```eval_rst
.. doxygenstruct:: tim::component::read_bytes

.. doxygenstruct:: tim::component::read_char

.. doxygenstruct:: tim::component::written_bytes

.. doxygenstruct:: tim::component::written_char
```

## User Bundle Components

Timemory provides the `user_bundle` component as a generic component bundler
that the user can use to insert components at runtime. This component is
heavily used when mapping timemory to languages other than C++. Timemory
implements many specialization of this template class for various tools.
For example, `user_mpip_bundle` is the bundle used by the MPI wrappers,
`user_profiler_bundle` is used by the Python function profiler,
`user_trace_bundle` is used by the dynamic instrumentation tool `timemory-run` and
the Python line tracing profiler, etc. These specialization are
all individually configurable and it is recommended that applications create
their own specialization specific to their project -- this will ensure that
the desired set of components configured by your application will not be
affected by a third-party library configuring their own set of components.

The general design is that each user-bundle:

- Has their own unique environment variable for exclusive configuration, usually `"TIMEMORY_<LABEL>_COMPONENTS"`, e.g.:
  - `"TIMEMORY_TRACE_COMPONENTS"` for `user_trace_bundle`
  - `"TIMEMORY_MPIP_COMPONENTS"` for `user_mpip_components`
- If the unique environment variable is set, only the components in the variable are used
  - Thus making the bundle uniquely configurable
- If the unique environment variable is *not* set, it searches one or more _backup_ environment variables, the last of which being `"TIMEMORY_GLOBAL_COMPONENTS"`
  - Thus, if no specific environment variables are set, all user bundles collect the components specified in `"TIMEMORY_GLOBAL_COMPONENTS"`
- If the unique environment variable is set to `"none"`, it terminates searching the backup environment variables
  - Thus, `"TIMEMORY_GLOBAL_COMPONENTS"` can be set but the user can suppress a specific bundle from being affected by this configuration
- If the unique environment variable contains `"fallthrough"`, it will continue adding the components specified by the backup environment variables
  - Thus, the components specified in `"TIMEMORY_GLOBAL_COMPONENTS"` and `"TIMEMORY_<LABEL>_COMPONENTS"` will be added

```eval_rst
.. doxygenstruct:: tim::component::user_bundle

.. doxygentypedef:: tim::component::user_global_bundle

.. doxygentypedef:: tim::component::user_mpip_bundle

.. doxygentypedef:: tim::component::user_ncclp_bundle

.. doxygentypedef:: tim::component::user_ompt_bundle

.. doxygentypedef:: tim::component::user_profiler_bundle

.. doxygentypedef:: tim::component::user_trace_bundle

.. doxygentypedef:: tim::component::user_kokkosp_bundle

```

## Third-Party Interface Components

```eval_rst
.. doxygenstruct:: tim::component::allinea_map

.. doxygenstruct:: tim::component::caliper_marker

.. doxygenstruct:: tim::component::caliper_config

.. doxygenstruct:: tim::component::caliper_loop_marker

.. doxygenstruct:: tim::component::craypat_counters

.. doxygenstruct:: tim::component::craypat_flush_buffer

.. doxygenstruct:: tim::component::craypat_heap_stats

.. doxygenstruct:: tim::component::craypat_record

.. doxygenstruct:: tim::component::craypat_region

.. doxygenstruct:: tim::component::gperftools_cpu_profiler

.. doxygenstruct:: tim::component::gperftools_heap_profiler

.. doxygenstruct:: tim::component::likwid_marker

.. doxygenstruct:: tim::component::likwid_nvmarker

.. doxygenstruct:: tim::component::ompt_handle

.. doxygenstruct:: tim::component::tau_marker

.. doxygenstruct:: tim::component::vtune_event

.. doxygenstruct:: tim::component::vtune_frame

.. doxygenstruct:: tim::component::vtune_profiler

```

## Hardware Counter Components

```eval_rst
.. doxygenstruct:: tim::component::papi_tuple

.. doxygenstruct:: tim::component::papi_rate_tuple

.. doxygenstruct:: tim::component::papi_array

.. doxygenstruct:: tim::component::papi_vector

```

## Miscellaneous Components

```eval_rst
.. doxygenstruct:: tim::component::cpu_roofline

.. doxygentypedef:: tim::component::cpu_roofline_dp_flops

.. doxygentypedef:: tim::component::cpu_roofline_flops

.. doxygentypedef:: tim::component::cpu_roofline_sp_flops

```

## GPU Components

```eval_rst
.. doxygenstruct:: tim::component::cuda_event

.. doxygenstruct:: tim::component::cupti_activity

.. doxygenstruct:: tim::component::cupti_counters

.. doxygenstruct:: tim::component::cupti_profiler

.. doxygenstruct:: tim::component::gpu_roofline

.. doxygentypedef:: tim::component::gpu_roofline_dp_flops

.. doxygentypedef:: tim::component::gpu_roofline_flops

.. doxygentypedef:: tim::component::gpu_roofline_hp_flops

.. doxygentypedef:: tim::component::gpu_roofline_sp_flops

.. doxygenstruct:: tim::component::nvtx_marker
   :members:

.. doxygenstruct:: tim::component::hip_event

.. doxygenstruct:: tim::component::roctx_marker
   :members:
```

## Data Tracking Components

```eval_rst
.. doxygenstruct:: tim::component::data_tracker
   :members:
.. doxygentypedef:: tim::component::data_tracker_integer

.. doxygentypedef:: tim::component::data_tracker_unsigned

.. doxygentypedef:: tim::component::data_tracker_floating
```

## Function Wrapping Components

```eval_rst
.. doxygenstruct:: tim::component::gotcha
   :members:
.. doxygenstruct:: tim::component::malloc_gotcha
   :members:
.. doxygenstruct:: tim::component::memory_allocations
   :members:
```

## Base Components

```eval_rst
.. doxygenstruct:: tim::component::base
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::base_state
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::base_data
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::base_format
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::base_units
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::empty_base
   :members:
   :undoc-members:
```
