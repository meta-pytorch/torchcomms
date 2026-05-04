# timemory-ompt library

Produces a `libtimemory-ompt.so` library which uses the OpenMP-tools (OMP-T) interface.

Four functions are provided for C, C++, and Fortran:

- `uint64_t timemory_start_ompt()`
  - Returns the number of initializations
- `uint64_t timemory_stop_ompt(uint64_t idx)`
  - Removes the initialization request at `idx`
  - Returns the number of remaining initializations
- `void timemory_register_ompt()`
  - Ensures a global initialization exists until it deregistration
- `void timemory_deregister_ompt()`
  - Deactivates the global initialization

## Usage

The environement variable `ENABLE_TIMEMORY_OMPT` (default: `"ON"`) controls configuration of the instrumentation.
This library configures the `tim::user_ompt_bundle` component with the components specified by the following environment variables in terms of priority:

- `TIMEMORY_OMPT_COMPONENTS`
- `TIMEMORY_TRACE_COMPONENTS`
- `TIMEMORY_PROFILER_COMPONENTS`
- `TIMEMORY_GLOBAL_COMPONENTS`

When one of the above environment variables are set to `"none"`, then the priority search for component configurations is abandoned.

### Examples

The following will result in MPI function instrumented with `cpu_clock`:

```console
export TIMEMORY_OMPT_COMPONENTS="cpu_clock"
export TIMEMORY_PROFILER_COMPONENTS="peak_rss"
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock"
```

The following will result in MPI functions containing no instrumentation:

```console
export TIMEMORY_OMPT_COMPONENTS="none"
export TIMEMORY_PROFILER_COMPONENTS="peak_rss"
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock"
```

The following will result in MPI function instrumented with `wall_clock` and `page_rss`:

```console
export TIMEMORY_OMPT_COMPONENTS=""
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock,page_rss"
```
