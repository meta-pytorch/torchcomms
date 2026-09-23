# timemory-ncclp library

Produces a `libtimemory-ncclp.so` that uses GOTCHA wrappers around ~12 NCCL functions.

Four functions are provided for C, C++, and Fortran:

- `uint64_t timemory_start_ncclp()`
  - Returns the number of initializations
- `uint64_t timemory_stop_ncclp(uint64_t idx)`
  - Removes the initialization request at `idx`
  - Returns the number of remaining initializations
- `void timemory_register_ncclp()`
  - Ensures a global initialization exists until it deregistration
- `void timemory_deregister_ncclp()`
  - Deactivates the global initialization

## Usage

The environement variable `ENABLE_TIMEMORY_NCCLP` (default: `"ON"`) controls configuration of the instrumentation.
This library configures the `tim::user_ncclp_bundle` component with the components specified by the following environment variables in terms of priority:

- `TIMEMORY_NCCLP_COMPONENTS`
- `TIMEMORY_MPIP_COMPONENTS`
- `TIMEMORY_PROFILER_COMPONENTS`
- `TIMEMORY_GLOBAL_COMPONENTS`
- `TIMEMORY_COMPONENT_LIST_INIT`

When one of the above environment variables are set to `"none"`, then the priority search for component configurations is abandoned.

### Examples

The following will result in NCCL function instrumented with `cpu_clock`:

```console
export TIMEMORY_NCCLP_COMPONENTS="cpu_clock"
export TIMEMORY_PROFILER_COMPONENTS="peak_rss"
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock"
```

The following will result in NCCL functions containing no instrumentation:

```console
export TIMEMORY_NCCLP_COMPONENTS="none"
export TIMEMORY_PROFILER_COMPONENTS="peak_rss"
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock"
```

The following will result in NCCL function instrumented with `wall_clock` and `page_rss`:

```console
export TIMEMORY_NCCLP_COMPONENTS=""
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock,page_rss"
```
