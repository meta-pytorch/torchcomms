# timemory-mallocp library

Produces a `libtimemory-mallocp.so` that uses GOTCHA wrappers around common memory allocation functions:

- `malloc`
- `calloc`
- `free`
- `cudaMalloc` (requires `TIMEMORY_USE_CUDA`)
- `cudaMallocHost` (requires `TIMEMORY_USE_CUDA`)
- `cudaMallocManaged` (requires `TIMEMORY_USE_CUDA`)
- `cudaHostAlloc` (requires `TIMEMORY_USE_CUDA`)
- `cudaFree` (requires `TIMEMORY_USE_CUDA`)
- `cudaFreeHost` (requires `TIMEMORY_USE_CUDA`)

Four functions are provided for C, C++, and Fortran:

- `uint64_t timemory_start_mallocp()`
  - Returns the number of initializations
- `uint64_t timemory_stop_mallocp(uint64_t idx)`
  - Removes the initialization request at `idx`
  - Returns the number of remaining initializations
- `void timemory_register_mallocp()`
  - Ensures a global initialization exists until it deregistration
- `void timemory_deregister_mallocp()`
  - Deactivates the global initialization

## Usage

The environement variable `ENABLE_TIMEMORY_MALLOCP` (default: `"ON"`) controls configuration of the instrumentation.

### C and C++

```cpp
#include <timemory/tools/timemory-mallocp.h>

void foo()
{
    // ... something using memory allocation functions ...
}

void bar(int val)
{
    uint64_t mallocp_idx = UINT64_MAX;
    if(val > 100)
        mallocp_idx = timemory_start_mallocp();

    // ... something using memory allocation functions ...

    if(val > 100)
        timemory_stop_mallocp(mallocp_idx);
}

int main()
{
    timemory_register_mallocp();
    // mallocp will be active for all the following
    foo();
    bar(10);
    bar(1000);
    foo();
    timemory_deregister_mallocp();

    foo();     // mallocp will NOT be activate
    bar(10);   // mallocp will NOT be active
    bar(1000); // mallocp will be activated and deactivated
    foo();     // mallocp will NOT be activate
}
```

### Python

```python
import timemory

def foo():
    # ... something using memory allocation functions ...

if __name__ == "__main__":
    idx = timemory.start_mallocp()
    foo()
    timemory.stop_mallocp(idx)
```

## Example

### Code

```python
import numpy as np
import timemory as tim

tim.settings.memory_units = "KB"

_idx = tim.start_mallocp()
# dummy marker to malloc gotcha to create a hierarchy
# malloc/free calls used by timemory internally when
# creating the marker will be included
with tim.util.marker(["malloc_gotcha"], key="test"):
    _arr = np.ones([1000, 1000], dtype=np.float64)
    _sum = np.sum(_arr)
    del _arr
    gc.collect()
_idx = tim.stop_mallocp(_idx)
```

### Output

```console
|---------------------------------------------------------------------------------------------------------------------------------------------------|
|                                        GOTCHA WRAPPER FOR MEMORY ALLOCATION FUNCTIONS: MALLOC, CALLOC, FREE                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|
|    LABEL     |   COUNT    |   DEPTH    |    METRIC     |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|--------------|------------|------------|---------------|------------|------------|------------|------------|------------|------------|------------|
| >>> malloc   |         21 |          0 | malloc_gotcha | KB         |      0.819 |      0.039 |      0.024 |      0.024 |      0.077 |      100.0 |
| >>> free     |         12 |          0 | malloc_gotcha | KB         |      0.553 |      0.046 |      0.017 |      0.017 |      0.100 |       71.1 |
| >>> |_malloc |          1 |          1 | malloc_gotcha | KB         |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> test     |          1 |          0 | malloc_gotcha | KB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> |_malloc |          4 |          1 | malloc_gotcha | KB         |   8000.072 |   2000.018 |      0.032 |      0.032 |   3999.988 |      100.0 |
| >>> |_free   |          5 |          1 | malloc_gotcha | KB         |   8000.060 |   1600.012 |      0.004 |      0.004 |   3577.702 |      100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------|
```
