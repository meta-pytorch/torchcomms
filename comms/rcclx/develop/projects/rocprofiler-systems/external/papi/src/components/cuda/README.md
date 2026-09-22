# Cuda Component

The `cuda` component exposes counters and controls for NVIDIA GPUs.

* [Enabling the Cuda Component](#enabling-the-cuda-component)
* [Hardware and Software Support](#hardware-and-software-support)
* [Partially Disabled Cuda Component](#partially-disabled-cuda-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***

## Enabling the `cuda` Component

To enable reading or writing of CUDA counters the user needs to link against a
PAPI library that was configured with the `cuda` component. As an example:
```
./configure --with-components="cuda"
```

For the component to be active, PAPI requires one environment variable to be set: `PAPI_CUDA_ROOT`. This environment variable **must** be set to the root of the Cuda Toolkit that is desired to be used for both compiling and runtime. As an example:

```
PAPI_CUDA_ROOT=/packages/cuda/#.#.#
```

Within `PAPI_CUDA_ROOT`, we expect the following standard directories for building:

```
PAPI_CUDA_ROOT/include
PAPI_CUDA_ROOT/extras/CUPTI/include
```

and for runtime:

```
PAPI_CUDA_ROOT/lib64
PAPI_CUDA_ROOT/extras/CUPTI/lib64
```

To verify the `cuda` component was configured with your PAPI build and is active,
run `papi_component_avail` (available in `utils/papi_component_avail`). This 
utility will display the components configured in your PAPI build and whether they are active or disabled. If a component is disabled a message on why the component
has been disabled will be directly below it.

For the `cuda` component to be operational, the following dynamic libraries must be found at runtime for both the Event/Metric APIs and the PerfWorks Metrics API:

```
libcuda.so
libcudart.so
libcupti.so
```

For the PerfWorks Metrics API, the dynamic library `libnvperf_host.so` must also be found.

If those libraries cannot be found or some of those are stub libraries in the
standard `PAPI_CUDA_ROOT` subdirectories, you must add the correct paths,
e.g. `/usr/lib64` or `/usr/lib` to `LD_LIBRARY_PATH`, separated by colons `:`.
This can be set using export; e.g. 

```
export LD_LIBRARY_PATH=$PAPI_CUDA_ROOT/lib64:$LD_LIBRARY_PATH
```
## Hardware and Software Support

To see the `cuda` component's current supported hardware and software please visit the GitHub wiki page [Hardware and Software Support - Cuda Component](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-Cuda-Component).

## Partially Disabled Cuda Component
As shown on the GitHub wiki page [Hardware and Software Support - Cuda Component](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-Cuda-Component) the `cuda` component supports a total of three primary APIs
to expose counters and controls for NVIDIA GPUs.

The Event/Metric APIs (Legacy) only overlaps with the PerfWorks Metrics API at CC 7.0 (V100). Meaning in the case of machines with NVIDIA GPUs with mixed compute capabilities e.g. P100 - CC 6.0 and A100 - CC 8.0 a choice must be made for which CCs the counters and controls will be exposed for.

To allow for this choice to be made the `cuda` component supports being ***Partially Disabled***. Which means:

* If exposing counters and controls for CCs <= 7.0 (e.g. P100 and V100), then support for exposing counters and controls for CCs > 7.0 will be disabled
* If exposing counters and controls for CCs >= 7.0 (e.g. V100 and A100), then support for exposing counters and controls for CCs < 7.0 will be disabled

By default on mixed compute capability machines, counters and controls for CCs >= 7.0 will be exposed. However, at runtime the choice of which CCs the counter and controls will be exposed for can be changed via the environment variable `PAPI_CUDA_API`. Simply
set `PAPI_CUDA_API` equal to `LEGACY`, e.g:

```
export PAPI_CUDA_API=LEGACY
```

Important note, in the case of machines that only have GPUs with CCs = 7.0 there will be no partially disabled Cuda component. Counter and controls will be exposed via the PerfWorks Metrics API; however, if you would like to expose counters and controls via the Legacy APIs please see the aforementioned environment variable.

## Known Limitations
* Exposing counters on machines that have NVIDIA GPUs with CCS >= 7.0 is done via the Pefworks API. This API vastly expands the number of possible counters from roughly a few hundred to over 140,000 per GPU. Due to this, the PAPI utility `utils/papi_native_avail` may take a few minutes to run (as much as 2 minutes per GPU). If the output from `utils/papi_native_avail` is redirected to a file, it may appear as if it has "hung"; however, give it time and it will complete.
***

## FAQ

1. [Unusual installations](#unusual-installations)
2. [CUDA contexts](#cuda-contexts)
3. [CUDA toolkit versions](#cuda-toolkit-versions)
4. [Custom library paths](#custom-library-paths)
5. [Compute capability 7.0 with CUDA toolkit version 11.0](#compute-capability-70-with-cuda-toolkit-version-110)

## Unusual installations
Three libraries are required for the PAPI CUDA component. `libcuda.so`,
`libcudart.so` (The CUDA run-time library), and `libcupti.so`. For CUpti_11,
`libnvperf_host.so` is also necessary. 

For the CUDA component to be operational, it must find the dynamic libraries
mentioned above. If they are not found anywhere in the standard `PAPI_CUDA_ROOT`
subdirectories mentioned above, or `PAPI_CUDA_ROOT` does not exist at runtime, the component looks in the Linux default directories listed by `/etc/ld.so.conf`, 
usually `/usr/lib64`, `/lib64`, `/usr/lib` and `/lib`. 

The system will also search the directories listed in `LD_LIBRARY_PATH`,
separated by colons `:`. This can be set using export; e.g.

```
export LD_LIBRARY_PATH=/WhereLib1CanBeFound:WhereLib2CanBeFound:$LD_LIBRARY_PATH
```

* If CUDA libraries are installed on your system, such that the OS can find `nvcc`, the header files, and the shared libraries, then `PAPI_CUDA_ROOT` and `LD_LIBRARY_PATH` may not be necessary.

## CUDA contexts
The CUDA component can profile using contexts created by `cuCtxCreate` or primary device contexts activated by `cudaSetDevice`. Refer to test codes `HelloWorld`, `simpleMultiGPU`, `pthreads`, etc, that use created contexts. Refer to corresponding `*_noCuCtx` tests for profiling using primary device contexts.

## CUDA toolkit versions
Once your binaries are compiled, it is possible to swap the CUDA toolkit versions without needing to recompile the source. Simply update `PAPI_CUDA_ROOT` to point to the path where the cuda toolkit version can be found. You might need to update `LD_LIBRARY_PATH` as well.

## Custom library paths
PAPI CUDA component loads the CUDA driver library from the system installed path. It loads the other libraries from `$PAPI_CUDA_ROOT`. If that is not set, then it tries to load them from system paths.

However, it is possible to load each of these libraries from custom paths by setting each of the following environment variables to point to the desired files. These are,

- `PAPI_CUDA_RUNTIME` to point to `libcudart.so`
- `PAPI_CUDA_CUPTI` to point to `libcupti.so`
- `PAPI_CUDA_PERFWORKS` to point to `libnvperf_host.so`

## Compute capability 7.0 with CUDA toolkit version 11.0
NVIDIA GPUs with compute capability 7.0 support profiling on both PerfWorks API and the older Event/Metric APIs.

If CUDA toolkit version > 11.0 is used, then PAPI uses the newer API, but using toolkit version 11.0, PAPI uses the events API by default.

If the environment variable `PAPI_CUDA_110_CC_70_PERFWORKS_API` is set to any non-empty value, then compute capability 7.0 using toolkit version 11.0 will use the PerfWorks Metrics API. Eg:

    `export PAPI_CUDA_110_CC_70_PERFWORKS_API=1`
