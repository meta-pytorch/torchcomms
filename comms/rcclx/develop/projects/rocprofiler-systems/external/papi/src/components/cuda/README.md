# CUDA Component

The CUDA component exposes counters and controls for NVIDIA GPUs.

* [Enabling the CUDA Component](#enabling-the-cuda-component)
* [Environment Variables](#environment-variables)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***
## Enabling the CUDA Component

To enable reading or writing of CUDA counters the user needs to link against a
PAPI library that was configured with the CUDA component enabled. As an
example the following command:

    ./configure --with-components="cuda"
    
is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For CUDA, PAPI requires one environment variable: `PAPI_CUDA_ROOT`. This is
required for both compiling and runtime. 

Typically in Linux one would export this (examples are shown below) variable but
some systems have software to manage environment variables (such as `module` or
`spack`), so consult with your sysadmin if you have such management software. Eg:

    export PAPI_CUDA_ROOT=/path/to/installed/cuda

Within PAPI_CUDA_ROOT, we expect the following standard directories for building:

    PAPI_CUDA_ROOT/include
    PAPI_CUDA_ROOT/extras/CUPTI/include

and for runtime:

    PAPI_CUDA_ROOT/lib64
    PAPI_CUDA_ROOT/extras/CUPTI/lib64

As of this writing (07/2021) Nvidia has overhauled performance reporting;
divided now into "Legacy CUpti" and "CUpti_11", the new approach. Legacy
Cupti works on devices up to Compute Capability 7.0; while only CUpti_11
works on devices with Compute Capability >=7.0. Both work on CC==7.0.

This component automatically distinguishes between the two; but it cannot
handle a "mix", one device that can only work with Legacy and another that
can only work with CUpti_11.

For the CUDA component to be operational, both versions require
the following dynamic libraries be found at runtime:

    libcuda.so
    libcudart.so
    libcupti.so

CUpti\_11 also requires:

    libnvperf_host.so

If those libraries cannot be found or some of those are stub libraries in the
standard `PAPI_CUDA_ROOT` subdirectories, you must add the correct paths,
e.g. `/usr/lib64` or `/usr/lib` to `LD_LIBRARY_PATH`, separated by colons `:`.
This can be set using export; e.g. 

    export LD_LIBRARY_PATH=$PAPI_CUDA_ROOT/lib64:$LD_LIBRARY_PATH

## Known Limitations
* In CUpti\_11, the number of possible events is vastly expanded; e.g. from
  some hundreds of events per device to over 110,000 events per device. this can
  make the utility `papi/src/utils/papi_native_avail` run for several minutes;
  as much as 2 minutes per GPU. If the output is redirected to a file, this 
  may appear to "hang up". Give it time.

* Currently the CUDA component profiling only works with GPUs with compute capability > 7.0 using the NVIDIA Perfworks libraries.

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

    export LD_LIBRARY_PATH=/WhereLib1CanBeFound:/WhereLib2CanBeFound:$LD_LIBRARY_PATH

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
NVIDIA GPUs with compute capability 7.0 support profiling on both PerfWorks API and the older Events & Metrics API.

If CUDA toolkit version > 11.0 is used, then PAPI uses the newer API, but using toolkit version 11.0, PAPI uses the events API by default.

If the environment variable `PAPI_CUDA_110_CC_70_PERFWORKS_API` is set to any non-empty value, then compute capability 7.0 using toolkit version 11.0 will use the Perfworks API. Eg:

    `export PAPI_CUDA_110_CC_70_PERFWORKS_API=1`
