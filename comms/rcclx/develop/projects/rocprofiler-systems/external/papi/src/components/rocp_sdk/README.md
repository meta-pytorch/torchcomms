# ROCP\_SDK Component

The ROCP\_SDK component exposes numerous performance events on AMD GPUs and APUs.
The component is an adapter to the ROCm profiling library ROCprofiler-SDK which is included in a standard ROCM release.

* [Enabling the ROCP\_SDK Component](#enabling-the-rocp_sdk-component)
* [Environment Variables](#environment-variables)
* [Hardware and Software Support](#hardware-and-software-support)
* [Known Limitations](#known-limitations)
***
## Enabling the ROCP\_SDK Component
    
To enable reading ROCP\_SDK events the user needs to link against a PAPI library that was configured with the ROCP\_SDK component enabled. As an example the following command: `./configure --with-components="rocp_sdk"` is sufficient to enable the component.

Typically, the utility `papi_component_avail` (available in `papi/src/utils/papi_component_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Library Version Limitations
AMD ROCprofiler-SDK released before rocm-6.3.2 has known bugs.

## Environment Variables

PAPI requires the location of the ROCM install directory. This can be
specified by one environment variable: **PAPI\_ROCP\_SDK\_ROOT**.
    
Access to the rocm main directory is required at both compile (for include
files) and at runtime (for libraries).
            
Example:
 
    export PAPI_ROCP_SDK_ROOT=/opt/rocm

Within PAPI\_ROCP\_SDK\_ROOT, we expect the following standard directories:

    PAPI_ROCP_SDK_ROOT/include
    PAPI_ROCP_SDK_ROOT/include/rocprofiler-sdk
    PAPI_ROCP_SDK_ROOT/lib

### Counter Collection Modes

The default mode is device sampling, which allows counter collection during the execution of a kernel. If a PAPI user wants to use dispatch mode, they must set the environment variable: **PAPI\_ROCP\_SDK\_DISPATCH\_MODE** before initializing PAPI.

Example:

    export PAPI_ROCP_SDK_DISPATCH_MODE=1

### Unusual Installations

For the ROCP\_SDK component to be operational, it must find the dynamic library `librocprofiler-sdk.so` at runtime. This is normally found in the standard directory structure mentioned above. For unusual installations that do not follow this structure, the user may provide the full path to the library using the environment variable: **PAPI\_ROCP\_SDK\_LIB**.

Example:

    export PAPI_ROCP_SDK_LIB=/opt/rocm-6.3.2/lib/librocprofiler-sdk.so.0

Note that this variable takes precedence over PAPI\_ROCP\_SDK\_ROOT.

Additionally, the component supports installations where ROCprofiler-SDK does not reside inside the ROCm installation. For example, if installed from source (https://github.com/ROCm/rocprofiler-sdk). In such a case, the PAPI user must set PAPI_ROCP_SDK_ROOT to point to the ROCprofiler-SDK installation and PAPI_ROCM_ROOT to point to the ROCm installation.

Example:
    export PAPI_ROCP_SDK_ROOT=${HOME}/my_packages
    export PAPI_ROCM_ROOT=/opt/rocm

## Hardware and Software Support

To see the ROCP\_SDK component's current supported hardware and software please visit the GitHub wiki page [Hardware and Software Support - ROCP\_SDK](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-ROCP_SDK-Component).

## Known Limitations

* In dispatch mode, PAPI may read zeros if reading takes place immediately after the return of a GPU kernel. This is not a PAPI bug. It may occur because calls such as hipDeviceSynchronize() do not guarantee that ROCprofiler has been called and all counter buffers have been flushed.  Therefore, it is recommended that the user code adds a delay between the return of a kernel and calls to PAPI_read(), PAPI_stop(), etc.
* If an application is linked against the static PAPI library libpapi.a, then the application must call PAPI_library_init() before calling any hip routines (e.g. hipInit(), hipGetDeviceCount(), hipLaunchKernelGGL(), etc). If the application is linked against the dynamic library libpapi.so, then the order of operations does not matter.
