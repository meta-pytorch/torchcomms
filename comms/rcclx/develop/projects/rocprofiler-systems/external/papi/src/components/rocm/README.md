# ROCM Component

The ROCM component exposes numerous performance events on AMD GPUs.
The component is an adapter to the ROCm profiling library (ROC-profiler) which is included in a standard ROCM release.


* [Enabling the ROCM Component](#enabling-the-rocm-component)
* [Environment Variables](#environment-variables)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***
## Enabling the ROCM Component

To enable reading ROCM events the user needs to link against a PAPI library that was configured with the ROCM component enabled. As an example the following command: `./configure --with-components="rocm"` is sufficient to enable the component.

Typically, the utility `papi_component_avail` (available in `papi/src/utils/papi_component_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For ROCM, PAPI requires the location of the ROCM install directory. This can be
specified by one environment variable: **PAPI\_ROCM\_ROOT**.

Access to the rocm main directory is required at both compile (for include
files) and at runtime (for libraries).

Example:

    export PAPI_ROCM_ROOT=/opt/rocm

Within PAPI\_ROCM\_ROOT, we expect the following standard directories:

    PAPI_ROCM_ROOT/include
    PAPI_ROCM_ROOT/include/hsa
    PAPI_ROCM_ROOT/lib
    PAPI_ROCM_ROOT/rocprofiler/lib
    PAPI_ROCM_ROOT/rocprofiler/include

Besides the PAPI\_ROCM\_ROOT environment variable, the PAPI user may optionally set
two more environment variables that are otherwise set internally by the component
based on the PAPI\_ROCM\_ROOT variable. These are not needed by PAPI, but by the AMD
ROCPROFILER software we interface with.

These added environment variables are typically set as follows, after
PAPI\_ROCM\_ROOT has been exported. An example is provided below.

For ROCM versions < 5.2.0:

    export ROCP_METRICS=$PAPI_ROCM_ROOT/rocprofiler/lib/metrics.xml
    export HSA_TOOLS_LIB=$PAPI_ROCM_ROOT/rocprofiler/lib/librocprofiler64.so

For ROCM versions >= 5.2.0:

    export ROCP_METRICS=${PAPI_ROCM_ROOT}/lib/rocprofiler/metrics.xml
    export HSA_TOOLS_LIB=${PAPI_ROCM_ROOT}/lib/librocprofiler64.so

The first of these, ROCP\_METRICS, must point at a file containing the
descriptions of metrics. The second is the location of the rocprofiler library
needed by HSA runtime.

If the user relies on PAPI to export these, then it is important to execute the
function PAPI\_library\_init() in the user code BEFORE any HIP functions are
executed. Because these values are read once by AMD with the first HIP function
call, and if HIP sets up without them, PAPI may not read counters correctly.

If the user code cannot be changed, then export the variables explicitly before
execution.

One important exception is represented by the PAPI high-level API. In this case
there is no explicit call to PAPI\_library\_init() in the user's code. Instead
the PAPI library is initialized by the first high-level API call. In this case
it is still possible to make PAPI export the above environment variables by
setting the ROCP\_TOOL\_LIB to the PAPI library as follows:

    export ROCP_TOOL_LIB=<path_to_papi_lib>/libpapi.so

## Known Limitations

* PAPI may read zeros for many events if rocprofiler environment variables are
  not exported and HIP functions are executed by the user before the user
  executes PAPI\_library\_init().

* Only sets of metrics and events that can be gathered in a single pass are supported.

* Although AMD metrics may be floating point, all values are recast and returned as long long integers.

    The binary image of a `double` is intact; but users must recast to `double` for display purposes.

* Some of the ROCm events are known to cause an error when the rocm component is used in sampling mode

    For example `TA_BUSY_avr`

    ```console
    $ papi_command_line TA_BUSY_avr

    This utility lets you add events from the command line interface to see if they work.

    Successfully added: rocm:::TA_BUSY_avr:device=0

    Memory access fault by GPU node-4 (Agent handle: 0x46d6d10) on address 0x7ffed888c000. Reason: Unknown.
    Aborted
    ```

    The error appears to happen when the ROCr runtime shuts down

***
## FAQ

1. [Unusual installations](#unusual-installations)

## Unusual installations
For the ROCM component to be operational, it must find the dynamic libraries `libhsa-runtime64.so` and `librocprofiler64.so`. These are normally found in the above standard directories. If these libraries are not found (or are not functional) then the component will be listed as "disabled" with a reason explaining the problem. If libraries were not found, then they are not in the expected places.

2. [Device isolation](#device-isolation)

## Device isolation
Compute clusters resource managers can isolate GPU devices, on compute nodes,
into subgroups. This means that a job might only see part of the devices on the
node. How many devices are visible depends on the value of a set of environment
variables, configured by the resource manager (e.g. HIP\_VISIBLE\_DEVICES,
ROCR\_VISIBLE\_DEVICES, etc).

In order to detect available devices, the ROCm component relies on the HSA ROCm
runtime functions (i.e. hsa\_iterate\_agents). The ROCR\_VISIBLE\_DEVICES
environment variable establishes how many devices will be visible to the ROCm
runtime. Therefore, by extension, the PAPI ROCm component will only see as many
devices as allocated by the resource manager through the aforementioned
environment variable. The component assigns them integer identifiers in the
range [0, N-1], where N is the number of devices for the partition.

Therefore, when using the component in a HIP context, the application would
need to map the device index given by hipGetDevice to this index range and use
the index in the event name, e.g., rocm:::GPUBusy:device=X. Preferably the UUID
of the device should be used for this mapping (see hipDeviceGetUuid and
HSA\_AMD\_AGENT\_INFO\_UUID).

The AMD isolation mechanism is described in more details here:
https://rocmdocs.amd.com/en/latest/conceptual/gpu-isolation.html
