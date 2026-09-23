# ROCM_SMI Component

The ROCM_SMI (System Management Interface) component exposes hardware management
counters and controls for AMD GPUs, such as power consumption, fan speed and
temperature readings; it also allows capping of power consumption.

> **Configure note.** When both `rocm_smi` and the newer `amd_smi` component
> are requested, configure now inspects the installed ROCm release and enables
> only the backend that matches it. `rocm_smi` is selected for ROCm versions
> earlier than 6.4.0; newer stacks default to `amd_smi`.

* [Enabling the ROCM_SMI Component](#enabling-the-rocm_smi-component)
* [Environment Variables](#environment-variables)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

***

## Enabling the ROCM_SMI Component

To enable reading or writing of ROCM_SMI counters the user needs to link
against a PAPI library that was configured with the ROCM_SMI component enabled.
As an example the following command: `./configure --with-components="rocm_smi"`
is sufficient to enable the component.

Typically, the utility `papi_component_avail` (available in `papi/src/utils/papi_component_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For ROCM_SMI, PAPI requires the environment variable `PAPI_ROCMSMI_ROOT` to be set such that the shared object `librocm_smi64.so` and the directory `rocm_smi` are found. This variable is required at both compile and run time.

There are two common cases for setting this variable:

1. **Case 1: For ROCm versions 5.2 and newer:**
    Set `PAPI_ROCMSMI_ROOT` to the top-level ROCM directory, e.g.:

        export PAPI_ROCMSMI_ROOT=/opt/rocm

2. **Case 2: For ROCm versions prior to 5.2:**
    Set `PAPI_ROCMSMI_ROOT` directly to the ROCM_SMI directory, e.g.:

        export PAPI_ROCMSMI_ROOT=/opt/rocm/rocm_smi

In both cases, the directory specified by `PAPI_ROCMSMI_ROOT` **must contain** the following subdirectories:

* `PAPI_ROCMSMI_ROOT/lib` (which should include the dynamic library `librocm_smi64.so`)
* `PAPI_ROCMSMI_ROOT/include/rocm_smi`

## Known Limitations

* Only sets of metrics and events that can be gathered in a single pass are supported.

* Although AMD metrics may be floating point, all values are recast and returned as long long integers.

  The binary image of a `double` is intact; but users must recast to `double` for display purposes.

***

## FAQ

1. [Unusual installations](#unusual-installations)

## Unusual installations

For the ROCM_SMI component to be operational, it must find the dynamic
library `librocm_smi64.so`. This is normally found in the above standard lib directory, or one of the Linux default
directories listed by `/etc/ld.so.conf`, usually `/usr/lib64`, `/lib64`,
`/usr/lib` and `/lib`. If the library is not found (or is not functional)
then the component will be listed as "disabled" with a reason explaining the
problem. If the library was not found, it is not in the expected places.

The system will search the directories listed in `LD_LIBRARY_PATH`. You can add an additional path with a colon, e.g.:

    export LD_LIBRARY_PATH=/WhereALibraryCanBeFound:$LD_LIBRARY_PATH
