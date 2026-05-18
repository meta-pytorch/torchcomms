# ROCM_SMI Component

The ROCM_SMI (System Management Interface) component exposes hardware management
counters and controls for AMD GPUs, such as power consumption, fan speed and
temperature readings; it also allows capping of power consumption.

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

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For ROCM_SMI, PAPI requires one environment variable: `PAPI_ROCMSMI_ROOT`. Note
in most installations, this is a subdirectory under the ROCM directory. This is 
required at both compile and run time.

Example:

    export PAPI_ROCMSMI_ROOT=/opt/rocm/rocm_smi

Within PAPI_ROCMSMI_ROOT, we expect the following standard directories:

    PAPI_ROCMSMI_ROOT/lib
    PAPI_ROCMSMI_ROOT/include/rocm_smi

## Known Limitations

* Only sets of metrics and events that can be gathered in a single pass are supported.

* Although AMD metrics may be floating point, all values are recast and returned as long long integers.

    The binary image of a `double` is intact; but users must recast to `double` for display purposes.

***
## FAQ

1. [Unusual installations](#unusual-installations)

## Unusual installations
For the ROCM_SMI component to be operational, it must find the dynamic
library `librocm_smi64.so`. This is normally
found in the above standard lib directory, or one of the Linux default
directories listed by `/etc/ld.so.conf`, usually `/usr/lib64`, `/lib64`,
`/usr/lib` and `/lib`. If the library is not found (or is not functional)
then the component will be listed as "disabled" with a reason explaining the
problem. If library was not found, it is not in the expected places. 

The system will search the directories listed in **LD\_LIBRARY\_PATH**. You can add an additional path with a colon e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereALibraryCanBeFound
