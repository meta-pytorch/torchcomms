# LMSENSORS Component

The LMSENSORS Component enables PAPI to access hardware monitoring sensors through the libsensors library. This component requires lmsensors version >= 3.0.0.

* [Enabling the LMSENSORS Component](#enabling-the-lmsensors-component)
* [Environment Variables](#environment-variables)
* [FAQ](#faq)

***
## Enabling the LMSENSORS Component

To enable reading LMSENSORS events the user needs to link against a PAPI library that was configured with the LMSENSORS component enabled. As an example the following command: `./configure --with-components="lmsensors"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For LMSENSORS, PAPI requires one environment variable: **PAPI\_LMSENSORS\_ROOT**.

This is required for both compiling, and at runtime. 

Example:

    export PAPI_LMSENSORS_ROOT=/usr

Within PAPI\_LMSENSORS\_ROOT, we expect the following standard directories:

    PAPI_LMSENSORS_ROOT/include or PAPI_LMSENSORS_ROOT/include/sensors
    PAPI_LMSENSORS_ROOT/lib64   or PAPI_LMSENSROS_ROOT/lib
***
## FAQ

1. [Unusual installations](#unusual-installations)
2. [List LMSENSORS Supported Events](#list-lmsensors-supported-events)

## Unusual installations

For the LMSENSORS component to be operational, it must find the dynamic library `libsensors.so`.

If it is not found (or is not functional) then the component will be
listed as "disabled" with a reason explaining the problem. If library was not found, then it is not in the expected place.  The component can be configured to look for the library in a specific place, and using an alternate name if desired. Detailed instructions are contained in the `Rules.lmsensors` file.  They are technical, users may wish to enlist the help of a sysadmin.

## List LMSENSORS Supported Events
From papi/src:

    utils/papi_native_avail | grep -i sensors
    
