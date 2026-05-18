# HOST\_MICPOWER Component

The HOST\_MICPOWER component exports power information for Intel Xeon Phi cards (MIC).
The component makes use of the MicAccessAPI distributed with the Intel Manycore Platform Software Stack.
(http://software.intel.com/en-us/articles/intel-manycore-platform-software-stack-mpss)
Specifically in the intel-mic-sysmgmt package.

* [Enabling the HOST\_MICPOWER Component](#enabling-the-host_micpower-component)
* [FAQ](#faq)

***
## Enabling the HOST\_MICPOWER Component

A configure script allows for non-default locations for the sysmgmt sdk.
See:

    cd src/components/host_micpower
    ./configure --help

To enable reading of HOST\_MICPOWER counters the user needs to link against a
PAPI library that was configured with the HOST\_MICPOWER component enabled.  As an
example the following command: `./configure --with-components="host_micpower"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## FAQ

PAPI retrieves the data via the MicGetPowerUsage call.

Per the SDK documentation:
MicGetPowerUsage - Retrieve power usage values of Intel® Xeon Phi™ Coprocessor and components.

    Data Fields
    MicPwrPws  total0
               Total power utilization by Intel® Xeon Phi™ product codenamed “Knights Corner” device, Averaged over Time Window 0 (uWatts).
    MicPwrPws  total1
               Total power utilization by Intel® Xeon Phi™ product codenamed “Knights Corner” device, Averaged over Time Window 1 (uWatts).
    MicPwrPws  inst
               Instantaneous power (uWatts).
    MicPwrPws  imax
               Max instantaneous power (uWatts).
    MicPwrPws  pcie
               PCI-E connector power (uWatts).
    MicPwrPws  c2x3
               2x3 connector power (uWatts).
    MicPwrPws  c2x4
               2x4 connector power (uWatts).
    MicPwrVrr  vccp
               Core rail (uVolts).
    MicPwrVrr  vddg
               Uncore rail (uVolts).
    MicPwrVrr  vddq
               Memory subsystem rail (uVolts).
