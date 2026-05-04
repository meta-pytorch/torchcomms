# INFINIBAND Component

The INFINIBAND component uses the sysfs interface to access infiniband
performance counters from user space.

* [Enabling the INFINIBAND Component](#enabling-the-infiniband-component)
* [FAQ](#faq)

***
## Enabling the INFINIBAND Component

To enable reading of INFINIBAND counters the user needs to link against a
PAPI library that was configured with the INFINIBAND component enabled.  As an
example the following command: `./configure --with-components="infiniband"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

***
## FAQ

On initialization, the INFINIBAND component checks for the existence of folder /sys/class/infiniband/ and it auto-detects all the active IB devices and associated ports.

This component supports both the short IB counters, which are at most
32-bit, overflowing and auto resetting, thus, not very useful, and the IBoE
extended counters, which are 64-bit and free running. If available, the
latter counters are recommended for performance monitoring.

