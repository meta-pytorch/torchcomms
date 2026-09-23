# CORETEMP_FREEBSD Component

The CORETEMP_FREEBSD component is intended to access CPU On-Die Thermal Sensors in the Intel Core architecture in a FreeBSD machine using the coretemp.ko kernel module. The returned values represent Kelvin degrees.

* [Enabling the CORETEMP_FREEBSD Component](#enabling-the-coretemp_freebsd-component)

***
## Enabling the CORETEMP_FREEBSD Component

To enable reading CORETEMP\_FREEBSD events the user needs to link against a PAPI library that was configured with the CORETEMP_FREEBSD component enabled. As an example the following command: `./configure --with-components="coretemp_freebsd"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.
