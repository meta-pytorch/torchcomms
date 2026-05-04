# CORETEMP Component

The CORETEMP component enables PAPI-C to access hardware monitoring sensors through the coretemp sysfs interface. This component will dynamically create a native events table for all the sensors that can be found under /sys/class/hwmon/hwmon[0-9]+.

* [Enabling the CORETEMP Component](#enabling-the-coretemp-component)

***
## Enabling the CORETEMP Component

To enable reading CORETEMP events the user needs to link against a PAPI library that was configured with the CORETEMP component enabled. As an example the following command: `./configure --with-components="coretemp"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.
