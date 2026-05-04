# SENSORS\_PPC Component

The SENSORS\_PPC component supports reading system metrics on recent IBM PowerPC architectures (Power9 and later) using the OCC memory exposed through the Linux kernel.

* [Enabling the SENSORS\_PPC Component](#enabling-the-sensors_ppc-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

***
## Enabling the SENSORS\_PPC Component

To enable reading of SENSORS\_PPC counters the user needs to link against a
PAPI library that was configured with the SENSORS\_PPC component enabled. As an
example the following command: `./configure --with-components="sensors_ppc"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Known Limitations
The actions described below will generally require superuser ability.
Note, these actions may have security and performance consequences, so please
make sure you know what you are doing.

Use chmod to set site-appropriate access permissions (e.g. 440).

Use chown to set group ownership, for /sys/firmware/opal/exports/occ\_inband\_sensors.

And finally, have your user added to said group, granting you read access.

## FAQ

1. [Measuring System](#measuring-system)

## Measuring System

The opal/exports sysfs interface exposes sensors and counters as read only
registers. The sensors and counters apply to the Power9.

These counters and settings are exposed though this PAPI component and can be
accessed just like any normal PAPI counter. Running the "sensors\_ppc\_basic"
test in the tests directory will report a very limited sub-set of information
on a system. For instance, voltage received by socket 0, and its extrema since
the last reboot.

Note: /sys/firmware/opal/exports/occ\_inband\_sensors is RDONLY for root. PAPI
library will need read permissions to access it.

