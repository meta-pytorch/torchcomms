# POWERCAP\_PPC Component

The POWERCAP\_PPC component supports measuring and capping power usage
on recent IBM PowerPC architectures (Power9 and later) using the powercap
interface exposed through the Linux kernel.

* [Enabling the POWERCAP\_PPC Component](#enabling-the-powercap_ppc-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

***
## Enabling the POWERCAP\_PPC Component

To enable reading of POWERCAP\_PPC counters the user needs to link against a
PAPI library that was configured with the POWERCAP\_PPC component enabled. As an
example the following command: `./configure --with-components="powercap_ppc"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Known Limitations
The actions described below will generally require superuser ability.
Note, these actions may have security and performance consequences, so
please make sure you know what you are doing.

Use chmod to set site-appropriate access permissions (e.g. 444) for /sys/firmware/opal/powercap/powercap-(min|max)

Use chmod to set site-appropriate access permissions (e.g. 664) for /sys/firmware/opal/powercap/powercap-current

## FAQ

1. [Measuring and Capping Power](#measuring-and-capping-power)

## Measuring and Capping Power

The powercap sysfs interface exposes power measurments as R/W regsiter-like
power settings. The counters and R/W settings apply to the Power9.

These counters and settings are exposed though this PAPI component and can be
accessed just like any normal PAPI counter. Running the "powercap\_basic" test
in the test directory will list all the events on a system. There is also a
"powercap\_limit" test in the test directory that shows how a power limit is
applied.

Note: Power Limiting using powercap\_ppc **does not** require root privileges.
Write permission to the file /sys/firmware/opal/powercap/powercap-current is
"enough".

