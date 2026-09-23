# POWERCAP Component

The POWERCAP component supports measuring and capping power usage on recent Intel architectures (Sandybridge or later) using the powercap interface exposed through the Linux kernel.

* [Enabling the POWERCAP Component](#enabling-the-powercap-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

***
## Enabling the POWERCAP Component

To enable reading of POWERCAP counters the user needs to link against a
PAPI library that was configured with the POWERCAP component enabled. As an
example the following command: `./configure --with-components="powercap"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Known Limitations
The actions described below will generally require superuser ability.
Note, these actions may have security and performance consequences, so
please make sure you know what you are doing.

Ensure the "CONFIG\_POWERCAP" and "CONFIG\_INTEL\_RAPL" kernel modules are enabled.

Use chmod to set site-appropriate access permissions (e.g. 766) for /sys/class/powercap/*

## FAQ

1. [Measuring and Capping Power](#measuring-and-capping-power)

## Measuring and Capping Power

The powercap sysfs interface exposes energy counters and R/W regsiter-like
power settings. The counters and R/W settings apply to a power domain on a system.

For example, a single KNL chip exposes package power and DRAM power information. 
On KNL this component can be used to read package/DRAM energy counters and set package/DRAM power limits.
There are two limits in the package domain and a single limit in the DRAM domain. The two limits 
in the package domain correspond to long/short term limits. 

For all supported processors, each package/DRAM power limit has an associated
time window. The time window for each limit can also be changed, which changes the enforcement time window of
that limit.

These counters and settings are exposed though this PAPI component and can be accessed just like any normal PAPI
counter. Running the "powercap\_basic" test in the test directory will list all the events on a system. There is also a 
"powercap\_limit" test in the test directory that shows how a power limit is applied.

Note: Power Limiting using powercap requires root or write permission to the files situated in the /sys/class/powercap directory.

