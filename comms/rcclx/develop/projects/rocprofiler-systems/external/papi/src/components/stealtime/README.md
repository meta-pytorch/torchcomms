# STEALTIME Component

The STEALTIME component enables PAPI to access stealtime filesystem statistics.

* [Enabling the STEALTIME Component](#enabling-the-stealtime-component)

***
## Enabling the STEALTIME Component

To enable reading of STEALTIME counters the user needs to link against a
PAPI library that was configured with the STEALTIME component enabled. As an
example the following command: `./configure --with-components="stealtime"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.
