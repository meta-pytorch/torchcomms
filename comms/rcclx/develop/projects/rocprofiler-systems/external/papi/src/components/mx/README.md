# MX Component

The MX component enables PAPI to access counters provided by Myricom MX (Myrinet Express).

* [Enabling the MX Component](#enabling-the-mx-component)

***
## Enabling the MX Component

To enable reading of MX counters the user needs to link against a
PAPI library that was configured with the MX component enabled.  As an
example the following command: `./configure --with-components="mx"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.
