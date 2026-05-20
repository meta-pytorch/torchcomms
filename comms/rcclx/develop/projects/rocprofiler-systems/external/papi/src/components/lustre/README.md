# LUSTRE Component

The LUSTRE component enables PAPI to access IO statistics provided by the Lustre filesystem.

* [Enabling the LUSTRE Component](#enabling-the-lustre-component)

***
## Enabling the LUSTRE Component

To enable reading of LUSTRE counters the user needs to link against a
PAPI library that was configured with the LUSTRE component enabled.  As an
example the following command: `./configure --with-components="lustre"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.
