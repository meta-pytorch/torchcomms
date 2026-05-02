# EXAMPLE Component

The EXAMPLE component demos the component interface and implements three example counters.

* [Enabling the EXAMPLE Component](#enabling-the-example-component)

***
## Enabling the EXAMPLE Component

To enable reading of EXAMPLE counters the user needs to link against a
PAPI library that was configured with the EXAMPLE component enabled.  As an
example the following command: `./configure --with-components="example"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

