# SDE Component

The SDE component enables PAPI to read Software Defined Events (SDEs) exported by third party libraries.

* [Enabling the SDE Component](#enabling-the-sde-component)
* [FAQ](#faq)

## Enabling the SDE Component

To enable reading of SDE counters the user needs to link against a
PAPI library that was configured with the SDE component enabled. As an
example the following command: `./configure --with-components="sde"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## FAQ

* [Software Defined Events](https://github.com/icl-utk-edu/papi/wiki/Software_Defined_Events.md)
