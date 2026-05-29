# SYSDETECT Component

The SYSDETECT component allows PAPI users to query comprehensive system
information. The information is gathered at PAPI_library_init() time and
presented to the user through appropriate APIs. The component works
similarly to other components, which means that hardware information for
a specific device might not be available at runtime if, e.g., the device
runtime software is not installed.

* [Enabling the SYSDETECT Component](#enabling-the-sysdetect-component)

## Enabling the SYSDETECT Component

The sysdetect component is enabled by default, however, support for various
devices accessed by the component has to be enabled by exporting appropriate
environment variables. This is required so that the component can dlopen the
needed libraries and access hardware information. For example, for the
sysdetect component to access cuda devices information the PAPI_CUDA_ROOT
environment variable has to be set to the cuda toolkit installation path
(in the same way the user has to enable the cuda component).

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

The utility program papi_hardware_avail uses the SYSDETECT component to report
installed and configured hardware information to the command line.
