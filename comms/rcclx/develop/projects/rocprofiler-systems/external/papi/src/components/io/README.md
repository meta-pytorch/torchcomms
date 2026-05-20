# IO Component

The IO component enables PAPI to access the io statistics exported by the Linux kernel through the /proc pseudo-file system (file /proc/self/io).

* [Enabling the IO Component](#enabling-the-io-component)
* [FAQ](#faq)

***
## Enabling the IO Component

To enable reading of IO counters the user needs to link against a
PAPI library that was configured with the IO component enabled.  As an
example the following command: `./configure --with-components="io"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

***
## FAQ

This component will dynamically create a native events table.

    Event names
    -------------------------
      "<ifname>.rchar",
      "<ifname>.wchar",
      "<ifname>.syscr",
      "<ifname>.syscw",
      "<ifname>.read_bytes",
      "<ifname>.write_bytes",
      "<ifname>.cancelled_write_bytes"

By default the Linux kernel only updates the io statistics once every second (see the references listed in the "man proc" section for some problems you may come across and for how to change the default polling period).


