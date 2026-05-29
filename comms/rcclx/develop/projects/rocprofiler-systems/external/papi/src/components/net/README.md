# NET Component

The NET component enables PAPI to access the network statistics exported by the Linux kernel through the /proc pseudo-file system (file /proc/net/dev).

* [Enabling the NET Component](#enabling-the-net-component)
* [FAQ](#faq)

***
## Enabling the NET Component

To enable reading of NET counters the user needs to link against a
PAPI library that was configured with the NET component enabled.  As an
example the following command: `./configure --with-components="net"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

***
## FAQ

This component will dynamically create a native events table based on the number of interfaces listed in /proc/net/dev (16 entries for each network interface).

    Event names
    -------------------------
      "<ifname>.rx.bytes",
      "<ifname>.rx.packets",
      "<ifname>.rx.errors",
      "<ifname>.rx.dropped",
      "<ifname>.rx.fifo",
      "<ifname>.rx.frame",
      "<ifname>.rx.compressed",
      "<ifname>.rx.multicast",
      "<ifname>.tx.bytes",
      "<ifname>.tx.packets",
      "<ifname>.tx.errors",
      "<ifname>.tx.dropped",
      "<ifname>.tx.fifo",
      "<ifname>.tx.colls",
      "<ifname>.tx.carrier",
      "<ifname>.tx.compressed"

By default the Linux kernel only updates the network statistics once every second (see the references listed in the "SEE ALSO" section for some problems you may come across and for how to change the default polling period).

Note: The Linux network statistics are updated by code that resides in the file net/core/dev.c.


## SEE ALSO

* Network Stats Anomaly
  http://collectl.sourceforge.net/NetworkStats.html

*  /proc/net/dev which lists the various network devices configured on the system, complete with transmit and receive statistics.

