# PCP Component

The PCP component interfaces PAPI and the Performance Co-Pilot, for the
Performance Metrics Domain Agent (PMDA) perfevent. This allows monitoring
of IBM Power 9 'nest' events via PCP without elevated privileges. This 
component was developed and tested using PCP version 3.12.2. 

* [Enabling the PCP Component](#enabling-the-pcp-component)
* [Environment Variables](#environment-variables)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***
## Enabling the PCP Component

To enable reading PCP events the user needs to link against a PAPI library that
was configured with the PCP component enabled. As an example the following
command: `./configure --with-components="pcp"` is sufficient to enable the
component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For PCP, PAPI requires one environment variable: `PAPI_PCP_ROOT`.

Typically in Linux one would export these (examples are show below) but some
systems have software to manage environment variables (such as `module` or
`spack`), so consult with your sysadmin if you have such management software.

Example:

    export PAPI_PCP_ROOT=/usr

Within PAPI_PCP_ROOT, we expect the following standard directories:

    PAPI_PCP_ROOT/include #OR# PAPI_PCP_ROOT/include/pcp
    PAPI_PCP_ROOT/lib64

## Known Limitations

* PCP interfaces with a daemon (a background program running on the machines). If
you use a batch system (like SLURM) so that your programs run on a different
machine (node) than your login node, then it is possible one machine can have
the daemon installed and the other machine does not.

    For example, on the Summit supercomputer login nodes do not execute the PCP
daemon, only work nodes do. Thus PCP code can be compiled but not tested on the
login node, it can only be tested by submitting a batch job. In order for the
PCP deamon to work resources have to be allocated using the `-alloc_flags "PMCD"`
flag. This can be done either with an interactive job:

  $ bsub -W 00:10 -nnodes 1 -P <PROJECT> -alloc_flags "PMCD" -Is /bin/bash
  $ jsrun -n1 -a1 -r1 <program>

or with a job script:

  ...
  #BSUB -alloc_flags "PMCD".
  jsrun -n1 -a1 -r1 <program>

* The P9 nest events contain both counters and "dutycycle" events.
The dutycycle events return instantaneous double precision floating point
values (between 0.0 and 1.0) that cannot be 'reset'. A PAPI_get_event_info() on
these events will return, in PAPI_event_info_t, a 'timescope' of
PAPI_TIMESCOPE_POINT. Counters will have a PAPI_TIMESCOPE_SINCE_START.  

* PAPI may not report any "help text" for the counters (for example, using the 
`papi/src/utils/papi_native_avail` utility). PAPI relies on the the PCP daemon
to provide those descriptions, and it is possible for a sysadmin to install the
daemon without the file containing the descriptions.

* The component translates all 32 bit counters (int, unsigned int,
float) into corresponding 64 bit counters (long long int, unsigned long long,
double) for return to PAPI. e.g. a float = -1.23 will become a double = -1.23;
an unsigned int of 0xFFFFFFFF will become an unsigned long long of
0x00000000FFFFFFFF; a signed int of 0xFFFFFFFF (-1) becomes a long long of
0xFFFFFFFFFFFFFFFF (-1).

* PCP STRING EVENTS AND OTHER EVENTS: PCP has events that return arbitrarily
long strings. There is no char* data type in PAPI, so we exclude such events
from the list of events we support. The only one we excluded on P9 was a string
containing the version number of the PCP code in use.

    We also exclude PCP aggregate types and some other types that cannot be
reasonably represented by PAPI's numerical types.

* PCP ARRAYS OF VALUES: A single PCP event can return an array of values; for
example on the Saturn test system nearly all events return an array of 64
values: 1 per Core. Others return 4 values; one per socket. We call these
multi-valued events.

    However, PAPI is designed to return exactly one value per event. Our
approach here is to 'explode' the multi-valued events, into one PAPI event per
value, and make the event names unique by appending the PCP index name to the
PCP event name with a ':'. For example:

    PCP_EVENT_NAME -> PCP_EVENT_NAME:cpu0, PCP_EVENT_NAME:cpu1, ...
PCP_EVENT_NAME:cpu63.

***
## FAQ

1. [Unusual installations](#unusual-installations)

## Unusual installations
For the PCP component to be operational, it must find the dynamic library
`libpcp.so`. This is normally found in the above standard lib directory, or one
of the Linux default directories listed by `/etc/ld.so.conf`, usually
`/usr/lib64`, `/lib64`, `/usr/lib` and `/lib`. If the library is not found (or
is not functional) then the component will be listed as "disabled" with a
reason explaining the problem. If library was not found, it is not in the
expected places. 

The system will search the directories listed in **LD\_LIBRARY\_PATH**. You can
add an additional path with a colon e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereALibraryCanBeFound
