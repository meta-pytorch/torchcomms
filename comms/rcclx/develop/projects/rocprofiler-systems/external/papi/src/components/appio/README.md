# APPIO Component

The APPIO component enables PAPI to access application level file and socket I/O information. 

* [Enabling the APPIO Component](#enabling-the-appio-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

***
## Enabling the APPIO Component

To enable reading of APPIO counters the user needs to link against a
PAPI library that was configured with the APPIO component enabled. As an
example the following command: `./configure --with-components="appio"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Known Limitations

The most important aspect to note is that the code is likely to only work on
Linux, given the low-level dependencies on libc features. 

At present the component intercepts the open(), close(), read(), write(),
fread() and fwrite(). In the future it's expected that these will be expanded 
to cover lseek(), select(), other I/O calls.

While READ\_* and WRITE\_* calls will not distinguish between file and network
I/O, the user can explicitly determine network statistics using SOCK_* calls.

Threads are handled using thread-specific structures in the backend. However, no aggregation is currently performed across threads. There is also NO global structure that has the statistics of all the threads. This means the user can call a PAPI read to get statitics for a running thread. However, if the thread has joined, then it's statistics can no longer be queried.

***
## FAQ

1. [Testing](#testing)

## Testing
Tests lie in the tests/ sub-directory. All test take no argument.
