# BGPM Component

Five components have been added to PAPI to support hardware performance monitoring for the BG/Q platform; in particular the BG/Q network, the I/O system, the Compute Node Kernel in addition to the processing core. 

* [Enabling the BGPM Component](#enabling-the-bgpm-component)

***
## Enabling the BGPM Component

To enable reading of BGPM counters the user needs to link against a
PAPI library that was configured with the BGPM component enabled. There are no specific component configure scripts for L2unit, IOunit, NWunit, CNKunit. In order to configure PAPI for BG/Q, use the following configure options at the papi/src level:

    ./configure --prefix=< your_choice >  \
      --with-OS=bgq  \
      --with-bgpm_installdir=/bgsys/drivers/ppcfloor  \
      CC=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gcc  \
      F77=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gfortran  \
      --with-components="bgpm/L2unit bgpm/CNKunit bgpm/IOunit bgpm/NWunit"

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.
