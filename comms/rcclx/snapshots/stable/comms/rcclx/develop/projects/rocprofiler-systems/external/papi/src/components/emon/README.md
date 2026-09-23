# EMON Component

The EMON component provide access to Evniromental MONitoring power data on BG/Q systems. 

* [Enabling the EMON Component](#enabling-the-emon-component)

***
## Enabling the EMON Component

To enable reading of EMON counters the user needs to link against a
PAPI library that was configured with the EMON component enabled. There are no specific component configure scripts. In order to configure PAPI for BG/Q, use the following configure options at the papi/src level:

    ./configure --prefix=< your_choice >  \
      --with-OS=bgq  \
      --with-EMON_installdir=/bgsys/drivers/ppcfloor  \
      CC=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gcc  \
      F77=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gfortran  \
      --with-components="EMON/L2unit EMON/CNKunit EMON/IOunit EMON/NWunit emon"

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.
