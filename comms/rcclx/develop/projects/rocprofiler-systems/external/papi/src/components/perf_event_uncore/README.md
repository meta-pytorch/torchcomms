# PERF\_EVENT\_UNCORE Component

The PERF\_EVENT_UNCORE component enables PAPI to access perf\_event CPU uncore and northbridge.

* [Enabling the PERF\_EVENT\_UNCORE Component](#enabling-the-perf_event_uncore-component)
* [FAQ](#faq)

***
## Enabling the PERF\_EVENT\_UNCORE Component

This component is enabled by default.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## FAQ

1. [Measuring Uncore Events](#measuring-uncore-events)

## Measuring Uncore Events

Uncore events (like e.g. `hswep_unc_ha0`) are per-package (not per-process like core events). 
Therefore, you need to make sure you are specifying the CPU package you want to monitor. 
You can make this specification with the **`:cpu=`** qualifier.

If you have more than one package in your machine, then you can measure with multiple events.

For example on Haswell, the following measures the uncore event `hswep_unc_ha0::UNC_H_RING_AD_USED:CW`
on one socket:   
     
        papi_command_line hswep_unc_ha0::UNC_H_RING_AD_USED:CW:cpu=0   
	
**Hint**: 

* You can use `lscpu` on the respective node to see the distribution of 
  CPU-core identifiers across the sockets. 
* It shows you which of the logical cores belong to the same package.
* Based on that output, you can pick the appropriate **`:cpu=`** to get 
  the uncore event count for the second (third, etc) socket. 


**Example for a dual-socket Intel Haswell node with 24 physical cores**:   
   
        lscpu
	    Architecture:          x86_64
	    CPU op-mode(s):        32-bit, 64-bit
	    Byte Order:            Little Endian
	    CPU(s):                24
	    ...
	    NUMA node0 CPU(s):     0-11
        NUMA node1 CPU(s):     12-23


The following measures the same uncore event as above for the second socket (NUMA node1):   
   
        papi_command_line hswep_unc_ha0::UNC_H_RING_AD_USED:CW:cpu=12

