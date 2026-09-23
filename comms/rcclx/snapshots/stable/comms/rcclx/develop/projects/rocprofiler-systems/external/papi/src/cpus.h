/** @file cpus.h
 * Author:  Gary Mohr
 *          gary.mohr@bull.com
 *          - based on threads.h by unknown author -
 */

#ifndef PAPI_CPUS_H
#define PAPI_CPUS_H

typedef struct _CpuInfo
{
	unsigned int cpu_num;
	struct _CpuInfo *next;
  	hwd_context_t **context;
	EventSetInfo_t **running_eventset;
  	EventSetInfo_t *from_esi;          /* ESI used for last update this control state */
        int num_users;
} CpuInfo_t;

int _papi_hwi_initialize_cpu( CpuInfo_t **dest, unsigned int cpu_num );
int _papi_hwi_shutdown_cpu( CpuInfo_t *cpu );
int _papi_hwi_lookup_or_create_cpu( CpuInfo_t ** here, unsigned int cpu_num );

#endif
