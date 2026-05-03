/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    freebsd-map.c
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include "freebsd.h"
#include "papiStdEventDefs.h"
#include "map.h"

/** See other freebsd-map*.* for more details! **/

Native_Event_Info_t _papi_hwd_native_info[CPU_LAST+1];

void init_freebsd_libpmc_mappings (void)
{
	_papi_hwd_native_info[CPU_UNKNOWN].info = UnkProcessor_info;
	_papi_hwd_native_info[CPU_P6].info = P6Processor_info;
	_papi_hwd_native_info[CPU_P6_C].info = P6_C_Processor_info;
	_papi_hwd_native_info[CPU_P6_2].info = P6_2_Processor_info;
	_papi_hwd_native_info[CPU_P6_3].info = P6_3_Processor_info;
	_papi_hwd_native_info[CPU_P6_M].info = P6_M_Processor_info;
	_papi_hwd_native_info[CPU_P4].info = P4Processor_info;
	_papi_hwd_native_info[CPU_K7].info = K7Processor_info;
	_papi_hwd_native_info[CPU_K8].info = K8Processor_info;
	_papi_hwd_native_info[CPU_ATOM].info = AtomProcessor_info;
	_papi_hwd_native_info[CPU_CORE].info = CoreProcessor_info;
	_papi_hwd_native_info[CPU_CORE2].info = Core2Processor_info;
	_papi_hwd_native_info[CPU_CORE2EXTREME].info = Core2ExtremeProcessor_info;
	_papi_hwd_native_info[CPU_COREI7].info = i7Processor_info;
	_papi_hwd_native_info[CPU_COREWESTMERE].info = WestmereProcessor_info;

	_papi_hwd_native_info[CPU_LAST].info = NULL;
}

int freebsd_number_of_events (int processortype)
{
	int counter = 0;

	while (_papi_hwd_native_info[processortype].info[counter].name != NULL)
		counter++;

	return counter;
}
