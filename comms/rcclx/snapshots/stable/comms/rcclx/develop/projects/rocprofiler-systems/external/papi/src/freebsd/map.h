/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    freebsd-map.h
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#ifndef _FreeBSD_MAP_H_
#define _FreeBSD_MAP_H_

#include "../papi.h"
#include "../papi_internal.h"
#include "../papi_vector.h"

enum 
{
	CPU_UNKNOWN = 0,
	CPU_P6,
	CPU_P6_C,
	CPU_P6_2,
	CPU_P6_3,
	CPU_P6_M,
	CPU_P4,
	CPU_K7,
	CPU_K8,
	CPU_ATOM,
	CPU_CORE,
	CPU_CORE2,
	CPU_CORE2EXTREME,
	CPU_COREI7,
	CPU_COREWESTMERE,
	CPU_LAST
};

typedef struct Native_Event_LabelDescription 
{
	char *name;
	char *description;
} Native_Event_LabelDescription_t;

typedef struct Native_Event_Info
{
	/* Name and description for all native events */
	Native_Event_LabelDescription_t *info;
} Native_Event_Info_t;

extern Native_Event_Info_t _papi_hwd_native_info[CPU_LAST+1];
extern void init_freebsd_libpmc_mappings (void);
extern int freebsd_number_of_events (int processortype);

#include "map-unknown.h"
#include "map-p6.h"
#include "map-p6-c.h"
#include "map-p6-2.h"
#include "map-p6-3.h"
#include "map-p6-m.h"
#include "map-p4.h"
#include "map-k7.h"
#include "map-k8.h"
#include "map-atom.h"
#include "map-core.h"
#include "map-core2.h"
#include "map-core2-extreme.h"
#include "map-i7.h"
#include "map-westmere.h"

#endif /* _FreeBSD_MAP_H_ */
