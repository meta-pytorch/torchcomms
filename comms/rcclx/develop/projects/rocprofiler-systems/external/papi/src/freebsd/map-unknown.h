/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-unknown.h
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#ifndef FreeBSD_MAP_UNKNOWN
#define FreeBSD_MAP_UNKNOWN

enum NativeEvent_Value_UnknownProcessor {
	PNE_UNK_BRANCHES = PAPI_NATIVE_MASK,
	PNE_UNK_BRANCH_MISPREDICTS,
	/* PNE_UNK_CYCLES, -- libpmc only supports cycles in system wide mode and this 
	requires root privileges */
	PNE_UNK_DC_MISSES,
	PNE_UNK_IC_MISSES,
	PNE_UNK_INSTRUCTIONS,
	PNE_UNK_INTERRUPTS,
	PNE_UNK_UNHALTED_CYCLES,
	PNE_UNK_NATNAME_GUARD
};

extern Native_Event_LabelDescription_t UnkProcessor_info[];
extern hwi_search_t UnkProcessor_map[];

#endif
