/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-NWunit.h
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	< your name here >
 *			< your email address >
 * BGPM / NWunit component 
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the bgpm library.
 */

#ifndef _PAPI_NWUNIT_H
#define _PAPI_NWUNIT_H

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "../../../linux-bgq-common.h"


/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/

/* this number assumes that there will never be more events than indicated */
//#define NWUNIT_MAX_COUNTERS UPC_NW_ALL_LINKCTRS
#define NWUNIT_MAX_COUNTERS UPC_NW_NUM_CTRS
#define NWUNIT_MAX_EVENTS PEVT_NWUNIT_LAST_EVENT
#define OFFSET ( PEVT_IOUNIT_LAST_EVENT + 1 )


/** Structure that stores private information of each event */
typedef struct NWUNIT_register
{
	unsigned int selector;
	/* Signifies which counter slot is being used */
	/* Indexed from 1 as 0 has a special meaning  */
} NWUNIT_register_t;


typedef struct NWUNIT_reg_alloc
{
	NWUNIT_register_t ra_bits;
} NWUNIT_reg_alloc_t;


typedef struct NWUNIT_control_state
{
	int EventGroup;
	long long counts[NWUNIT_MAX_COUNTERS];
} NWUNIT_control_state_t;


typedef struct NWUNIT_context
{
	NWUNIT_control_state_t state;
} NWUNIT_context_t;


#endif /* _PAPI_NWUNIT_H */
