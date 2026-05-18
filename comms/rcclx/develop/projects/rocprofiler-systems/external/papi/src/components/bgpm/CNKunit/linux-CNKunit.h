/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-CNKunit.h
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	< your name here >
 *			< your email address >
 * BGPM / CNKunit component 
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the bgpm library.
 */

#ifndef _PAPI_CNKUNIT_H
#define _PAPI_CNKUNIT_H

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "../../../linux-bgq-common.h"


/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/

/* this number assumes that there will never be more events than indicated */
#define CNKUNIT_MAX_COUNTERS PEVT_CNKUNIT_LAST_EVENT
#define OFFSET ( PEVT_NWUNIT_LAST_EVENT + 1 )


/** Structure that stores private information of each event */
typedef struct CNKUNIT_register
{
	unsigned int selector;
	/* Signifies which counter slot is being used */
	/* Indexed from 1 as 0 has a special meaning  */
} CNKUNIT_register_t;


typedef struct CNKUNIT_reg_alloc
{
	CNKUNIT_register_t ra_bits;
} CNKUNIT_reg_alloc_t;


typedef struct CNKUNIT_control_state
{
	int EventGroup;
	long long counts[CNKUNIT_MAX_COUNTERS];
} CNKUNIT_control_state_t;


typedef struct CNKUNIT_context
{
	CNKUNIT_control_state_t state;
} CNKUNIT_context_t;


#endif /* _PAPI_CNKUNIT_H */
