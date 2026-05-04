/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-IOunit.h
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	< your name here >
 *			< your email address >
 * BGPM / IOunit component 
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the bgpm library.
 */

#ifndef _PAPI_IOUNIT_H
#define _PAPI_IOUNIT_H

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "../../../linux-bgq-common.h"


/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/

/* this number assumes that there will never be more events than indicated */
#define IOUNIT_MAX_COUNTERS UPC_C_IOSRAM_NUM_COUNTERS
#define IOUNIT_MAX_EVENTS PEVT_IOUNIT_LAST_EVENT
#define OFFSET ( PEVT_L2UNIT_LAST_EVENT + 1 )


/** Structure that stores private information of each event */
typedef struct IOUNIT_register
{
	unsigned int selector;
	/* Signifies which counter slot is being used */
	/* Indexed from 1 as 0 has a special meaning  */
} IOUNIT_register_t;


typedef struct IOUNIT_reg_alloc
{
	IOUNIT_register_t ra_bits;
} IOUNIT_reg_alloc_t;

typedef struct IOUNIT_overflow
{
  	int threshold;
	int EventIndex;
} IOUNIT_overflow_t;

typedef struct IOUNIT_control_state
{
	int EventGroup;
	int overflow;				// overflow enable
    int overflow_count;
    IOUNIT_overflow_t overflow_list[512];
	long long counts[IOUNIT_MAX_COUNTERS];
} IOUNIT_control_state_t;


typedef struct IOUNIT_context
{
	IOUNIT_control_state_t state;
} IOUNIT_context_t;


#endif /* _PAPI_IOUNIT_H */
