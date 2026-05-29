/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq.h
 * CVS:     $Id$
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	< your name here >
 *			< your email address >
 * Blue Gene/Q CPU component: BGPM / Punit
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the BGPM library.
 */


#ifndef _LINUX_BGQ_H
#define _LINUX_BGQ_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/profil.h>
#include <assert.h>
#include <limits.h>
#include <signal.h>
#include <errno.h>
#include <sys/ucontext.h>
#include <stdarg.h>
#include <ctype.h>
//#include <pthread.h>

#include "linux-bgq-common.h"
/* Header required to obtain BGQ personality */
#include "process_impl.h"
#include "linux-context.h"

/* this number assumes that there will never be more events than indicated */
#define BGQ_PUNIT_MAX_COUNTERS UPC_P_NUM_COUNTERS 
#define BGQ_PUNIT_MAX_EVENTS PEVT_PUNIT_LAST_EVENT
#define MAX_COUNTER_TERMS BGQ_PUNIT_MAX_COUNTERS
// keep a large enough gap between actual BGPM events and our local opcode events
#define OPCODE_BUF ( MAX_COUNTERS + MAX_COUNTERS ) 


#include "papi.h"
#include "papi_preset.h"



typedef struct
{
	int preset;				/* Preset code */
	int derived;				/* Derived code */
	char *( findme[MAX_COUNTER_TERMS] );	/* Strings to look for, more than 1 means derived */
	char *operation;			/* PostFix operations between terms */
	char *note;				/* In case a note is included with a preset */
} bgq_preset_search_entry_t;


// Context structure not used...
typedef struct bgq_context
{
	int reserved;
} bgq_context_t;

typedef struct bgq_overflow
{
  	int threshold;
	int EventIndex;
} bgq_overflow_t;

// Control state structure...  Holds local copy of read counters...
typedef struct bgq_control_state
{
	int EventGroup;
	int EventGroup_local[512];
	int count;
	long_long counters[BGQ_PUNIT_MAX_COUNTERS];
	int muxOn;					// multiplexing on or off flag
	int overflow;				// overflow enable
    int overflow_count;
    bgq_overflow_t overflow_list[512];
	int bgpm_eventset_applied;	// BGPM eventGroup applied yes or no flag
} bgq_control_state_t;

// Register allocation structure
typedef struct bgq_reg_alloc
{
	//_papi_hwd_bgq_native_event_id_t id;
} bgq_reg_alloc_t;

// Register structure not used...
typedef struct bgq_register
{
	/* This is used by the framework.It likes it to be !=0 to do something */
	unsigned int selector;
	/* This is the information needed to locate a BGPM / Punit event */
	unsigned eventID;
} bgq_register_t;

/** This structure is used to build the table of events */
typedef struct bgq_native_event_entry
{
	bgq_register_t resources;
	char name[PAPI_MAX_STR_LEN];
	char description[PAPI_2MAX_STR_LEN];
} bgq_native_event_entry_t;

/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef  hwd_reg_alloc_t
#undef  hwd_register_t
#undef  hwd_control_state_t
#undef  hwd_context_t

typedef bgq_reg_alloc_t hwd_reg_alloc_t;
typedef bgq_register_t hwd_register_t;
typedef bgq_control_state_t hwd_control_state_t;
typedef bgq_context_t hwd_context_t;

extern void _papi_hwd_lock( int );
extern void _papi_hwd_unlock( int );

/* Signal handling functions */
//#undef hwd_siginfo_t
//#undef hwd_ucontext_t
//typedef int hwd_siginfo_t;
//typedef ucontext_t hwd_ucontext_t;

#endif
