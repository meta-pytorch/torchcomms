/*******************************************************************************
 * >>>>>> "Development of a PAPI Backend for the Sun Niagara 2 Processor" <<<<<<
 * -----------------------------------------------------------------------------
 *
 * Fabian Gorsler <fabian.gorsler@smail.inf.h-bonn-rhein-sieg.de>
 *
 *       Hochschule Bonn-Rhein-Sieg, Sankt Augustin, Germany
 *       University of Applied Sciences
 *
 * -----------------------------------------------------------------------------
 *
 * File:   solaris-niagara2.c
 * Author: fg215045
 *
 * Description: Data structures used for the communication between PAPI and the 
 * component. Additionally some macros are defined here. See solaris-niagara2.c.
 *
 *      ***** Feel free to convert this header to the PAPI default *****
 *
 * -----------------------------------------------------------------------------
 * Created on April 23, 2009, 7:31 PM
 ******************************************************************************/

#ifndef _SOLARIS_NIAGARA2_H
#define _SOLARIS_NIAGARA2_H

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <limits.h>
#include <synch.h>
#include <procfs.h>
#include <libcpc.h>
#include <libgen.h>
#include <ctype.h>
#include <errno.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>
#include <sys/ucontext.h>
#include <syms.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdarg.h>

#include "papi_defines.h"

////////////////////////////////////////////////////////////////////////////////
/// COPIED ITEMS FROM THE OLD PORT TO SOLARIS //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/* DESCRIPTION:
 * -----------------------------------------------------------------------------
 * The following lines are taken from the old Solaris port of PAPI. If changes 
 * have been made there are (additional) comments.
 *
 ******************************************************************************/

#define MAX_COUNTERS 2
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define PAPI_MAX_NATIVE_EVENTS 71
#define MAX_NATIVE_EVENT PAPI_MAX_NATIVE_EVENTS

typedef int niagara2_reg_alloc_t;

/* libcpc 2 does not need any bit masks */
typedef struct _niagara2_register
{
	int event_code;
} _niagara2_register_t;


#define BUF_T0 0
#define BUF_T1 1

#define EVENT_NOT_SET -1;

#define SYNTHETIC_EVENTS_SUPPORTED  1

/* This structured bundles everything needed for sampling up to MAX_COUNTERS */
typedef struct _niagara2_control_state
{
	/* A set instruments the hardware counters */
	cpc_set_t *set;

	/* A buffer stores the events counted. For measuring a start of measurment
	   and an end is needed as measurement does not always start from 0. This is
	   done by using an array of bufs, accessed by the indexes BUF_T0 as start
	   and BUF_T1 as end. */
	cpc_buf_t *counter_buffer;

	/* The indexes are needed for accessing the single counter events, if the
	   value of these indexes is equal to EVENT_NOT_SET this means it is unused */
	int idx[MAX_COUNTERS];

	/* The event codes applied to this set */
	_niagara2_register_t code[MAX_COUNTERS];

	/* The total number of events being counted */
	int count;

	/* The values retrieved from the counter */
	uint64_t result[MAX_COUNTERS];

	/* Flags for controlling overflow handling and binding, see
	   cpc_set_create(3CPC) for more details on this topic. */
	uint_t flags[MAX_COUNTERS];

	/* Preset values for the counters */
	uint64_t preset[MAX_COUNTERS];

	/* Memory to store values when an overflow occours */
	long_long threshold[MAX_COUNTERS];
	long_long hangover[MAX_COUNTERS];

#ifdef SYNTHETIC_EVENTS_SUPPORTED
	int syn_count;
	uint64_t syn_hangover[MAX_COUNTERS];
#endif
} _niagara2_control_state_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void*)(ctx->ucontext->uc_mcontext.gregs[REG_PC])

typedef int hwd_register_map_t;

#include "solaris-context.h"

typedef _niagara2_control_state_t _niagara2_context_t;

// Needs an explicit declaration, no longer externally found.
rwlock_t lock[PAPI_MAX_LOCK];

// For setting and releasing locks.
#define _papi_hwd_lock(lck)     rw_wrlock(&lock[lck]);
#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);

#define DEFAULT_CNTR_PRESET (0)
#define NOT_A_PAPI_HWD_READ -666
#define CPC_COUNTING_DOMAINS (CPC_COUNT_USER|CPC_COUNT_SYSTEM|CPC_COUNT_HV)
#define EVENT_NOT_SET -1;

/* Clean the stubbed data structures from framework initialization */
#undef  hwd_context_t
#define hwd_context_t		_niagara2_context_t

#undef  hwd_control_state_t
#define hwd_control_state_t	_niagara2_control_state_t

#undef  hwd_register_t
#define hwd_register_t		_niagara2_register_t

#endif
