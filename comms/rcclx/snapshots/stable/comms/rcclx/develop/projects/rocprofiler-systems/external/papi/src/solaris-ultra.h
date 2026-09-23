#ifndef _PAPI_SOLARIS_ULTRA_H
#define _PAPI_SOLARIS_ULTRA_H

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
#include <syms.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include "papi_defines.h"

#define MAX_COUNTERS 2
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define PAPI_MAX_NATIVE_EVENTS 71
#define MAX_NATIVE_EVENT PAPI_MAX_NATIVE_EVENTS
#define MAX_NATIVE_EVENT_USII  22

/* Defines in papi_internal.h cause compile warnings on solaris because typedefs are done here */
#undef hwd_context_t
#undef hwd_control_state_t
#undef hwd_reg_alloc_t
#undef hwd_register_t
#undef hwd_siginfo_t
#undef hwd_ucontext_t

typedef int hwd_reg_alloc_t;

typedef struct US_register
{
	int event[MAX_COUNTERS];
} hwd_register_t;

typedef struct papi_cpc_event
{
	/* Structure to libcpc */
	cpc_event_t cmd;
	/* Flags to kernel */
	int flags;
} papi_cpc_event_t;

typedef struct hwd_control_state
{
	/* Buffer to pass to the kernel to control the counters */
	papi_cpc_event_t counter_cmd;
	/* overflow event counter */
	int overflow_num;
} hwd_control_state_t;

typedef int hwd_register_map_t;

typedef struct _native_info
{
	/* native name */
	char name[40];
	/* Buffer to pass to the kernel to control the counters */
	int encoding[MAX_COUNTERS];
} native_info_t;

#include "solaris-context.h"

typedef int hwd_context_t;

/* Assembler prototypes */

extern void cpu_sync( void );
extern unsigned long long get_tick( void );
extern vptr_t _start, _end, _etext, _edata;

extern rwlock_t lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);

#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);

#endif
