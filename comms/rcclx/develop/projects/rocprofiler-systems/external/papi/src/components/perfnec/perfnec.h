#ifndef _PAPI_PERFNEC_H
#define _PAPI_PERFNEC_H
/* 
* File:    perfmon.h
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <fcntl.h>
#include <ctype.h>
#include <inttypes.h>
#include <libgen.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include "perfnec/pfmlib.h"
//#include "perfnec/perfmon_dfl_smpl.h"
#include "papi_lock.h"


#include "linux-context.h"

#if defined(DEBUG)
#define DEBUGCALL(a,b) { if (ISLEVEL(a)) { b; } }
#else
#define DEBUGCALL(a,b)
#endif

#define PFNECLIB_MAX_PMDS 32

typedef int pfnec_register_map_t;
typedef int pfnec_reg_alloc_t;

typedef int pfnec_dfl_smpl_arg_t;
typedef int pfnec_dfl_smpl_hdr_t;
typedef int pfnec_register_t;
typedef int pfneclib_regmask_t;
typedef int pfnec_dfl_smpl_entry_t;

#define PKG_NUM_EVENTS 16

static int   pkg_events[PKG_NUM_EVENTS]
  = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
static const char *pkg_event_names[PKG_NUM_EVENTS]
  = {"EX",  "VX",  "FPEC", "VE", "VECC", "L1MCC", "VE2", "VAREC",
     "VLDEC", "PCCC", "VLDCC", "VLEC", "VLCME2", "FMAEC", "PTCC", "TTCC"};
static const char *pkg_units[PKG_NUM_EVENTS]
  = {"", "", "", "",
     "", "", "", "",
     "", "", "", "",
     "", "", "", ""
    };
static const char *pkg_event_descs[PKG_NUM_EVENTS] = {
	"Execution count",
	"Vector execution count",
	"Floating point data element count",
	"Vector elements count",
	"Vector execution clock count",
	"L1 cache miss clock count",
	"Vector elements count 2",
	"Vector arithmetic execution clock count",
	"Vector load execution clock count",
	"Port conflict clock count",
	"Vector load packet count",
	"Vector load element count",
	"Vector load cache miss element count 2",
	"Fused multiply add element count",
	"Power throttling clock count",
	"Thermal throttling clock count"};

typedef struct _perfnec_register {
    unsigned int selector;
} _perfnec_register_t;

typedef struct _perfnec_native_event_entry {
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  int socket_id;
  int component_id;
  int event_id;
  int type;
  int return_type;
  _perfnec_register_t resources;
} _perfnec_native_event_entry_t;

static _perfnec_native_event_entry_t perfnec_ntv_events[PKG_NUM_EVENTS];

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

/* Null terminated version of strncpy */
static
char *_local_strlcpy( char *dst, const char *src, size_t size )
{
    char *retval = strncpy( dst, src, size );
    if ( size > 0 ) dst[size-1] = '\0';

    return( retval );
}


#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

typedef struct
{
	/* Context structure to kernel, different for attached */
	int ctx_fd;
	long long count[PFNECLIB_MAX_PMDS];
    int active_counters;
    long long which_counter[PKG_NUM_EVENTS];
} pfnec_control_state_t;

typedef struct
{
#if defined(USE_PROC_PTTIMER)
	int stat_fd;
#endif
	/* Main context structure to kernel */
	int ctx_fd;
	/* Address of mmap()'ed sample buffer */
	void *smpl_buf;
} pfnec_context_t;

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef pfnec_reg_alloc_t cmp_reg_alloc_t;
typedef pfnec_register_t cmp_register_t;
typedef pfnec_control_state_t _perfnec_control_state_t;
typedef pfnec_context_t _perfnec_context_t;

#endif
