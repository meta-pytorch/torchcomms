#ifndef _PAPI_PERFMON_H
#define _PAPI_PERFMON_H
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
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon_dfl_smpl.h"
#include "papi_lock.h"


#include "linux-context.h"

#if defined(DEBUG)
#define DEBUGCALL(a,b) { if (ISLEVEL(a)) { b; } }
#else
#define DEBUGCALL(a,b)
#endif

typedef pfmlib_event_t pfm_register_t;
typedef int pfm_register_map_t;
typedef int pfm_reg_alloc_t;

#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

typedef struct
{
	/* Context structure to kernel, different for attached */
	int ctx_fd;
	pfarg_ctx_t *ctx;
	/* Load structure to kernel, different for attached */
	pfarg_load_t *load;
	/* Which counters to use? Bits encode counters to use, may be duplicates */
	pfm_register_map_t bits;
	/* Buffer to pass to library to control the counters */
	pfmlib_input_param_t in;
	/* Buffer to pass from the library to control the counters */
	pfmlib_output_param_t out;
	/* Is this eventset multiplexed? Actually it holds the microseconds of the switching interval, 0 if not mpx. */
	int multiplexed;
	/* Arguments to kernel for multiplexing, first number of sets */
	int num_sets;
	/* Arguments to kernel to set up the sets */
	pfarg_setdesc_t set[PFMLIB_MAX_PMDS];
	/* Buffer to get information out of the sets when reading */
	pfarg_setinfo_t setinfo[PFMLIB_MAX_PMDS];
	/* Arguments to the kernel */
	pfarg_pmc_t pc[PFMLIB_MAX_PMCS];
	/* Arguments to the kernel */
	pfarg_pmd_t pd[PFMLIB_MAX_PMDS];
	/* Buffer to gather counters */
	long long counts[PFMLIB_MAX_PMDS];
} pfm_control_state_t;

typedef struct
{
#if defined(USE_PROC_PTTIMER)
	int stat_fd;
#endif
	/* Main context structure to kernel */
	int ctx_fd;
	pfarg_ctx_t ctx;
	/* Main load structure to kernel */
	pfarg_load_t load;
	/* Structure to inform the kernel about sampling */
	pfm_dfl_smpl_arg_t smpl;
	/* Address of mmap()'ed sample buffer */
	void *smpl_buf;
} pfm_context_t;

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef pfm_reg_alloc_t cmp_reg_alloc_t;
typedef pfm_register_t cmp_register_t;
typedef pfm_control_state_t cmp_control_state_t;
typedef pfm_context_t cmp_context_t;

#endif
