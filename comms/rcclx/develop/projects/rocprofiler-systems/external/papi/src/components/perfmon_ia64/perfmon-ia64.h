#ifndef _PAPI_PERFMON_IA64_H
#define _PAPI_PERFMON_IA64_H
/* 
* File:    perfmon-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*          Kevin London
*	   london@cs.utk.edu
*
* Mods:    Per Ekman
*          pek@pdc.kth.se
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
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>
#include <sys/types.h>
#include <sys/ipc.h>

#if defined(HAVE_MMTIMER)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/mmtimer.h>
#ifndef MMTIMER_FULLNAME
#define MMTIMER_FULLNAME "/dev/mmtimer"
#endif
#endif

#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
#endif

#include "papi_defines.h"
#include "config.h"
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
#include "perfmon/perfmon_default_smpl.h"
#include "perfmon/pfmlib_montecito.h"
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_itanium.h"

typedef int ia64_register_t;
typedef int ia64_register_map_t;
typedef int ia64_reg_alloc_t;


#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

typedef struct param_t
{
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_reg_t pc[NUM_PMCS];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	void *mod_inp;					   /* model specific input parameters to libpfm    */
	void *mod_outp;					   /* model specific output parameters from libpfm */
} pfmw_param_t;
//   #ifdef ITANIUM3
typedef struct mont_param_t
{
	pfmlib_mont_input_param_t mont_input_param;
	pfmlib_mont_output_param_t mont_output_param;
} pfmw_mont_param_t;
//   typedef pfmw_mont_param_t pfmw_ita_param_t;
//   #elif defined(ITANIUM2)
typedef struct ita2_param_t
{
	pfmlib_ita2_input_param_t ita2_input_param;
	pfmlib_ita2_output_param_t ita2_output_param;
} pfmw_ita2_param_t;
//   typedef pfmw_ita2_param_t pfmw_ita_param_t;
//   #else
typedef int pfmw_ita1_param_t;
//   #endif

#define PMU_FIRST_COUNTER  4

typedef union
{
	pfmw_ita1_param_t ita_param;
	pfmw_ita2_param_t ita2_param;
	pfmw_mont_param_t mont_param;
} pfmw_ita_param_t;


#define MAX_COUNTERS 12
#define MAX_COUNTER_TERMS MAX_COUNTERS

typedef struct ia64_control_state
{
	/* Which counters to use? Bits encode counters to use, may be duplicates */
	ia64_register_map_t bits;

	pfmw_ita_param_t ita_lib_param;

	/* Buffer to pass to kernel to control the counters */
	pfmw_param_t evt;

	long long counters[MAX_COUNTERS];
	pfarg_reg_t pd[NUM_PMDS];

/* sampling buffer address */
	void *smpl_vaddr;
	/* Buffer to pass to library to control the counters */
} ia64_control_state_t;


typedef struct itanium_preset_search
{
	/* Preset code */
	int preset;
	/* Derived code */
	int derived;
	/* Strings to look for */
	char *( findme[MAX_COUNTERS] );
	char operation[MAX_COUNTERS * 5];
} itanium_preset_search_t;

typedef struct
{
	int fd;							   /* file descriptor */
	pid_t tid;						   /* thread id */
#if defined(USE_PROC_PTTIMER)
	int stat_fd;
#endif
} ia64_context_t;

#undef hwd_context_t
typedef ia64_context_t hwd_context_t;

#include "linux-context.h"

//#undef  hwd_ucontext_t
//typedef struct sigcontext hwd_ucontext_t;

/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef  hwd_reg_alloc_t
typedef ia64_reg_alloc_t hwd_reg_alloc_t;
#undef  hwd_register_t
typedef ia64_register_t hwd_register_t;
#undef  hwd_control_state_t
typedef ia64_control_state_t hwd_control_state_t;

#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))

#define MONT_DEAR_REGS_MASK	    (M_PMD(32)|M_PMD(33)|M_PMD(36))
#define MONT_ETB_REGS_MASK		(M_PMD(38)| M_PMD(39)| \
		                 M_PMD(48)|M_PMD(49)|M_PMD(50)|M_PMD(51)|M_PMD(52)|M_PMD(53)|M_PMD(54)|M_PMD(55)|\
				 M_PMD(56)|M_PMD(57)|M_PMD(58)|M_PMD(59)|M_PMD(60)|M_PMD(61)|M_PMD(62)|M_PMD(63))

#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))

#endif /* _PAPI_PERFMON_IA64_H */
