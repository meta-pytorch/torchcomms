#ifndef _PAPI_AIX_H			 /* _PAPI_AIX */
#define _PAPI_AIX_H

/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    pmapi-ppc64.h
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/



#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <time.h>
#if defined( _AIXVERSION_510) || defined(_AIXVERSION_520)
#include <sys/procfs.h>
#include <sys/cred.h>
#endif
#include <procinfo.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <sys/atomic_op.h>
#include <sys/utsname.h>


#include "pmapi.h"

#define ANY_THREAD_GETS_SIGNAL
#define POWER_MAX_COUNTERS MAX_COUNTERS
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define MAX_MPX_COUNTERS 32
#define INVALID_EVENT -2
#define POWER_MAX_COUNTERS_MAPPING 8

extern _text;
extern _etext;
extern _edata;
extern _end;
extern _data;

/* globals */
#ifdef PM_INITIALIZE
#ifdef _AIXVERSION_510
#define PMINFO_T pm_info2_t
#define PMEVENTS_T pm_events2_t
#else
#define PMINFO_T pm_info_t
#define PMEVENTS_T pm_events_t
#endif
PMINFO_T pminfo;
#else
#define PMINFO_T pm_info_t
#define PMEVENTS_T pm_events_t
/*pm_info_t pminfo;*/
#endif

#include "aix-context.h"

/* define the vector structure at the bottom of this file */

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS

#ifdef PM_INITIALIZE
typedef pm_info2_t hwd_pminfo_t;
typedef pm_events2_t hwd_pmevents_t;
#else
typedef pm_info_t hwd_pminfo_t;
typedef pm_events_t hwd_pmevents_t;
#endif

#include "ppc64_events.h"

typedef struct ppc64_pmapi_control
{
	/* Buffer to pass to the kernel to control the counters */
	pm_prog_t counter_cmd;
	int group_id;
	/* Space to read the counters */
	pm_data_t state;
} ppc64_pmapi_control_t;

typedef struct ppc64_reg_alloc
{
	int ra_position;
	unsigned int ra_group[GROUP_INTS];
	int ra_counter_cmd[MAX_COUNTERS];
} ppc64_reg_alloc_t;

typedef struct ppc64_pmapi_context
{
	/* this structure is a work in progress */
	ppc64_pmapi_control_t cntrl;
} ppc64_pmapi_context_t;

/* Override void* definitions from PAPI framework layer */
/* typedefs to conform to hardware independent PAPI code. */
#undef hwd_control_state_t
#undef hwd_reg_alloc_t
#undef hwd_context_t
typedef ppc64_pmapi_control_t hwd_control_state_t;
typedef ppc64_reg_alloc_t hwd_reg_alloc_t;
typedef ppc64_pmapi_context_t hwd_context_t;

/*
typedef struct hwd_groups {
  // group number from the pmapi pm_groups_t struct 
  //int group_id;
  // Buffer containing counter cmds for this group 
  unsigned char counter_cmd[POWER_MAX_COUNTERS];
} hwd_groups_t;
*/

/* prototypes */
extern int _aix_set_granularity( hwd_control_state_t * this_state, int domain );
extern int _papi_hwd_init_preset_search_map( hwd_pminfo_t * info );

extern int _aix_get_memory_info( PAPI_hw_info_t * mem_info, int type );
extern int _aix_get_dmem_info( PAPI_dmem_info_t * d );

/* Machine dependent info structure */
extern pm_groups_info_t pmgroups;

#endif /* _PAPI_AIX */

