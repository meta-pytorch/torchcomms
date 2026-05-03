/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/
/* 
* File:    freebsd-libpmc.c
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    Harald Servat
*          redcrash@gmail.com
*/

#ifndef _PAPI_FreeBSD_H
#define _PAPI_FreeBSD_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "papi.h"
#include <pmc.h>

#include "freebsd-config.h"

#define MAX_COUNTERS		HWPMC_NUM_COUNTERS
#define MAX_COUNTER_TERMS	MAX_COUNTERS

#undef hwd_siginfo_t
#undef hwd_register_t
#undef hwd_reg_alloc_t
#undef hwd_control_state_t
#undef hwd_context_t
#undef hwd_libpmc_context_t

typedef struct hwd_siginfo {
	int placeholder;
} hwd_siginfo_t;

typedef struct hwd_register {
	int placeholder;
} hwd_register_t;

typedef struct hwd_reg_alloc {
	int placeholder;
} hwd_reg_alloc_t;

typedef struct hwd_control_state {
	int n_counters;      /* Number of counters */
	int hwc_domain;      /* HWC domain {user|kernel} */
	unsigned *caps;      /* Capabilities for each counter */
	pmc_id_t *pmcs;      /* PMC identifiers */
	pmc_value_t *values; /* Stored values for each counter */
	char **counters;     /* Name of each counter (with mode) */
} hwd_control_state_t;

typedef struct hwd_context {
	int placeholder; 
} hwd_context_t;

#include "freebsd-context.h"

typedef struct hwd_libpmc_context {
	int CPUtype;
	int use_rdtsc;
} hwd_libpmc_context_t;

#define _papi_hwd_lock_init() { ; }

#endif /* _PAPI_FreeBSD_H */
