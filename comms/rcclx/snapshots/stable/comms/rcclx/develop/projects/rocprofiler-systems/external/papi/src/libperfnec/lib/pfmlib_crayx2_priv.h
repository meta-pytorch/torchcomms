/*
 * Copyright (c) 2007 Cray Inc.
 * Contributed by Steve Kaufmann <sbk@cray.com> based on code from
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PMLIB_CRAYX2_PRIV_H__
#define __PMLIB_CRAYX2_PRIV_H__ 1

#include <perfmon/pfmlib_crayx2.h>

/*	Chips (substrates) that contain performance counters.
 */
#define PME_CRAYX2_CHIP_CPU 1
#define PME_CRAYX2_CHIP_CACHE 2
#define PME_CRAYX2_CHIP_MEMORY 3

/*	Number of chips monitored per single process.
 */
#define PME_CRAYX2_CPU_CHIPS 1
#define PME_CRAYX2_CACHE_CHIPS 1
#define PME_CRAYX2_MEMORY_CHIPS 16

/*	Number of events per physical counter.
 */
#define PME_CRAYX2_EVENTS_PER_COUNTER 4

/*	Number of counters per chip (CPU, L2 Cache, Memory)
 */
#define PME_CRAYX2_CPU_CTRS_PER_CHIP PFM_CPU_PMD_COUNT
#define PME_CRAYX2_CACHE_CTRS_PER_CHIP PFM_CACHE_PMD_PER_CHIP
#define PME_CRAYX2_MEMORY_CTRS_PER_CHIP PFM_MEM_PMD_PER_CHIP

/*	Number of events per chip (CPU, L2 Cache, Memory)
*/
#define PME_CRAYX2_CPU_EVENTS \
	(PME_CRAYX2_CPU_CHIPS*PME_CRAYX2_CPU_CTRS_PER_CHIP*PME_CRAYX2_EVENTS_PER_COUNTER)
#define PME_CRAYX2_CACHE_EVENTS \
	(PME_CRAYX2_CACHE_CHIPS*PME_CRAYX2_CACHE_CTRS_PER_CHIP*PME_CRAYX2_EVENTS_PER_COUNTER)
#define PME_CRAYX2_MEMORY_EVENTS \
	(PME_CRAYX2_MEMORY_CHIPS*PME_CRAYX2_MEMORY_CTRS_PER_CHIP*PME_CRAYX2_EVENTS_PER_COUNTER)

/*	No unit masks are (currently) used.
 */
#define PFMLIB_CRAYX2_MAX_UMASK 1

typedef struct {
	const char	*pme_uname;	/* unit mask name */
	const char	*pme_udesc;	/* event/umask description */
	unsigned int	pme_ucode;	/* unit mask code */
} pme_crayx2_umask_t;

/*	Description of each performance counter event available on all
 *	substrates. Listed contiguously for all substrates.
 */
typedef struct {
	const char	*pme_name;	/* event name */
	const char	*pme_desc;	/* event description */
	unsigned int	pme_code;	/* event code */
	unsigned int	pme_flags;	/* flags */
	unsigned int	pme_numasks;	/* number of unit masks */
	pme_crayx2_umask_t pme_umasks[PFMLIB_CRAYX2_MAX_UMASK];
					/* unit masks (chip numbers) */
	unsigned int	pme_chip;	/* substrate/chip containing counter */
	unsigned int	pme_ctr;	/* counter on chip */
	unsigned int	pme_event;	/* event number on counter */
	unsigned int	pme_chipno;	/* chip# upon which the event lies */
	unsigned int	pme_base;	/* PMD base reg_num for this chip */
	unsigned int	pme_nctrs;	/* PMDs/counters per chip */
	unsigned int	pme_nchips;	/* number of chips per process */
} pme_crayx2_entry_t;

#endif /* __PMLIB_CRAYX2_PRIV_H__ */
