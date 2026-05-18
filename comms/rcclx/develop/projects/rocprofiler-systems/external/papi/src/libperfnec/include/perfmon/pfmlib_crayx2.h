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

#ifndef __PFMLIB_CRAYX2_H__
#define __PFMLIB_CRAYX2_H__ 1
/*
 * Allows <asm/perfmon.h> to be included on its own.
 */
#define PFM_MAX_HW_PMCS 12
#define PFM_MAX_HW_PMDS 512

#include <asm/perfmon.h>
#include <sys/types.h>

/*	Priviledge level mask for Cray-X2:
 *
 *	PFM_PLM0 = Kernel
 *	PFM_PLM1 = Kernel
 *	PFM_PLM2 = Exception
 *	PFM_PLM3 = User 
 */

/*	The performance control (PMC) registers appear as follows:
 *	PMC0	control for CPU chip
 *	PMC1	events on CPU chip
 *	PMC2	enable for CPU chip
 *	PMC3	control for L2 Cache chip
 *	PMC4	events on L2 Cache chip
 *	PMC5	enable for L2 Cache chip
 *	PMC6	control for Memory chip
 *	PMC7	events on Memory chip
 *	PMC8	enable for Memory chip
 *
 *	The performance data (PMD) registers appear for
 *	CPU (32), L2 Cache (16), and Memory (28*16) chips contiguously.
 *	There are four events per chip.
 *
 *	PMD0	P chip, counter 0
 *	...
 *	PMD31	P chip, counter 31
 *	PMD32	C chip, counter 0
 *	...
 *	PMD47	C chip, counter 15
 *	PMD48	M chip 0, counter 0
 *	...
 *	PMD495	M chip 15, counter 27
 */

#ifdef __cplusplus
extern "C" {
#endif

/*	PMC counts
 */
#define PMU_CRAYX2_CPU_PMC_COUNT PFM_CPU_PMC_COUNT
#define PMU_CRAYX2_CACHE_PMC_COUNT PFM_CACHE_PMC_COUNT
#define PMU_CRAYX2_MEMORY_PMC_COUNT PFM_MEM_PMC_COUNT

/*	PMC bases
 */
#define PMU_CRAYX2_CPU_PMC_BASE PFM_CPU_PMC
#define PMU_CRAYX2_CACHE_PMC_BASE PFM_CACHE_PMC
#define PMU_CRAYX2_MEMORY_PMC_BASE PFM_MEM_PMC

/*	PMD counts
 */
#define PMU_CRAYX2_CPU_PMD_COUNT PFM_CPU_PMD_COUNT
#define PMU_CRAYX2_CACHE_PMD_COUNT PFM_CACHE_PMD_COUNT
#define PMU_CRAYX2_MEMORY_PMD_COUNT PFM_MEM_PMD_COUNT

/*	PMD bases
*/
#define PMU_CRAYX2_CPU_PMD_BASE PFM_CPU_PMD
#define PMU_CRAYX2_CACHE_PMD_BASE PFM_CACHE_PMD
#define PMU_CRAYX2_MEMORY_PMD_BASE PFM_MEM_PMD

/*	Total number of PMCs and PMDs
 */
#define PMU_CRAYX2_PMC_COUNT PFM_PMC_COUNT
#define PMU_CRAYX2_PMD_COUNT PFM_PMD_COUNT
#define PMU_CRAYX2_NUM_COUNTERS PFM_PMD_COUNT

/*	Counter width (can also be acquired via /sys/kernel/perfmon)
 */
#define PMU_CRAYX2_COUNTER_WIDTH 63

/*	PMU name (can also be acquired via /sys/kernel/perfmon)
 */
#define PMU_CRAYX2_NAME "Cray X2"

#ifdef __cplusplus
}
#endif /* extern C */

#endif /* __PFMLIB_CRAYX2_H__ */
