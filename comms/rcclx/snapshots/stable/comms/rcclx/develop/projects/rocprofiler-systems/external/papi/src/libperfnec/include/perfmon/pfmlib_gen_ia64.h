/*
 * Generic IA-64 PMU specific types and definitions
 *
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
#ifndef __PFMLIB_GEN_IA64_H__
#define __PFMLIB_GEN_IA64_H__

#include <perfmon/pfmlib.h>
#include <endian.h>

#if BYTE_ORDER != LITTLE_ENDIAN
#error "this file only supports little endian environments"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_GEN_IA64_FIRST_COUNTER	4	/* index of first PMC/PMD counter */
#define PMU_GEN_IA64_NUM_COUNTERS	4	/* total numbers of PMC/PMD pairs used as counting monitors */
#define PMU_GEN_IA64_NUM_PMCS		8	/* total number of PMCS defined */
#define PMU_GEN_IA64_NUM_PMDS		4	/* total number of PMDS defined */

/*
 * architected PMC register structure
 */
typedef union {
	unsigned long pmc_val;			/* generic PMC register */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* reserved */
		unsigned long pmc_es:8;		/* event select */
		unsigned long pmc_ig2:48;	/* reserved */
	} pmc_gen_count_reg;
} pfm_gen_ia64_pmc_reg_t;

typedef struct {
	unsigned long	pmd_val;	/* generic counter value */
} pfm_gen_ia64_pmd_reg_t;

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_GEN_IA64_H__ */

