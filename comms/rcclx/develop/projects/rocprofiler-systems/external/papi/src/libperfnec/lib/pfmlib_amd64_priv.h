/*
 * Copyright (c) 2004-2006 Hewlett-Packard Development Company, L.P.
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */
#ifndef __PFMLIB_AMD64_PRIV_H__
#define __PFMLIB_AMD64_PRIV_H__

/*
 * PERFSEL/PERFCTR include IBS registers:
 *
 *		PMCs	PMDs
 *
 * PERFCTRS	6	6
 * IBS FETCH	1	3
 * IBS OP	1	7
 *
 * total	8	16
 */
#define PMU_AMD64_NUM_PERFSEL		8	/* number of PMCs defined */
#define PMU_AMD64_NUM_PERFCTR		16	/* number of PMDs defined */
#define PMU_AMD64_NUM_COUNTERS		4	/* number of EvtSel/EvtCtr */
#define PMU_AMD64_NUM_COUNTERS_F15H	6	/* number of EvtSel/EvtCtr */
#define PMU_AMD64_COUNTER_WIDTH		48	/* hw counter bit width */
#define PMU_AMD64_CNT_MASK_MAX		4 	/* max cnt_mask value */
#define PMU_AMD64_IBSFETCHCTL_PMC	6	/* IBS: fetch PMC base */
#define PMU_AMD64_IBSFETCHCTL_PMD 	6	/* IBS: fetch PMD base */
#define PMU_AMD64_IBSOPCTL_PMC		7	/* IBS: op PMC base */
#define PMU_AMD64_IBSOPCTL_PMD		9	/* IBS: op PMD base */

#define PFMLIB_AMD64_MAX_UMASK	13

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_uflags; /* unit mask flags */
} pme_amd64_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	pme_amd64_umask_t	pme_umasks[PFMLIB_AMD64_MAX_UMASK]; /* umask desc */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
} pme_amd64_entry_t;

typedef enum {
	AMD64_CPU_UN,
	AMD64_K7,
	AMD64_K8_REV_B,
	AMD64_K8_REV_C,
	AMD64_K8_REV_D,
	AMD64_K8_REV_E,
	AMD64_K8_REV_F,
	AMD64_K8_REV_G,
	AMD64_FAM10H_REV_B,
	AMD64_FAM10H_REV_C,
	AMD64_FAM10H_REV_D,
	AMD64_FAM10H_REV_E,
	AMD64_FAM15H_REV_B,
} amd64_rev_t;

static const char *amd64_rev_strs[]= {
	"?", "?",
	/* K8 */
	"B", "C", "D", "E", "F", "G",
	/* Family 10h */
	"B", "C", "D", "E",
	/* Family 15h */
	"B",
};

static const char *amd64_cpu_strs[] = {
	"AMD64 (unknown model)",
	"AMD64 (K7)",
	"AMD64 (K8 RevB)",
	"AMD64 (K8 RevC)",
	"AMD64 (K8 RevD)",
	"AMD64 (K8 RevE)",
	"AMD64 (K8 RevF)",
	"AMD64 (K8 RevG)",
	"AMD64 (Family 10h RevB, Barcelona)",
	"AMD64 (Family 10h RevC, Shanghai)",
	"AMD64 (Family 10h RevD, Istanbul)",
	"AMD64 (Family 10h RevE)",
	"AMD64 (Family 15h RevB)",
};

/* 
 * pme_flags values
 */
#define PFMLIB_AMD64_UMASK_COMBO	0x1 /* unit mask can be combined */
#define PFMLIB_AMD64_FROM_REV(rev)	((rev)<<8)
#define PFMLIB_AMD64_TILL_REV(rev)	((rev)<<16)
#define PFMLIB_AMD64_NOT_SUPP		0x1ff00
#define PFMLIB_AMD64_TILL_K8_REV_C	PFMLIB_AMD64_TILL_REV(AMD64_K8_REV_C)
#define PFMLIB_AMD64_K8_REV_D		PFMLIB_AMD64_FROM_REV(AMD64_K8_REV_D)
#define PFMLIB_AMD64_K8_REV_E		PFMLIB_AMD64_FROM_REV(AMD64_K8_REV_E)
#define PFMLIB_AMD64_TILL_K8_REV_E	PFMLIB_AMD64_TILL_REV(AMD64_K8_REV_E)
#define PFMLIB_AMD64_K8_REV_F		PFMLIB_AMD64_FROM_REV(AMD64_K8_REV_F)
#define PFMLIB_AMD64_TILL_FAM10H_REV_B	PFMLIB_AMD64_TILL_REV(AMD64_FAM10H_REV_B)
#define PFMLIB_AMD64_FAM10H_REV_C	PFMLIB_AMD64_FROM_REV(AMD64_FAM10H_REV_C)
#define PFMLIB_AMD64_TILL_FAM10H_REV_C	PFMLIB_AMD64_TILL_REV(AMD64_FAM10H_REV_C)
#define PFMLIB_AMD64_FAM10H_REV_D	PFMLIB_AMD64_FROM_REV(AMD64_FAM10H_REV_D)

static inline int from_revision(unsigned int flags)
{
	return ((flags) >> 8) & 0xff;
}

static inline int till_revision(unsigned int flags)
{
	int till = (((flags)>>16) & 0xff);
	if (!till)
		return 0xff;
	return till;
}

#endif /* __PFMLIB_AMD64_PRIV_H__ */
