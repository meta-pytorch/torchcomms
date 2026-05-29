/*
 * pfmlib_intel_nhm.c : Intel Nehalem PMU
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
 * Nehalem PMU = architectural perfmon v3 + OFFCORE + PEBS v2 + uncore PMU + LBR
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_intel_nhm.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_intel_nhm_priv.h"

/* Intel Westmere event tables */
#include "intel_wsm_events.h"
#include "intel_wsm_unc_events.h"

/* Intel Core i7 event tables */
#include "intel_corei7_events.h"
#include "intel_corei7_unc_events.h"

/* let's define some handy shortcuts! */
#define usel_event	unc_perfevtsel.usel_event
#define usel_umask	unc_perfevtsel.usel_umask
#define usel_occ	unc_perfevtsel.usel_occ
#define usel_edge	unc_perfevtsel.usel_edge
#define usel_int	unc_perfevtsel.usel_int
#define usel_en		unc_perfevtsel.usel_en
#define usel_inv	unc_perfevtsel.usel_inv
#define usel_cnt_mask	unc_perfevtsel.usel_cnt_mask

#define sel_event	perfevtsel.sel_event
#define sel_umask	perfevtsel.sel_umask
#define sel_usr		perfevtsel.sel_usr
#define sel_os		perfevtsel.sel_os
#define sel_edge	perfevtsel.sel_edge
#define sel_pc		perfevtsel.sel_pc
#define sel_int		perfevtsel.sel_int
#define sel_en		perfevtsel.sel_en
#define sel_inv		perfevtsel.sel_inv
#define sel_anythr	perfevtsel.sel_anythr
#define sel_cnt_mask	perfevtsel.sel_cnt_mask


/*
 * Description of the PMC registers mappings:
 *
 * 0  -> PMC0  -> PERFEVTSEL0
 * 1  -> PMC1  -> PERFEVTSEL1 
 * 2  -> PMC2  -> PERFEVTSEL2 
 * 3  -> PMC3  -> PERFEVTSEL3 
 * 16 -> PMC16 -> FIXED_CTR_CTRL
 * 17 -> PMC17 -> PEBS_ENABLED
 * 18 -> PMC18 -> PEBS_LD_LATENCY_THRESHOLD
 * 19 -> PMC19 -> OFFCORE_RSP0
 * 20 -> PMC20 -> UNCORE_FIXED_CTRL
 * 21 -> PMC21 -> UNCORE_EVNTSEL0
 * 22 -> PMC22 -> UNCORE_EVNTSEL1
 * 23 -> PMC23 -> UNCORE_EVNTSEL2
 * 24 -> PMC24 -> UNCORE_EVNTSEL3
 * 25 -> PMC25 -> UNCORE_EVNTSEL4
 * 26 -> PMC26 -> UNCORE_EVNTSEL5
 * 27 -> PMC27 -> UNCORE_EVNTSEL6
 * 28 -> PMC28 -> UNCORE_EVNTSEL7
 * 29 -> PMC31 -> UNCORE_ADDROP_MATCH
 * 30 -> PMC32 -> LBR_SELECT
 *
 * Description of the PMD registers mapping:
 *
 * 0  -> PMD0  -> PMC0
 * 1  -> PMD1  -> PMC1
 * 2  -> PMD2  -> PMC2
 * 3  -> PMD3  -> PMC3
 * 16 -> PMD16 -> FIXED_CTR0
 * 17 -> PMD17 -> FIXED_CTR1
 * 18 -> PMD18 -> FIXED_CTR2
 * 19 not used
 * 20 -> PMD20 -> UNCORE_FIXED_CTR0
 * 21 -> PMD21 -> UNCORE_PMC0
 * 22 -> PMD22 -> UNCORE_PMC1
 * 23 -> PMD23 -> UNCORE_PMC2
 * 24 -> PMD24 -> UNCORE_PMC3
 * 25 -> PMD25 -> UNCORE_PMC4
 * 26 -> PMD26 -> UNCORE_PMC5
 * 27 -> PMD27 -> UNCORE_PMC6
 * 28 -> PMD28 -> UNCORE_PMC7
 *
 * 31 -> PMD31 -> LBR_TOS
 * 32-63 -> PMD32-PMD63 -> LBR_FROM_0/LBR_TO_0 - LBR_FROM15/LBR_TO_15
 */
#define NHM_SEL_BASE		0x186
#define NHM_CTR_BASE		0xc1
#define NHM_FIXED_CTR_BASE	0x309

#define UNC_NHM_SEL_BASE	0x3c0
#define UNC_NHM_CTR_BASE	0x3b0
#define UNC_NHM_FIXED_CTR_BASE	0x394

#define MAX_COUNTERS	28 /* highest implemented counter */

#define PFMLIB_NHM_ALL_FLAGS \
	(PFM_NHM_SEL_INV|PFM_NHM_SEL_EDGE|PFM_NHM_SEL_ANYTHR)

#define NHM_NUM_GEN_COUNTERS	4
#define NHM_NUM_FIXED_COUNTERS	3

pfm_pmu_support_t intel_nhm_support;
pfm_pmu_support_t intel_wsm_support;

static pfmlib_regmask_t nhm_impl_pmcs, nhm_impl_pmds;
static pfmlib_regmask_t nhm_impl_unc_pmcs, nhm_impl_unc_pmds;
static pme_nhm_entry_t *pe, *unc_pe;
static unsigned int num_pe, num_unc_pe;
static int cpu_model, aaj80;
static int pme_cycles, pme_instr;

#ifdef __i386__
static inline void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx,
			 unsigned int *ecx, unsigned int *edx)
{
	/*
	 * because ebx is used in Pic mode, we need to save/restore because
	 * cpuid clobbers it. I could not figure out a way to get ebx out in
	 * one cpuid instruction. To extract ebx, we need to  move it to another
	 * register (here eax)
	 */
	__asm__("pushl %%ebx;cpuid; popl %%ebx"
			:"=a" (*eax)
			: "a" (op)
			: "ecx", "edx");

	__asm__("pushl %%ebx;cpuid; movl %%ebx, %%eax;popl %%ebx"
			:"=a" (*ebx)
			: "a" (op)
			: "ecx", "edx");
}
#else
static inline void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx,
			 unsigned int *ecx, unsigned int *edx)
{
        __asm__("cpuid"
                        : "=a" (*eax),
                        "=b" (*ebx),
                        "=c" (*ecx),
                        "=d" (*edx)
                        : "0" (op), "c"(0));
}
#endif

static inline pme_nhm_entry_t *
get_nhm_entry(unsigned int i)
{
        return i < num_pe ? pe+i : unc_pe+(i-num_pe);
}

static int
pfm_nhm_midx2uidx(unsigned int ev, unsigned int midx)
{
	int i, num = 0;
	pme_nhm_entry_t *ne;
	int model;

	ne = get_nhm_entry(ev);

	for (i=0; i < ne->pme_numasks; i++) {
		model = ne->pme_umasks[i].pme_umodel;
		if (!model || model == cpu_model) {
			if (midx == num) 
				return i;
			num++;
		}
	}
	DPRINT("cannot find umask %d for event %s\n", midx, ne->pme_name);
	return -1;
}

static int
pfm_nhm_detect_common(void)
{
	int ret;
	int family;
	char buffer[128];

	ret = __pfm_getcpuinfo_attr("vendor_id", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	if (strcmp(buffer, "GenuineIntel"))
		return PFMLIB_ERR_NOTSUPP;

	ret = __pfm_getcpuinfo_attr("cpu family", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	family = atoi(buffer);

	ret = __pfm_getcpuinfo_attr("model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	cpu_model = atoi(buffer);

	if (family != 6)
		return PFMLIB_ERR_NOTSUPP;

	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_detect(void)
{
#define INTEL_ARCH_MISP_BR_RETIRED	(1 << 6)
	unsigned int eax, ebx, ecx, edx;
	int ret;

	ret = pfm_nhm_detect_common();
	if (ret != PFMLIB_SUCCESS)
		return ret;

	switch(cpu_model) {
		case 26: /* Nehalem */
		case 30:
		case 31:
		case 46:
			/*
	 		 * check for erratum AAJ80
	 		 *
	 		 * MISPREDICTED_BRANCH_RETIRED may be broken
	 		 * in which case it appears in the list of
	 		 * unavailable architected events
	 		 */
			cpuid(0xa, &eax, &ebx, &ecx, &edx);
			if (ebx & INTEL_ARCH_MISP_BR_RETIRED)
				aaj80 = 1;
			break;
		default:
			return PFMLIB_ERR_NOTSUPP;
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_wsm_detect(void)
{
	switch(cpu_model) {
		case 37: /* Westmere */
		case 44:
			break;
		default:
			return PFMLIB_ERR_NOTSUPP;
	}
	return PFMLIB_SUCCESS;
}

static inline void setup_nhm_impl_unc_regs(void)
{
	pfm_regmask_set(&nhm_impl_unc_pmds, 20);
	pfm_regmask_set(&nhm_impl_unc_pmds, 21);
	pfm_regmask_set(&nhm_impl_unc_pmds, 22);
	pfm_regmask_set(&nhm_impl_unc_pmds, 23);
	pfm_regmask_set(&nhm_impl_unc_pmds, 24);
	pfm_regmask_set(&nhm_impl_unc_pmds, 25);
	pfm_regmask_set(&nhm_impl_unc_pmds, 26);
	pfm_regmask_set(&nhm_impl_unc_pmds, 27);
	pfm_regmask_set(&nhm_impl_unc_pmds, 28);

	/* uncore */
	pfm_regmask_set(&nhm_impl_unc_pmcs, 20);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 21);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 22);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 23);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 24);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 25);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 26);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 27);
	pfm_regmask_set(&nhm_impl_unc_pmcs, 28);
	/* unnhm_addrop_match */
	pfm_regmask_set(&nhm_impl_unc_pmcs, 29);

}

static void
fixup_mem_uncore_retired(void)
{
	size_t i;

	for(i=0; i < PME_COREI7_EVENT_COUNT; i++) {
		if (corei7_pe[i].pme_code != 0xf)
			continue;
		
		/*
 		 * assume model46 umasks are at the end
 		 */
		corei7_pe[i].pme_numasks = 6;
		break;
	}
}

static int
pfm_nhm_init(void)
{
	pfm_pmu_support_t *supp;
	int i;
	int num_unc_cnt = 0;

	if (forced_pmu != PFMLIB_NO_PMU) {
		if (forced_pmu == PFMLIB_INTEL_NHM_PMU)
			cpu_model = 26;
		else
			cpu_model = 37;
	}

	/* core */
	pfm_regmask_set(&nhm_impl_pmcs, 0);
	pfm_regmask_set(&nhm_impl_pmcs, 1);
	pfm_regmask_set(&nhm_impl_pmcs, 2);
	pfm_regmask_set(&nhm_impl_pmcs, 3);
	pfm_regmask_set(&nhm_impl_pmcs, 16);
	pfm_regmask_set(&nhm_impl_pmcs, 17);
	pfm_regmask_set(&nhm_impl_pmcs, 18);
	pfm_regmask_set(&nhm_impl_pmcs, 19);

	pfm_regmask_set(&nhm_impl_pmds, 0);
	pfm_regmask_set(&nhm_impl_pmds, 1);
	pfm_regmask_set(&nhm_impl_pmds, 2);
	pfm_regmask_set(&nhm_impl_pmds, 3);
	pfm_regmask_set(&nhm_impl_pmds, 16);
	pfm_regmask_set(&nhm_impl_pmds, 17);
	pfm_regmask_set(&nhm_impl_pmds, 18);

	/* lbr */
	pfm_regmask_set(&nhm_impl_pmcs, 30);
	for(i=31; i < 64; i++)
		pfm_regmask_set(&nhm_impl_pmds, i);

	switch(cpu_model) {
	case 46:
		num_pe = PME_COREI7_EVENT_COUNT;
		num_unc_pe = 0;
		pe = corei7_pe;
		unc_pe = NULL;
		pme_cycles = PME_COREI7_UNHALTED_CORE_CYCLES;
		pme_instr = PME_COREI7_INSTRUCTIONS_RETIRED;
		num_unc_cnt = 0;
		fixup_mem_uncore_retired();
		supp = &intel_nhm_support;
		break;
	case 26: /* Nehalem */
	case 30: /* Lynnfield */
		num_pe = PME_COREI7_EVENT_COUNT;
		num_unc_pe = PME_COREI7_UNC_EVENT_COUNT;
		pe = corei7_pe;
		unc_pe = corei7_unc_pe;
		pme_cycles = PME_COREI7_UNHALTED_CORE_CYCLES;
		pme_instr = PME_COREI7_INSTRUCTIONS_RETIRED;
		setup_nhm_impl_unc_regs();
		num_unc_cnt = 9; /* one fixed + 8 generic */
		supp = &intel_nhm_support;
		break;
	case 37: /* Westmere */
	case 44:
		num_pe = PME_WSM_EVENT_COUNT;
		num_unc_pe = PME_WSM_UNC_EVENT_COUNT;
		pe = wsm_pe;
		unc_pe = intel_wsm_unc_pe;
		pme_cycles = PME_WSM_UNHALTED_CORE_CYCLES;
		pme_instr = PME_WSM_INSTRUCTIONS_RETIRED;
		setup_nhm_impl_unc_regs();
		num_unc_cnt = 9; /* one fixed + 8 generic */

		/* OFFCORE_RESPONSE_1 */
		pfm_regmask_set(&nhm_impl_pmcs, 31);
		supp = &intel_wsm_support;
		break;
	default:
		return PFMLIB_ERR_NOTSUPP;
	}
	
	supp->pme_count = num_pe + num_unc_pe;
	supp->num_cnt = NHM_NUM_GEN_COUNTERS
		      + NHM_NUM_FIXED_COUNTERS
		      + num_unc_cnt;
	/*
	 * propagate uncore registers to impl bitmaps
	 */
	pfm_regmask_or(&nhm_impl_pmds, &nhm_impl_pmds, &nhm_impl_unc_pmds);
	pfm_regmask_or(&nhm_impl_pmcs, &nhm_impl_pmcs, &nhm_impl_unc_pmcs);

	/*
	 * compute number of registers available
	 * not all CPUs may have uncore
	 */
	pfm_regmask_weight(&nhm_impl_pmds, &supp->pmd_count);
	pfm_regmask_weight(&nhm_impl_pmcs, &supp->pmc_count);

	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_is_fixed(pfmlib_event_t *e, unsigned int f)
{
	pme_nhm_entry_t *ne;
	unsigned int fl, flc, i;
	unsigned int mask = 0;

	ne = get_nhm_entry(e->event);
	fl = ne->pme_flags;

	/*
	 * first pass: check if event as a whole supports fixed counters
	 */
	switch(f) {
		case 0:
			mask = PFMLIB_NHM_FIXED0;
			break;
		case 1:
			mask = PFMLIB_NHM_FIXED1;
			break;
		case 2:
			mask = PFMLIB_NHM_FIXED2_ONLY;
			break;
		default:
			return 0;
	}
	if (fl & mask)
		return 1;
	/*
	 * second pass: check if unit mask supports fixed counter
	 *
	 * reject if mask not found OR if not all unit masks have
	 * same fixed counter mask
	 */
	flc = 0;
	for(i=0; i < e->num_masks; i++) {
		int midx = pfm_nhm_midx2uidx(e->event, e->unit_masks[i]);
		fl = ne->pme_umasks[midx].pme_uflags;
		if (fl & mask)
			flc++;
	}
	return flc > 0 && flc == e->num_masks ? 1 : 0;
}

/*
 * Allow combination of events when cnt_mask > 0 AND unit mask codes do
 * not overlap (otherwise, we do not know what is actually measured)
 */
static int
pfm_nhm_check_cmask(pfmlib_event_t *e, pme_nhm_entry_t *ne, pfmlib_nhm_counter_t *cntr)
{
	unsigned int ref, ucode;
	int i, j;

	if (!cntr)
		return -1;

	if (cntr->cnt_mask == 0)
		return -1;

	for(i=0; i < e->num_masks; i++) {
		int midx = pfm_nhm_midx2uidx(e->event, e->unit_masks[i]);
		ref =  ne->pme_umasks[midx].pme_ucode;
		for(j=i+1; j < e->num_masks; j++) {
			midx = pfm_nhm_midx2uidx(e->event, e->unit_masks[j]);
			ucode = ne->pme_umasks[midx].pme_ucode;
			if (ref & ucode)
				return -1;
		}
	}
	return 0;
}
 
/*
 * IMPORTANT: the interface guarantees that pfp_pmds[] elements are returned in the order the events
 *	      were submitted.
 */
static int
pfm_nhm_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_nhm_input_param_t *param, pfmlib_output_param_t *outp)
{
#define HAS_OPTIONS(x)	(cntrs && (cntrs[x].flags || cntrs[x].cnt_mask))
#define is_fixed_pmc(a) (a == 16 || a == 17 || a == 18)
#define is_uncore(a) (a > 19)

	pme_nhm_entry_t *ne;
	pfmlib_nhm_counter_t *cntrs;
	pfm_nhm_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	uint64_t val, unc_global_ctrl;
	uint64_t pebs_mask, ld_mask;
	unsigned long long fixed_ctr;
	unsigned int plm;
	unsigned int npc, npmc0, npmc01, nf2, nuf;
	unsigned int i, n, k, j, umask, use_pebs = 0;
	unsigned int assign_pc[PMU_NHM_NUM_COUNTERS];
	unsigned int next_gen, last_gen, u_flags;
	unsigned int next_unc_gen, last_unc_gen, lat;
	unsigned int offcore_rsp0_value = 0;
	unsigned int offcore_rsp1_value = 0;

	npc = npmc01 = npmc0 = nf2 = nuf = 0;
	unc_global_ctrl = 0;

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	n      = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_nhm_counters : NULL;
	pebs_mask = ld_mask = 0;
	use_pebs = param ? param->pfp_nhm_pebs.pebs_used : 0;
	lat = param ? param->pfp_nhm_pebs.ld_lat_thres : 0;

	if (n > PMU_NHM_NUM_COUNTERS)
		return PFMLIB_ERR_TOOMANY;

	/*
	 * error checking
	 */
	for(i=0; i < n; i++) {
		/*
		 * only supports two priv levels for perf counters
		 */
		if (e[i].plm & (PFM_PLM1|PFM_PLM2))
			return PFMLIB_ERR_INVAL;

		ne = get_nhm_entry(e[i].event);

		/* check for erratum AAJ80 */
		if (aaj80 && (ne->pme_code & 0xff) == 0xc5) {
			DPRINT("MISPREDICTED_BRANCH_RETIRED broken on this Nehalem processor, see eeratum AAJ80\n");
			return PFMLIB_ERR_NOTSUPP;
		}

		/*
		 * check for valid flags
		 */
		if (e[i].flags & ~PFMLIB_NHM_ALL_FLAGS)
			return PFMLIB_ERR_INVAL;

		if (ne->pme_flags & PFMLIB_NHM_UMASK_NCOMBO
		    && e[i].num_masks > 1 && pfm_nhm_check_cmask(e, ne, cntrs ? cntrs+i : NULL)) {
			DPRINT("events does not support unit mask combination\n");
				return PFMLIB_ERR_NOASSIGN;
		}
		/*
		 * check event-level single register constraint for uncore fixed
		 */
		if (ne->pme_flags & PFMLIB_NHM_UNC_FIXED) {
			if (++nuf > 1) {
				DPRINT("two events compete for a UNCORE_FIXED_CTR0\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (HAS_OPTIONS(i)) {
				DPRINT("uncore fixed counter does not support options\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		if (ne->pme_flags & PFMLIB_NHM_PMC0) {
			if (++npmc0 > 1) {
				DPRINT("two events compete for a PMC0\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
		 * check event-level single register constraint (PMC0/1 only)
		 * fail if more than two events requested for the same counter pair
		 */
		if (ne->pme_flags & PFMLIB_NHM_PMC01) {
			if (++npmc01 > 2) {
				DPRINT("two events compete for a PMC0\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
 		 * UNHALTED_REFERENCE_CYCLES (CPU_CLK_UNHALTED:BUS)
 		 * can only be measured on FIXED_CTR2
 		 */
		if (ne->pme_flags & PFMLIB_NHM_FIXED2_ONLY) {
			if (++nf2 > 1) {
				DPRINT("two events compete for FIXED_CTR2\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (cntrs && ((cntrs[i].flags & (PFM_NHM_SEL_INV|PFM_NHM_SEL_EDGE)) || cntrs[i].cnt_mask)) {
				DPRINT("UNHALTED_REFERENCE_CYCLES only accepts anythr filter\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
 		 * OFFCORE_RSP0 is shared, unit masks for all offcore_response events
 		 * must be identical
 		 */
		umask = 0;
                for(j=0; j < e[i].num_masks; j++) {
			int midx = pfm_nhm_midx2uidx(e[i].event, e[i].unit_masks[j]);
			umask |= ne->pme_umasks[midx].pme_ucode;
		}

		if (ne->pme_flags & PFMLIB_NHM_OFFCORE_RSP0) {
			if (offcore_rsp0_value && offcore_rsp0_value != umask) {
				DPRINT("all OFFCORE_RSP0 events must have the same unit mask\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (pfm_regmask_isset(r_pmcs, 19)) {
				DPRINT("OFFCORE_RSP0 register not available\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (!((umask & 0xff) && (umask & 0xff00))) {
				DPRINT("OFFCORE_RSP0 must have at least one request and response unit mask set\n");
				return PFMLIB_ERR_INVAL;
			}
			/* lock-in offcore_value */
			offcore_rsp0_value = umask;
		}
		if (ne->pme_flags & PFMLIB_NHM_OFFCORE_RSP1) {
			if (offcore_rsp1_value && offcore_rsp1_value != umask) {
				DPRINT("all OFFCORE_RSP1 events must have the same unit mask\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (pfm_regmask_isset(r_pmcs, 31)) {
				DPRINT("OFFCORE_RSP1 register not available\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (!((umask & 0xff) && (umask & 0xff00))) {
				DPRINT("OFFCORE_RSP1 must have at least one request and response unit mask set\n");
				return PFMLIB_ERR_INVAL;
			}
			/* lock-in offcore_value */
			offcore_rsp1_value = umask;
		}

		/*
		 * enforce PLM0|PLM3 for uncore events given they have no
		 * priv level filter. This is to ensure users understand what
		 * they are doing
		 */
		if (ne->pme_flags & (PFMLIB_NHM_UNC|PFMLIB_NHM_UNC_FIXED)) {
			if (inp->pfp_dfl_plm != (PFM_PLM0|PFM_PLM3)
			    && e[i].plm != (PFM_PLM0|PFM_PLM3)) {
				DPRINT("uncore events must have PLM0|PLM3\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
	}

	/*
	 * initilize to empty
	 */
	for(i=0; i < PMU_NHM_NUM_COUNTERS; i++)
		assign_pc[i] = -1;


	next_gen = 0; /* first generic counter */
	last_gen = 3; /* last generic counter */

	/*
	 * strongest constraint: only uncore_fixed_ctr0 or PMC0 only
	 */
	if (nuf || npmc0) {
		for(i=0; i < n; i++) {
			ne = get_nhm_entry(e[i].event);
			if (ne->pme_flags & PFMLIB_NHM_PMC0) {
				if (pfm_regmask_isset(r_pmcs, 0))
					return PFMLIB_ERR_NOASSIGN;
				assign_pc[i] = 0;
				next_gen = 1;
			}
			if (ne->pme_flags & PFMLIB_NHM_UNC_FIXED) {
				if (pfm_regmask_isset(r_pmcs, 20))
					return PFMLIB_ERR_NOASSIGN;
				assign_pc[i] = 20;
			}
		}
	}
	/*
	 * 2nd strongest constraint first: works only on PMC0 or PMC1
	 * On Nehalem, this constraint applies at the event-level
	 * (not unit mask level, fortunately)
	 *
	 * PEBS works on all 4 generic counters
	 *
	 * Because of sanity check above, we know we can find
	 * only up to 2 events with this constraint
	 */
	if (npmc01) {
		for(i=0; i < n; i++) {
			ne = get_nhm_entry(e[i].event);
			if (ne->pme_flags & PFMLIB_NHM_PMC01) {
				while (next_gen < 2 && pfm_regmask_isset(r_pmcs, next_gen))
					next_gen++;
				if (next_gen == 2)
					return PFMLIB_ERR_NOASSIGN;
				assign_pc[i] = next_gen++;
			}
		}
	}

	/*
	 * next constraint: fixed counters
	 *
	 * We abuse the mapping here for assign_pc to make it easier
	 * to provide the correct values for pd[].
	 * We use:
	 * 	- 16 : fixed counter 0 (pmc16, pmd16)
	 * 	- 17 : fixed counter 1 (pmc16, pmd17)
	 * 	- 18 : fixed counter 2 (pmc16, pmd18)
	 */
	fixed_ctr = pfm_regmask_isset(r_pmcs, 16) ? 0 : 0x7;
	if (fixed_ctr) {
		for(i=0; i < n; i++) {
			/* 
			 * Nehalem fixed counters (as for architected perfmon v3)
			 * does support anythr filter
			 */
			if (HAS_OPTIONS(i)) {
				if (use_pebs && pfm_nhm_is_pebs(e+i))
					continue;

				if (cntrs[i].flags != PFM_NHM_SEL_ANYTHR)
					continue;
			}
			if ((fixed_ctr & 0x1) && pfm_nhm_is_fixed(e+i, 0)) {
				assign_pc[i] = 16;
				fixed_ctr &= ~1;
			}
			if ((fixed_ctr & 0x2) && pfm_nhm_is_fixed(e+i, 1)) {
				assign_pc[i] = 17;
				fixed_ctr &= ~2;
			}
			if ((fixed_ctr & 0x4) && pfm_nhm_is_fixed(e+i, 2)) {
				assign_pc[i] = 18;
				fixed_ctr &= ~4;
			}
		}
	}
	/*
 	 * uncore events on any of the 8 counters
 	 */
	next_unc_gen = 21; /* first generic uncore counter config */
	last_unc_gen = 28; /* last generic uncore counter config */
	for(i=0; i < n; i++) {
		ne = get_nhm_entry(e[i].event);
		if (ne->pme_flags & PFMLIB_NHM_UNC) {
			for(; next_unc_gen <= last_unc_gen; next_unc_gen++) {
				if (!pfm_regmask_isset(r_pmcs, next_unc_gen))
					break;
			}
			if (next_unc_gen <= last_unc_gen)
				assign_pc[i] = next_unc_gen++;
			else {
				DPRINT("cannot assign generic uncore event\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
	}

	/*
	 * assign what is left of the generic events
	 */
	for(i=0; i < n; i++) {
		if (assign_pc[i] == -1) {
			for(; next_gen <= last_gen; next_gen++) {
				DPRINT("i=%d next_gen=%d last=%d isset=%d\n", i, next_gen, last_gen, pfm_regmask_isset(r_pmcs, next_gen));
				if (!pfm_regmask_isset(r_pmcs, next_gen))
					break;
			}
			if (next_gen <= last_gen) {
				assign_pc[i] = next_gen++;
			} else {
				DPRINT("cannot assign generic event\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
	}

	/*
 	 * setup core fixed counters
 	 */
	reg.val = 0;
	for (i=0; i < n ; i++ ) {
		if (!is_fixed_pmc(assign_pc[i]))
			continue;
		val = 0;
		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;
		if (plm & PFM_PLM0)
			val |= 1ULL;
		if (plm & PFM_PLM3)
			val |= 2ULL;
		if (cntrs && cntrs[i].flags & PFM_NHM_SEL_ANYTHR)
			val |= 4ULL;
		val |= 1ULL << 3;	 /* force APIC int (kernel may force it anyway) */

		reg.val |= val << ((assign_pc[i]-16)<<2);
	}

	if (reg.val) {
		pc[npc].reg_num   = 16;
		pc[npc].reg_value = reg.val;
		pc[npc].reg_addr  = 0x38D;
		pc[npc].reg_alt_addr  = 0x38D;

		__pfm_vbprintf("[FIXED_CTRL(pmc%u)=0x%"PRIx64" pmi0=1 en0=0x%"PRIx64" any0=%d pmi1=1 en1=0x%"PRIx64" any1=%d pmi2=1 en2=0x%"PRIx64" any2=%d] ",
				pc[npc].reg_num,
				reg.val,
				reg.val & 0x3ULL,
				!!(reg.val & 0x4ULL),
				(reg.val>>4) & 0x3ULL,
				!!((reg.val>>4) & 0x4ULL),
				(reg.val>>8) & 0x3ULL,
				!!((reg.val>>8) & 0x4ULL));

		if ((fixed_ctr & 0x1) == 0)
			__pfm_vbprintf("INSTRUCTIONS_RETIRED ");
		if ((fixed_ctr & 0x2) == 0)
			__pfm_vbprintf("UNHALTED_CORE_CYCLES ");
		if ((fixed_ctr & 0x4) == 0)
			__pfm_vbprintf("UNHALTED_REFERENCE_CYCLES ");
		__pfm_vbprintf("\n");

		npc++;

		if ((fixed_ctr & 0x1) == 0)
			__pfm_vbprintf("[FIXED_CTR0(pmd16)]\n");
		if ((fixed_ctr & 0x2) == 0)
			__pfm_vbprintf("[FIXED_CTR1(pmd17)]\n");
		if ((fixed_ctr & 0x4) == 0)
			__pfm_vbprintf("[FIXED_CTR2(pmd18)]\n");
	}

	/*
	 * setup core counter config
	 */
	for (i=0; i < n ; i++ ) {
		/* skip fixed counters */
		if (is_fixed_pmc(assign_pc[i]) || is_uncore(assign_pc[i]))
			continue;

		reg.val = 0; /* assume reserved bits are zeroed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;

		ne = get_nhm_entry(e[i].event);
		val = ne->pme_code;

		reg.sel_event = val & 0xff;

		umask = (val >> 8) & 0xff;

		u_flags = 0;

		/*
		 * for OFFCORE_RSP, the unit masks are all in the
		 * dedicated OFFCORE_RSP MSRs and event unit mask must be
		 * 0x1 (extracted from pme_code)
		 */
		if (!(ne->pme_flags & (PFMLIB_NHM_OFFCORE_RSP0|PFMLIB_NHM_OFFCORE_RSP1)))
			for(k=0; k < e[i].num_masks; k++) {
				int midx = pfm_nhm_midx2uidx(e[i].event, e[i].unit_masks[k]);
				umask |= ne->pme_umasks[midx].pme_ucode;
				u_flags |= ne->pme_umasks[midx].pme_uflags;
			}
		val |= umask << 8;

		reg.sel_umask  = umask;
		reg.sel_usr    = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os     = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en     = 1; /* force enable bit to 1 */
		reg.sel_int    = 1; /* force APIC int to 1 */

		reg.sel_cnt_mask = val >>24;
		reg.sel_inv = val >> 23;
		reg.sel_anythr = val >> 21;
		reg.sel_edge = val >> 18;

		if (cntrs) {
			/*
 			 * occupancy reset flag is for uncore counters only
 			 */
			if (cntrs[i].flags & PFM_NHM_SEL_OCC_RST)
				return PFMLIB_ERR_INVAL;

			if (!reg.sel_cnt_mask) {
				/*
				 * counter mask is 8-bit wide, do not silently
				 * wrap-around
				 */
				if (cntrs[i].cnt_mask > 255)
					return PFMLIB_ERR_INVAL;

				reg.sel_cnt_mask = cntrs[i].cnt_mask;
			}

			if (!reg.sel_edge)
				reg.sel_edge = cntrs[i].flags & PFM_NHM_SEL_EDGE ? 1 : 0;
			if (!reg.sel_inv)
				reg.sel_inv = cntrs[i].flags & PFM_NHM_SEL_INV ? 1 : 0;
			if (!reg.sel_anythr)
				reg.sel_anythr = cntrs[i].flags & PFM_NHM_SEL_ANYTHR ? 1 : 0;
		}

		if (u_flags || (ne->pme_flags & PFMLIB_NHM_PEBS))
				pebs_mask |= 1ULL << assign_pc[i];

		/*
		 * check for MEM_INST_RETIRED:LATENCY_ABOVE_THRESHOLD_0 to enable load latency filtering
		 * when PEBS is used. There is only one threshold possible, yet mutliple counters may be
		 * programmed with this event/umask. That means they all share the same threshold.
		 */
		if (reg.sel_event == 0xb && (umask & 0x10))
			ld_mask |= 1ULL << assign_pc[i];

		pc[npc].reg_num     = assign_pc[i];
		pc[npc].reg_value   = reg.val;
		pc[npc].reg_addr    = NHM_SEL_BASE+assign_pc[i];
		pc[npc].reg_alt_addr= NHM_SEL_BASE+assign_pc[i];

		__pfm_vbprintf("[PERFEVTSEL%u(pmc%u)=0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d anythr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
				pc[npc].reg_num,
				pc[npc].reg_num,
				reg.val,
				reg.sel_event,
				reg.sel_umask,
				reg.sel_os,
				reg.sel_usr,
				reg.sel_anythr,
				reg.sel_en,
				reg.sel_int,
				reg.sel_inv,
				reg.sel_edge,
				reg.sel_cnt_mask,
				ne->pme_name);

		__pfm_vbprintf("[PMC%u(pmd%u)]\n",
				pc[npc].reg_num,
				pc[npc].reg_num);

		npc++;
	}
	/*
	 * setup uncore fixed counter config
	 */
	if (nuf) {
		pc[npc].reg_num   = 20;
		pc[npc].reg_value = 0x5ULL; /* ena=1, PMI=dtermined by kernel */
		pc[npc].reg_addr  = 0x395;
		pc[npc].reg_alt_addr  = 0x395;
		__pfm_vbprintf("[UNC_FIXED_CTRL(pmc20)=0x%"PRIx64" pmi=1 ena=1] UNC_CLK_UNHALTED\n", pc[npc].reg_value);
		__pfm_vbprintf("[UNC_FIXED_CTR0(pmd20)]\n");
		unc_global_ctrl |= 1ULL<< 32;
		npc++;
	}
	/*
 	 * setup uncore counter config
 	 */
	for (i=0; i < n ; i++ ) {

		/* skip core counters, uncore fixed */
		if (!is_uncore(assign_pc[i]) || assign_pc[i] == 20)
			continue;

		reg.val = 0; /* assume reserved bits are zerooed */

		ne = get_nhm_entry(e[i].event);
		val = ne->pme_code;

		reg.usel_event = val & 0xff;

		umask = (val >> 8) & 0xff;

		for(k=0; k < e[i].num_masks; k++) {
			int midx = pfm_nhm_midx2uidx(e[i].event, e[i].unit_masks[k]);
			umask |= ne->pme_umasks[midx].pme_ucode;
		}

		val |= umask << 8;

		reg.usel_umask = umask;
		reg.usel_en    = 1; /* force enable bit to 1 */
		reg.usel_int   = 1; /* force APIC int to 1 */

		/*
 		 * allow hardcoded filters in event table
 		 */
		reg.usel_cnt_mask = val >>24;
		reg.usel_inv = val >> 23;
		reg.usel_edge = val >> 18;
		reg.usel_occ = val >> 17;

		if (cntrs) {
			/*
 			 * anythread if for core counters only
 			 */
			if (cntrs[i].flags & PFM_NHM_SEL_ANYTHR)
				return PFMLIB_ERR_INVAL;

			if (!reg.usel_cnt_mask) {
				/*
				 * counter mask is 8-bit wide, do not silently
				 * wrap-around
				 */
				if (cntrs[i].cnt_mask > 255)
					return PFMLIB_ERR_INVAL;

				reg.usel_cnt_mask = cntrs[i].cnt_mask;
			}
			if (!reg.usel_edge)
				reg.usel_edge = cntrs[i].flags & PFM_NHM_SEL_EDGE ? 1 : 0;
			
			if (!reg.usel_inv)
				reg.usel_inv = cntrs[i].flags & PFM_NHM_SEL_INV ? 1 : 0;

			if (!reg.usel_occ)
				reg.usel_occ = cntrs[i].flags & PFM_NHM_SEL_OCC_RST ? 1 : 0;
		}

		unc_global_ctrl |= 1ULL<< (assign_pc[i] - 21);
		pc[npc].reg_num     = assign_pc[i];
		pc[npc].reg_value   = reg.val;
		pc[npc].reg_addr    = UNC_NHM_SEL_BASE+assign_pc[i] - 21;
		pc[npc].reg_alt_addr= UNC_NHM_SEL_BASE+assign_pc[i] - 21;

		__pfm_vbprintf("[UNC_PERFEVTSEL%u(pmc%u)=0x%"PRIx64" event=0x%x umask=0x%x en=%d int=%d inv=%d edge=%d occ=%d cnt_msk=%d] %s\n",
				pc[npc].reg_num - 21,
				pc[npc].reg_num,
				reg.val,
				reg.usel_event,
				reg.usel_umask,
				reg.usel_en,
				reg.usel_int,
				reg.usel_inv,
				reg.usel_edge,
				reg.usel_occ,
				reg.usel_cnt_mask,
				ne->pme_name);

		__pfm_vbprintf("[UNC_PMC%u(pmd%u)]\n",
				pc[npc].reg_num - 21,
				pc[npc].reg_num);
		npc++;
	}

	/*
	 * setup pmds: must be in the same order as the events
	 */
	for (i=0; i < n ; i++) {
		switch (assign_pc[i]) {
			case 0 ... 4:
				pd[i].reg_num  = assign_pc[i];
				pd[i].reg_addr = NHM_CTR_BASE+assign_pc[i];
				/* index to use with RDPMC */
				pd[i].reg_alt_addr  = assign_pc[i];
				break;
			case 16 ... 18:
				/* setup pd array */
				pd[i].reg_num = assign_pc[i];
				pd[i].reg_addr = NHM_FIXED_CTR_BASE+assign_pc[i]-16;
				pd[i].reg_alt_addr = 0x40000000+assign_pc[i]-16;
				break;
			case 20:
				pd[i].reg_num = 20;
				pd[i].reg_addr = UNC_NHM_FIXED_CTR_BASE;
				pd[i].reg_alt_addr = UNC_NHM_FIXED_CTR_BASE;
				break;
			case 21 ... 28:
				pd[i].reg_num = assign_pc[i];
				pd[i].reg_addr = UNC_NHM_CTR_BASE + assign_pc[i] - 21;
				pd[i].reg_alt_addr = UNC_NHM_CTR_BASE + assign_pc[i] - 21;
				break;
		}
	}
	outp->pfp_pmd_count = i;

	/*
	 * setup PEBS_ENABLE
	 */
	if (use_pebs && pebs_mask) {
		if (!lat)
			ld_mask = 0;
		/*
		 * check that PEBS_ENABLE is available
		 */
		if (pfm_regmask_isset(r_pmcs, 17))
			return PFMLIB_ERR_NOASSIGN;

		pc[npc].reg_num   = 17;
		pc[npc].reg_value = pebs_mask | (ld_mask <<32);
		pc[npc].reg_addr  = 0x3f1; /* IA32_PEBS_ENABLE */
		pc[npc].reg_alt_addr  = 0x3f1; /* IA32_PEBS_ENABLE */

		__pfm_vbprintf("[PEBS_ENABLE(pmc%u)=0x%"PRIx64" ena0=%d ena1=%d ena2=%d ena3=%d ll0=%d ll1=%d ll2=%d ll3=%d]\n",
				pc[npc].reg_num,
				pc[npc].reg_value,
				pc[npc].reg_value & 0x1,
				(pc[npc].reg_value >> 1)  & 0x1,
				(pc[npc].reg_value >> 2)  & 0x1,
				(pc[npc].reg_value >> 3)  & 0x1,
				(pc[npc].reg_value >> 32)  & 0x1,
				(pc[npc].reg_value >> 33)  & 0x1,
				(pc[npc].reg_value >> 34)  & 0x1,
				(pc[npc].reg_value >> 35)  & 0x1);

		npc++;

		if (ld_mask) {
			if (lat < 3 || lat > 0xffff) {
				DPRINT("invalid load latency threshold %u (must be in [3:0xffff])\n", lat);
				return PFMLIB_ERR_INVAL;
			}

			if (pfm_regmask_isset(r_pmcs, 18))
				return PFMLIB_ERR_NOASSIGN;

			pc[npc].reg_num   = 18;
			pc[npc].reg_value = lat;
			pc[npc].reg_addr  = 0x3f1; /* IA32_PEBS_ENABLE */
			pc[npc].reg_alt_addr  = 0x3f1; /* IA32_PEBS_ENABLE */
			__pfm_vbprintf("[LOAD_LATENCY_THRESHOLD(pmc%u)=0x%"PRIx64"]\n",
					pc[npc].reg_num,
					pc[npc].reg_value);

			npc++;
		}
	}

	/*
 	 * setup OFFCORE_RSP0
 	 */
	if (offcore_rsp0_value) {
		pc[npc].reg_num     = 19;
		pc[npc].reg_value   = offcore_rsp0_value;
		pc[npc].reg_addr    = 0x1a6;
		pc[npc].reg_alt_addr = 0x1a6;
		__pfm_vbprintf("[OFFCORE_RSP0(pmc%u)=0x%"PRIx64"]\n",
				pc[npc].reg_num,
				pc[npc].reg_value);
		npc++;
	}
	/*
 	 * setup OFFCORE_RSP1
 	 */
	if (offcore_rsp1_value) {
		pc[npc].reg_num     = 31;
		pc[npc].reg_value   = offcore_rsp1_value;
		pc[npc].reg_addr    = 0x1a7;
		pc[npc].reg_alt_addr = 0x1a7;
		__pfm_vbprintf("[OFFCORE_RSP1(pmc%u)=0x%"PRIx64"]\n",
				pc[npc].reg_num,
				pc[npc].reg_value);
		npc++;
	}

	outp->pfp_pmc_count = npc;

	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_dispatch_lbr(pfmlib_input_param_t *inp, pfmlib_nhm_input_param_t *param, pfmlib_output_param_t *outp)
{
	static int lbr_plm_map[4]={
		0x3,	/* PLM0=0 PLM3=0 neq0=1 eq0=1 */
		0x1,	/* PLM0=0 PLM3=1 neq0=0 eq0=1 */
		0x2,	/* PLM0=1 PLM3=0 neq0=1 eq0=0 */
		0x0	/* PLM0=1 PLM3=1 neq0=0 eq0=0 */
	};
	pfm_nhm_sel_reg_t reg;
	unsigned int filter, i, c;
	unsigned int plm;

	/*
	 * check LBR_SELECT is available
	 */
	if (pfm_regmask_isset(&inp->pfp_unavail_pmcs, 30))
		return PFMLIB_ERR_NOASSIGN;

	reg.val = 0; /* capture everything */

	plm = param->pfp_nhm_lbr.lbr_plm;
	if (!plm)
		plm = inp->pfp_dfl_plm;

	/*
 	 * LBR does not distinguish PLM1, PLM2 from PLM3
 	 */

	i  = plm & PFM_PLM0 ? 0x2 : 0;
	i |= plm & PFM_PLM3 ? 0x1 : 0;

	if (lbr_plm_map[i] & 0x1)
		reg.lbr_select.cpl_eq0 = 1;

	if (lbr_plm_map[i] & 0x2)
		reg.lbr_select.cpl_neq0 = 1;

	filter = param->pfp_nhm_lbr.lbr_filter;

	if (filter & PFM_NHM_LBR_JCC)
		reg.lbr_select.jcc = 1;

	if (filter & PFM_NHM_LBR_NEAR_REL_CALL)
		reg.lbr_select.near_rel_call = 1;

	if (filter & PFM_NHM_LBR_NEAR_IND_CALL)
		reg.lbr_select.near_ind_call = 1;

	if (filter & PFM_NHM_LBR_NEAR_RET)
		reg.lbr_select.near_ret = 1;

	if (filter & PFM_NHM_LBR_NEAR_IND_JMP)
		reg.lbr_select.near_ind_jmp = 1;

	if (filter & PFM_NHM_LBR_NEAR_REL_JMP)
		reg.lbr_select.near_rel_jmp = 1;

	if (filter & PFM_NHM_LBR_FAR_BRANCH)
		reg.lbr_select.far_branch = 1;

	__pfm_vbprintf("[LBR_SELECT(PMC30)=0x%"PRIx64" eq0=%d neq0=%d jcc=%d rel=%d ind=%d ret=%d ind_jmp=%d rel_jmp=%d far=%d ]\n",
			reg.val,
			reg.lbr_select.cpl_eq0,
			reg.lbr_select.cpl_neq0,
			reg.lbr_select.jcc,
			reg.lbr_select.near_rel_call,
			reg.lbr_select.near_ind_call,
			reg.lbr_select.near_ret,
			reg.lbr_select.near_ind_jmp,
			reg.lbr_select.near_rel_jmp,
			reg.lbr_select.far_branch);

	__pfm_vbprintf("[LBR_TOS(PMD31)]\n");

	__pfm_vbprintf("[LBR_FROM-LBR_TO(PMD32..PMD63)]\n");

	c = outp->pfp_pmc_count;

	outp->pfp_pmcs[c].reg_num = 30;
	outp->pfp_pmcs[c].reg_value = reg.val;
	outp->pfp_pmcs[c].reg_addr = 0x1c8;
	outp->pfp_pmcs[c].reg_alt_addr = 0x1c8;
	c++;
	outp->pfp_pmc_count = c;

	c = outp->pfp_pmd_count;

	outp->pfp_pmds[c].reg_num = 31;
	outp->pfp_pmds[c].reg_value = 0;
	outp->pfp_pmds[c].reg_addr = 0x1c9;
	outp->pfp_pmds[c].reg_alt_addr = 0x1c9;
	c++;

	for(i=0; i < 32; i++, c++) {
		outp->pfp_pmds[c].reg_num = 32 + i;
		outp->pfp_pmds[c].reg_value = 0;
		outp->pfp_pmds[c].reg_addr = (i>>1) + ((i & 0x1) ? 0x6c0 : 0x680);
		outp->pfp_pmds[c].reg_alt_addr = (i>>1) + ((i & 0x1) ? 0x6c0 : 0x680);
	}
	outp->pfp_pmd_count = c;
	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_nhm_input_param_t *mod_in  = (pfmlib_nhm_input_param_t *)model_in;
	int ret;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT("invalid plm=%x\n", inp->pfp_dfl_plm);
		return PFMLIB_ERR_INVAL;
	}
	ret = pfm_nhm_dispatch_counters(inp, mod_in, outp);
	if (ret != PFMLIB_SUCCESS)
		return ret;

	if (mod_in && mod_in->pfp_nhm_lbr.lbr_used)
		ret = pfm_nhm_dispatch_lbr(inp, mod_in, outp);

	return ret;
}

static int
pfm_nhm_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	pfmlib_regmask_t cnts;

	pfm_get_impl_counters(&cnts);

	if (cnt != PFMLIB_CNT_FIRST
	    && (cnt > MAX_COUNTERS ||
		!pfm_regmask_isset(&cnts, cnt)))
		return PFMLIB_ERR_INVAL;

	*code = get_nhm_entry(i)->pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_nhm_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	pme_nhm_entry_t *ne;
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	ne = get_nhm_entry(j);

	if (ne->pme_flags & PFMLIB_NHM_UNC_FIXED) {
		pfm_regmask_set(counters, 20);
		return;
	}

	if (ne->pme_flags & PFMLIB_NHM_UNC) {
		pfm_regmask_set(counters, 20);
		pfm_regmask_set(counters, 21);
		pfm_regmask_set(counters, 22);
		pfm_regmask_set(counters, 23);
		pfm_regmask_set(counters, 24);
		pfm_regmask_set(counters, 25);
		pfm_regmask_set(counters, 26);
		pfm_regmask_set(counters, 27);
		return;
	}
	/*
	 * fixed counter events have no unit mask
	 */
	if (ne->pme_flags & PFMLIB_NHM_FIXED0)
		pfm_regmask_set(counters, 16);

	if (ne->pme_flags & PFMLIB_NHM_FIXED1)
		pfm_regmask_set(counters, 17);

	if (ne->pme_flags & PFMLIB_NHM_FIXED2_ONLY)
		pfm_regmask_set(counters, 18);

	/*
	 * extract from unit mask level
	 */
	for (i=0; i < ne->pme_numasks; i++) {
		if (ne->pme_umasks[i].pme_uflags & PFMLIB_NHM_FIXED0)
			pfm_regmask_set(counters, 16);
		if (ne->pme_umasks[i].pme_uflags & PFMLIB_NHM_FIXED1)
			pfm_regmask_set(counters, 17);
		if (ne->pme_umasks[i].pme_uflags & PFMLIB_NHM_FIXED2_ONLY)
			pfm_regmask_set(counters, 18);
	}

	/*
	 * event on FIXED_CTR2 is exclusive CPU_CLK_UNHALTED:REF
	 * PMC0|PMC1 only on 0,1, constraint at event-level
	 */
	if (!pfm_regmask_isset(counters, 18)) {
		pfm_regmask_set(counters, 0);
		if (!(ne->pme_flags & PFMLIB_NHM_PMC0))
			pfm_regmask_set(counters, 1);
		if (!(ne->pme_flags & (PFMLIB_NHM_PMC01|PFMLIB_NHM_PMC0))) {
			pfm_regmask_set(counters, 2);
			pfm_regmask_set(counters, 3);
		}
	}
}

static void
pfm_nhm_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	*impl_pmcs = nhm_impl_pmcs;
}

static void
pfm_nhm_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	*impl_pmds = nhm_impl_pmds;
}

static void
pfm_nhm_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	/* core generic */
	pfm_regmask_set(impl_counters, 0);
	pfm_regmask_set(impl_counters, 1);
	pfm_regmask_set(impl_counters, 2);
	pfm_regmask_set(impl_counters, 3);
	/* core fixed */
	pfm_regmask_set(impl_counters, 16);
	pfm_regmask_set(impl_counters, 17);
	pfm_regmask_set(impl_counters, 18);

	/* uncore pmd registers all counters */
	pfm_regmask_or(impl_counters, impl_counters, &nhm_impl_unc_pmds);
}

/*
 * Even though, CPUID 0xa returns in eax the actual counter
 * width, the architecture specifies that writes are limited
 * to lower 32-bits. As such, only the lower 32-bit have full
 * degree of freedom. That is the "useable" counter width.
 */
#define PMU_NHM_COUNTER_WIDTH       32

static void
pfm_nhm_get_hw_counter_width(unsigned int *width)
{
	/*
	 * Even though, CPUID 0xa returns in eax the actual counter
	 * width, the architecture specifies that writes are limited
	 * to lower 32-bits. As such, only the lower 31 bits have full
	 * degree of freedom. That is the "useable" counter width.
	 */
	*width = PMU_NHM_COUNTER_WIDTH;
}

static char *
pfm_nhm_get_event_name(unsigned int i)
{
	return get_nhm_entry(i)->pme_name;
}

static int
pfm_nhm_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = get_nhm_entry(ev)->pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}
static char *
pfm_nhm_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	midx = pfm_nhm_midx2uidx(ev, midx);
	return get_nhm_entry(ev)->pme_umasks[midx].pme_uname;
}

static int
pfm_nhm_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	midx = pfm_nhm_midx2uidx(ev, midx);
	s = get_nhm_entry(ev)->pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_nhm_get_num_event_masks(unsigned int ev)
{
	int i, num = 0;
	pme_nhm_entry_t *ne;
	int model;

	ne = get_nhm_entry(ev);

	for (i=0; i < ne->pme_numasks; i++) {
		model = ne->pme_umasks[i].pme_umodel;
		if (!model || model == cpu_model)
			num++;
	}
DPRINT("event %s numasks=%d\n", ne->pme_name, num);
	return num;
}

static int
pfm_nhm_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	midx = pfm_nhm_midx2uidx(ev, midx);
	*code =get_nhm_entry(ev)->pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_get_cycle_event(pfmlib_event_t *e)
{
	e->event = pme_cycles;
	return PFMLIB_SUCCESS;
}

static int
pfm_nhm_get_inst_retired(pfmlib_event_t *e)
{
	e->event = pme_instr;;
	return PFMLIB_SUCCESS;
}

/*
 * the following function implement the model
 * specific API directly available to user
 */

/*
 *  Check if event and all provided unit masks support PEBS
 *
 *  return:
 *  	PFMLIB_ERR_INVAL: invalid event e
 *  	1 event supports PEBS
 *  	0 event does not support PEBS
 *
 */
int
pfm_nhm_is_pebs(pfmlib_event_t *e)
{
	pme_nhm_entry_t *ne;
	unsigned int i, n=0;

	if (e == NULL || e->event >= intel_nhm_support.pme_count)
		return PFMLIB_ERR_INVAL;

	ne = get_nhm_entry(e->event);
	if (ne->pme_flags & PFMLIB_NHM_PEBS)
		return 1;

	/*
	 * ALL unit mask must support PEBS for this test to return true
	 */
	for(i=0; i < e->num_masks; i++) {
		int midx;
		/* check for valid unit mask */
		if (e->unit_masks[i] >= ne->pme_numasks)
			return PFMLIB_ERR_INVAL;
 		midx = pfm_nhm_midx2uidx(e->event, e->unit_masks[i]);
		if (ne->pme_umasks[midx].pme_uflags & PFMLIB_NHM_PEBS)
			n++;
	}
	return n > 0 && n == e->num_masks;
}

/*
 * Check if event is uncore
 * return:
 *  	PFMLIB_ERR_INVAL: invalid event e
 *  	1 event is uncore
 *  	0 event is not uncore
 */
int
pfm_nhm_is_uncore(pfmlib_event_t *e)
{
	if (PFMLIB_INITIALIZED() == 0)
		return 0;

	if (e == NULL || e->event >= num_pe)
		return PFMLIB_ERR_INVAL;

	return !!(get_nhm_entry(e->event)->pme_flags & (PFMLIB_NHM_UNC|PFMLIB_NHM_UNC_FIXED));
}

static const char *data_src_encodings[]={
/*  0 */	"unknown L3 cache miss",
/*  1 */	"minimal latency core cache hit. Request was satisfied by L1 data cache",
/*  2 */	"pending core cache HIT. Outstanding core cache miss to same cacheline address already underway",
/*  3 */	"data request satisfied by the L2",
/*  4 */	"L3 HIT. Local or remote home request that hit L3 in the uncore with no coherency actions required (snooping)",
/*  5 */	"L3 HIT. Local or remote home request that hit L3 and was serviced by another core with a cross core snoop where no modified copy was found (clean)",
/*  6 */	"L3 HIT. Local or remote home request that hit L3 and was serviced by another core with a cross core snoop where modified copies were found (HITM)",
/*  7 */	"reserved",
/*  8 */	"L3 MISS. Local homed request that missed L3 and was serviced by forwarded data following a cross package snoop where no modified copy was found (remote home requests are not counted)",
/*  9 */	"reserved",
/* 10 */	"L3 MISS. Local homed request that missed L3 and was serviced by local DRAM (go to shared state)",
/* 11 */	"L3 MISS. Remote homed request that missed L3 and was serviced by remote DRAM (go to shared state)",
/* 12 */	"L3 MISS. Local homed request that missed L3 and was serviced by local DRAM (go to exclusive state)",
/* 13 */	"L3 MISS. Remote homed request that missed L3 and was serviced by remote DRAM (go to exclusive state)",
/* 14 */	"reserved",
/* 15 */	"request to uncacheable memory"
};

/*
 * return data source encoding based on index in val
 * To be used with PEBS load latency filtering to decode
 * source of the load miss
 */
int pfm_nhm_data_src_desc(unsigned int val, char **desc)
{
	if (val > 15 || !desc)
		return PFMLIB_ERR_INVAL;

	*desc = strdup(data_src_encodings[val]);
	if (!*desc)
		return PFMLIB_ERR_NOMEM;

	return PFMLIB_SUCCESS;	
}

pfm_pmu_support_t intel_nhm_support={
	.pmu_name		= "Intel Nehalem",
	.pmu_type		= PFMLIB_INTEL_NHM_PMU,
	.pme_count		= 0,/* patched at runtime */
	.pmc_count		= 0,/* patched at runtime */
	.pmd_count		= 0,/* patched at runtime */
	.num_cnt		= 0,/* patched at runtime */
	.get_event_code		= pfm_nhm_get_event_code,
	.get_event_name		= pfm_nhm_get_event_name,
	.get_event_counters	= pfm_nhm_get_event_counters,
	.dispatch_events	= pfm_nhm_dispatch_events,
	.pmu_detect		= pfm_nhm_detect,
	.pmu_init		= pfm_nhm_init,
	.get_impl_pmcs		= pfm_nhm_get_impl_pmcs,
	.get_impl_pmds		= pfm_nhm_get_impl_pmds,
	.get_impl_counters	= pfm_nhm_get_impl_counters,
	.get_hw_counter_width	= pfm_nhm_get_hw_counter_width,
	.get_event_desc         = pfm_nhm_get_event_description,
	.get_num_event_masks	= pfm_nhm_get_num_event_masks,
	.get_event_mask_name	= pfm_nhm_get_event_mask_name,
	.get_event_mask_code	= pfm_nhm_get_event_mask_code,
	.get_event_mask_desc	= pfm_nhm_get_event_mask_desc,
	.get_cycle_event	= pfm_nhm_get_cycle_event,
	.get_inst_retired_event = pfm_nhm_get_inst_retired
};

pfm_pmu_support_t intel_wsm_support={
	.pmu_name		= "Intel Westmere",
	.pmu_type		= PFMLIB_INTEL_WSM_PMU,
	.pme_count		= 0,/* patched at runtime */
	.pmc_count		= 0,/* patched at runtime */
	.pmd_count		= 0,/* patched at runtime */
	.num_cnt		= 0,/* patched at runtime */
	.get_event_code		= pfm_nhm_get_event_code,
	.get_event_name		= pfm_nhm_get_event_name,
	.get_event_counters	= pfm_nhm_get_event_counters,
	.dispatch_events	= pfm_nhm_dispatch_events,
	.pmu_detect		= pfm_wsm_detect,
	.pmu_init		= pfm_nhm_init,
	.get_impl_pmcs		= pfm_nhm_get_impl_pmcs,
	.get_impl_pmds		= pfm_nhm_get_impl_pmds,
	.get_impl_counters	= pfm_nhm_get_impl_counters,
	.get_hw_counter_width	= pfm_nhm_get_hw_counter_width,
	.get_event_desc         = pfm_nhm_get_event_description,
	.get_num_event_masks	= pfm_nhm_get_num_event_masks,
	.get_event_mask_name	= pfm_nhm_get_event_mask_name,
	.get_event_mask_code	= pfm_nhm_get_event_mask_code,
	.get_event_mask_desc	= pfm_nhm_get_event_mask_desc,
	.get_cycle_event	= pfm_nhm_get_cycle_event,
	.get_inst_retired_event = pfm_nhm_get_inst_retired
};
