/*
 * pfmlib_coreduo.c : Intel Core Duo/Solo
 *
 * Copyright (c) 2009 Google, Inc
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
 *
 * This file implements support for Intel Core Duo/Solor PMU as specified in the
 * following document:
 * 	"IA-32 Intel Architecture Software Developer's Manual - Volume 3B: System
 * 	Programming Guide"
 *
 * Core Dup/Solo PMU = architectural perfmon v1 + model specific events
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_coreduo.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_coreduo_priv.h"

#include "coreduo_events.h"

/* let's define some handy shortcuts! */
#define sel_event_select perfevtsel.sel_event_select
#define sel_unit_mask	 perfevtsel.sel_unit_mask
#define sel_usr		 perfevtsel.sel_usr
#define sel_os		 perfevtsel.sel_os
#define sel_edge	 perfevtsel.sel_edge
#define sel_pc		 perfevtsel.sel_pc
#define sel_int		 perfevtsel.sel_int
#define sel_en		 perfevtsel.sel_en
#define sel_inv		 perfevtsel.sel_inv
#define sel_cnt_mask	 perfevtsel.sel_cnt_mask

/*
 * Description of the PMC register mappings:
 *
 * 0  -> PMC0  -> PERFEVTSEL0
 * 1  -> PMC1  -> PERFEVTSEL1 
 * 16 -> PMC16 -> FIXED_CTR_CTRL
 * 17 -> PMC17 -> PEBS_ENABLED
 *
 * Description of the PMD register mapping:
 *
 * 0  -> PMD0 -> PMC0
 * 1  -> PMD1 -> PMC1
 * 16 -> PMD2 -> FIXED_CTR0
 * 17 -> PMD3 -> FIXED_CTR1
 * 18 -> PMD4 -> FIXED_CTR2
 */
#define COREDUO_SEL_BASE		0x186
#define COREDUO_CTR_BASE		0xc1

#define PFMLIB_COREDUO_ALL_FLAGS \
	(PFM_COREDUO_SEL_INV|PFM_COREDUO_SEL_EDGE)

static pfmlib_regmask_t coreduo_impl_pmcs, coreduo_impl_pmds;
static int highest_counter;

static int
pfm_coreduo_detect(void)
{
	char buffer[128];
	int family, model;
	int ret;

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

	model = atoi(buffer);

	return family == 6 && model == 14 ? PFMLIB_SUCCESS : PFMLIB_ERR_NOTSUPP;
}

static int
pfm_coreduo_init(void)
{
	pfm_regmask_set(&coreduo_impl_pmcs, 0);
	pfm_regmask_set(&coreduo_impl_pmcs, 1);

	pfm_regmask_set(&coreduo_impl_pmds, 0);
	pfm_regmask_set(&coreduo_impl_pmds, 1);

	highest_counter = 1;

	return PFMLIB_SUCCESS;
}

/*
 * IMPORTANT: the interface guarantees that pfp_pmds[] elements are returned in the order the events
 *	      were submitted.
 */
static int
pfm_coreduo_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_coreduo_input_param_t *param, pfmlib_output_param_t *outp)
{
#define HAS_OPTIONS(x)	(cntrs && (cntrs[x].flags || cntrs[x].cnt_mask))

	pfm_coreduo_counter_t *cntrs;
	pfm_coreduo_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	uint64_t val;
	unsigned long plm;
	unsigned int npc, npmc0, npmc1, nf2;
	unsigned int i, n, k, ucode;
	unsigned int assign_pc[PMU_COREDUO_NUM_COUNTERS];
	unsigned int next_gen, last_gen;

	npc = npmc0 = npmc1 = nf2 = 0;

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	n      = inp->pfp_event_count;

	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_coreduo_counters : NULL;

	if (n > PMU_COREDUO_NUM_COUNTERS)
		return PFMLIB_ERR_TOOMANY;

	/*
	 * initilize to empty
	 */
	for(i=0; i < PMU_COREDUO_NUM_COUNTERS; i++)
		assign_pc[i] = -1;

	/*
	 * error checking
	 */
	for(i=0; i < n; i++) {
		/*
		 * only supports two priv levels for perf counters
		 */
		if (e[i].plm & (PFM_PLM1|PFM_PLM2))
			return PFMLIB_ERR_INVAL;

		/*
		 * check for valid flags
		 */
		if (cntrs && cntrs[i].flags & ~PFMLIB_COREDUO_ALL_FLAGS)
			return PFMLIB_ERR_INVAL;

		/*
		 * check event-level single register constraint (PMC0, PMC1, FIXED_CTR2)
		 * fail if more than two events requested for the same counter
		 */
		if (coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_PMC0) {
			if (++npmc0 > 1) {
				DPRINT("two events compete for a PMC0\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
		 * check if PMC1 is available and if only one event is dependent on it
		 */
		if (coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_PMC1) {
			if (++npmc1 > 1) {
				DPRINT("two events compete for a PMC1\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
	}

	next_gen = 0; /* first generic counter */
	last_gen = 1; /* last generic counter */

	/*
	 * strongest constraint first: works only in IA32_PMC0, IA32_PMC1
	 */
	for(i=0; i < n; i++) {
		if ((coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_PMC0)) {
			if (pfm_regmask_isset(r_pmcs, 0))
				return PFMLIB_ERR_NOASSIGN;

			assign_pc[i] = 0;

			next_gen++;
		}
		if (coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_PMC1) {
			if (pfm_regmask_isset(r_pmcs, 1))
				return PFMLIB_ERR_NOASSIGN;

			assign_pc[i] = 1;

			next_gen = (next_gen+1) % PMU_COREDUO_NUM_COUNTERS;
		}
	}
	/*
	 * assign what is left
	 */
	for(i=0; i < n; i++) {
		if (assign_pc[i] == -1) {
			for(; next_gen <= last_gen; next_gen++) {
DPRINT("i=%d next_gen=%d last=%d isset=%d\n", i, next_gen, last_gen, pfm_regmask_isset(r_pmcs, next_gen));
				if (!pfm_regmask_isset(r_pmcs, next_gen))
					break;
			}
			if (next_gen <= last_gen)
				assign_pc[i] = next_gen++;
			else {
				DPRINT("cannot assign generic counters\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
	}

	for (i=0; i < n ; i++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;

		val = coreduo_pe[e[i].event].pme_code;

		reg.sel_event_select = val  & 0xff;

		ucode = (val >> 8) & 0xff;

		for(k=0; k < e[i].num_masks; k++) {
			ucode |= coreduo_pe[e[i].event].pme_umasks[e[i].unit_masks[k]].pme_ucode;
		}

		/*
		 * for events supporting Core specificity (self, both), a value
		 * of 0 for bits 15:14 (7:6 in our umask) is reserved, therefore we
		 * force to SELF if user did not specify anything
		 */
		if ((coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_CSPEC)
		    && ((ucode & (0x3 << 6)) == 0)) {
			ucode |= 1 << 6;
		}
		/*
		 * for events supporting MESI, a value
		 * of 0 for bits 11:8 (0-3 in our umask) means nothing will be
		 * counted. Therefore, we force a default of 0xf (M,E,S,I).
		 */
		if ((coreduo_pe[e[i].event].pme_flags & PFMLIB_COREDUO_MESI)
		    && ((ucode & 0xf) == 0)) {
			ucode |= 0xf;
		}

		val |= ucode << 8;

		reg.sel_unit_mask  = ucode;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */

		reg.sel_cnt_mask = val >>24;
		reg.sel_inv = val >> 23;
		reg.sel_edge = val >> 18;

		if (cntrs) {
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
				reg.sel_edge = cntrs[i].flags & PFM_COREDUO_SEL_EDGE ? 1 : 0;
			if (!reg.sel_inv)
				reg.sel_inv = cntrs[i].flags & PFM_COREDUO_SEL_INV ? 1 : 0;
		}

		pc[npc].reg_num     = assign_pc[i];
		pc[npc].reg_value   = reg.val;
		pc[npc].reg_addr    = COREDUO_SEL_BASE+assign_pc[i];
		pc[npc].reg_alt_addr= COREDUO_SEL_BASE+assign_pc[i];

		__pfm_vbprintf("[PERFEVTSEL%u(pmc%u)=0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
				pc[npc].reg_num,
				pc[npc].reg_num,
				reg.val,
				reg.sel_event_select,
				reg.sel_unit_mask,
				reg.sel_os,
				reg.sel_usr,
				reg.sel_en,
				reg.sel_int,
				reg.sel_inv,
				reg.sel_edge,
				reg.sel_cnt_mask,
				coreduo_pe[e[i].event].pme_name);

		__pfm_vbprintf("[PMC%u(pmd%u)]\n",
				pc[npc].reg_num,
				pc[npc].reg_num);

		npc++;
	}
	/*
	 * setup pmds: must be in the same order as the events
	 */
	for (i=0; i < n ; i++) {
		pd[i].reg_num  = assign_pc[i];
		pd[i].reg_addr = COREDUO_CTR_BASE+assign_pc[i];
		/* index to use with RDPMC */
		pd[i].reg_alt_addr  = assign_pc[i];
	}
	outp->pfp_pmd_count = i;
	outp->pfp_pmc_count = npc;

	return PFMLIB_SUCCESS;
}
static int
pfm_coreduo_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_coreduo_input_param_t *mod_in  = (pfmlib_coreduo_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT("invalid plm=%x\n", inp->pfp_dfl_plm);
		return PFMLIB_ERR_INVAL;
	}
	return pfm_coreduo_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_coreduo_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST
	    && (cnt > highest_counter ||
		!pfm_regmask_isset(&coreduo_impl_pmds, cnt)))
		return PFMLIB_ERR_INVAL;

	*code = coreduo_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_coreduo_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	memset(counters, 0, sizeof(*counters));

	pfm_regmask_set(counters, 0);
	pfm_regmask_set(counters, 1);

	if (coreduo_pe[j].pme_flags & PFMLIB_COREDUO_PMC0)
		pfm_regmask_clr(counters, 1);
	if (coreduo_pe[j].pme_flags & PFMLIB_COREDUO_PMC1)
		pfm_regmask_clr(counters, 0);
}

static void
pfm_coreduo_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	*impl_pmcs = coreduo_impl_pmcs;
}

static void
pfm_coreduo_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	*impl_pmds = coreduo_impl_pmds;
}

static void
pfm_coreduo_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	/* all pmds are counters */
	*impl_counters = coreduo_impl_pmds;
}

/*
 * Even though, CPUID 0xa returns in eax the actual counter
 * width, the architecture specifies that writes are limited
 * to lower 32-bits. As such, only the lower 32-bit have full
 * degree of freedom. That is the "useable" counter width.
 */
static void
pfm_coreduo_get_hw_counter_width(unsigned int *width)
{
	/*
	 * Even though, CPUID 0xa returns in eax the actual counter
	 * width, the architecture specifies that writes are limited
	 * to lower 32-bits. As such, only the lower 31 bits have full
	 * degree of freedom. That is the "useable" counter width.
	 */
	*width = 32;
}

static char *
pfm_coreduo_get_event_name(unsigned int i)
{
	return coreduo_pe[i].pme_name;
}

static int
pfm_coreduo_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = coreduo_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static char *
pfm_coreduo_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return coreduo_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_coreduo_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = coreduo_pe[ev].pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_coreduo_get_num_event_masks(unsigned int ev)
{
	return coreduo_pe[ev].pme_numasks;
}

static int
pfm_coreduo_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code = coreduo_pe[ev].pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_coreduo_get_cycle_event(pfmlib_event_t *e)
{
	e->event = PME_COREDUO_UNHALTED_CORE_CYCLES;
	return PFMLIB_SUCCESS;
}

static int
pfm_coreduo_get_inst_retired(pfmlib_event_t *e)
{
	e->event = PME_COREDUO_INSTRUCTIONS_RETIRED;
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t coreduo_support={
	.pmu_name		= "Intel Core Duo/Solo",
	.pmu_type		= PFMLIB_COREDUO_PMU,
	.pme_count		= PME_COREDUO_EVENT_COUNT,
	.pmc_count		= 2,
	.pmd_count		= 2,
	.num_cnt		= 2,
	.get_event_code		= pfm_coreduo_get_event_code,
	.get_event_name		= pfm_coreduo_get_event_name,
	.get_event_counters	= pfm_coreduo_get_event_counters,
	.dispatch_events	= pfm_coreduo_dispatch_events,
	.pmu_detect		= pfm_coreduo_detect,
	.pmu_init		= pfm_coreduo_init,
	.get_impl_pmcs		= pfm_coreduo_get_impl_pmcs,
	.get_impl_pmds		= pfm_coreduo_get_impl_pmds,
	.get_impl_counters	= pfm_coreduo_get_impl_counters,
	.get_hw_counter_width	= pfm_coreduo_get_hw_counter_width,
	.get_event_desc         = pfm_coreduo_get_event_description,
	.get_num_event_masks	= pfm_coreduo_get_num_event_masks,
	.get_event_mask_name	= pfm_coreduo_get_event_mask_name,
	.get_event_mask_code	= pfm_coreduo_get_event_mask_code,
	.get_event_mask_desc	= pfm_coreduo_get_event_mask_desc,
	.get_cycle_event	= pfm_coreduo_get_cycle_event,
	.get_inst_retired_event = pfm_coreduo_get_inst_retired
};
