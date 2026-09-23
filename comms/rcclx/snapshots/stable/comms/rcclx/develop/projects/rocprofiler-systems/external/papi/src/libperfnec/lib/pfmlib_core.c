/*
 * pfmlib_core.c : Intel Core PMU
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
 *
 * This file implements support for Intel Core PMU as specified in the following document:
 * 	"IA-32 Intel Architecture Software Developer's Manual - Volume 3B: System
 * 	Programming Guide"
 *
 * Core PMU = architectural perfmon v2 + PEBS
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_core.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_core_priv.h"

#include "core_events.h"

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

#define is_pebs(i)	(core_pe[i].pme_flags & PFMLIB_CORE_PEBS)


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
#define CORE_SEL_BASE		0x186
#define CORE_CTR_BASE		0xc1
#define FIXED_CTR_BASE		0x309

#define PFMLIB_CORE_ALL_FLAGS \
	(PFM_CORE_SEL_INV|PFM_CORE_SEL_EDGE)

static pfmlib_regmask_t core_impl_pmcs, core_impl_pmds;
static int highest_counter;

static int
pfm_core_detect(void)
{
	int ret;
	int family, model;
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

	if (family != 6)
		return PFMLIB_ERR_NOTSUPP;

	model = atoi(buffer);
	switch(model) {
		case 15: /* Merom */
		case 23: /* Penryn */
		case 29: /* Dunnington */
			  break;
		default:
			return PFMLIB_ERR_NOTSUPP;
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_core_init(void)
{
	int i;

	pfm_regmask_set(&core_impl_pmcs, 0);
	pfm_regmask_set(&core_impl_pmcs, 1);
	pfm_regmask_set(&core_impl_pmcs, 16);
	pfm_regmask_set(&core_impl_pmcs, 17);

	pfm_regmask_set(&core_impl_pmds, 0);
	pfm_regmask_set(&core_impl_pmds, 1);
	pfm_regmask_set(&core_impl_pmds, 16);
	pfm_regmask_set(&core_impl_pmds, 17);
	pfm_regmask_set(&core_impl_pmds, 18);

	/* lbr */
	pfm_regmask_set(&core_impl_pmds, 19);
	for(i=0; i < 8; i++)
		pfm_regmask_set(&core_impl_pmds, i);

	highest_counter = 18;

	return PFMLIB_SUCCESS;
}

static int
pfm_core_is_fixed(pfmlib_event_t *e, unsigned int f)
{
	unsigned int fl, flc, i;
	unsigned int mask = 0;

	fl = core_pe[e->event].pme_flags;

	/*
	 * first pass: check if event as a whole supports fixed counters
	 */
	switch(f) {
		case 0:
			mask = PFMLIB_CORE_FIXED0;
			break;
		case 1:
			mask = PFMLIB_CORE_FIXED1;
			break;
		case 2:
			mask = PFMLIB_CORE_FIXED2_ONLY;
			break;
		default:
			return 0;
	}
	if (fl & mask)
		return 1;
	/*
	 * second pass: check if unit mask support fixed counter
	 *
	 * reject if mask not found OR if not all unit masks have
	 * same fixed counter mask
	 */
	flc = 0;
	for(i=0; i < e->num_masks; i++) {
		fl = core_pe[e->event].pme_umasks[e->unit_masks[i]].pme_flags;
		if (fl & mask)
			flc++;
	}
	return flc > 0 && flc == e->num_masks ? 1 : 0;
}

/*
 * IMPORTANT: the interface guarantees that pfp_pmds[] elements are returned in the order the events
 *	      were submitted.
 */
static int
pfm_core_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_core_input_param_t *param, pfmlib_output_param_t *outp)
{
#define HAS_OPTIONS(x)	(cntrs && (cntrs[x].flags || cntrs[x].cnt_mask))
#define is_fixed_pmc(a) (a == 16 || a == 17 || a == 18)

	pfmlib_core_counter_t *cntrs;
	pfm_core_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	uint64_t val;
	unsigned long plm;
	unsigned long long fixed_ctr;
	unsigned int npc, npmc0, npmc1, nf2;
	unsigned int i, j, n, k, ucode, use_pebs = 0, done_pebs;
	unsigned int assign_pc[PMU_CORE_NUM_COUNTERS];
	unsigned int next_gen, last_gen;

	npc = npmc0 = npmc1 = nf2 = 0;

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	n      = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_core_counters : NULL;
	use_pebs = param ? param->pfp_core_pebs.pebs_used : 0;

	if (n > PMU_CORE_NUM_COUNTERS)
		return PFMLIB_ERR_TOOMANY;

	/*
	 * initilize to empty
	 */
	for(i=0; i < PMU_CORE_NUM_COUNTERS; i++)
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
		if (cntrs && cntrs[i].flags & ~PFMLIB_CORE_ALL_FLAGS)
			return PFMLIB_ERR_INVAL;

		if (core_pe[e[i].event].pme_flags & PFMLIB_CORE_UMASK_NCOMBO
		    && e[i].num_masks > 1) {
			DPRINT("events does not support unit mask combination\n");
				return PFMLIB_ERR_NOASSIGN;
		}

		/*
		 * check event-level single register constraint (PMC0, PMC1, FIXED_CTR2)
		 * fail if more than two events requested for the same counter
		 */
		if (core_pe[e[i].event].pme_flags & PFMLIB_CORE_PMC0) {
			if (++npmc0 > 1) {
				DPRINT("two events compete for a PMC0\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
		 * check if PMC1 is available and if only one event is dependent on it
		 */
		if (core_pe[e[i].event].pme_flags & PFMLIB_CORE_PMC1) {
			if (++npmc1 > 1) {
				DPRINT("two events compete for a PMC1\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
 		 * UNHALTED_REFERENCE_CYCLES can only be measured on FIXED_CTR2
 		 */
		if (core_pe[e[i].event].pme_flags & PFMLIB_CORE_FIXED2_ONLY) {
			if (++nf2 > 1) {
				DPRINT("two events compete for FIXED_CTR2\n");
				return PFMLIB_ERR_NOASSIGN;
			}
			if (HAS_OPTIONS(i)) {
				DPRINT("fixed counters do not support inversion/counter-mask\n");
				return PFMLIB_ERR_NOASSIGN;
			}
		}
		/*
 		 * unit-mask level constraint checking (PMC0, PMC1, FIXED_CTR2)
 		 */	
		for(j=0; j < e[i].num_masks; j++) {
			unsigned int flags;

			flags = core_pe[e[i].event].pme_umasks[e[i].unit_masks[j]].pme_flags;

			if (flags & PFMLIB_CORE_FIXED2_ONLY) {
				if (++nf2 > 1) {
					DPRINT("two events compete for FIXED_CTR2\n");
					return PFMLIB_ERR_NOASSIGN;
				}
				if (HAS_OPTIONS(i)) {
					DPRINT("fixed counters do not support inversion/counter-mask\n");
					return PFMLIB_ERR_NOASSIGN;
				}
			}
                }
	}

	next_gen = 0; /* first generic counter */
	last_gen = 1; /* last generic counter */

	/*
	 * strongest constraint first: works only in IA32_PMC0, IA32_PMC1, FIXED_CTR2
	 *
	 * When PEBS is used, we pick the first PEBS event and
	 * place it into PMC0. Subsequent PEBS events, will go
	 * in the other counters.
	 */
	done_pebs = 0;
	for(i=0; i < n; i++) {
		if ((core_pe[e[i].event].pme_flags & PFMLIB_CORE_PMC0)
		    || (use_pebs && pfm_core_is_pebs(e+i) && done_pebs == 0)) {
			if (pfm_regmask_isset(r_pmcs, 0))
				return PFMLIB_ERR_NOASSIGN;
			assign_pc[i] = 0;
			next_gen = 1;
			done_pebs = 1;
		}
		if (core_pe[e[i].event].pme_flags & PFMLIB_CORE_PMC1) {
			if (pfm_regmask_isset(r_pmcs, 1))
				return PFMLIB_ERR_NOASSIGN;
			assign_pc[i] = 1;
			if (next_gen == 1)
				next_gen = 2;
			else
				next_gen = 0;
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
	 * 	- 18 : fixed counter 1 (pmc16, pmd18)
	 */
	fixed_ctr = pfm_regmask_isset(r_pmcs, 16) ? 0 : 0x7;
	if (fixed_ctr) {
		for(i=0; i < n; i++) {
			/* fixed counters do not support event options (filters) */
			if (HAS_OPTIONS(i) || (use_pebs && pfm_core_is_pebs(e+i)))
				continue;

			if ((fixed_ctr & 0x1) && pfm_core_is_fixed(e+i, 0)) {
				assign_pc[i] = 16;
				fixed_ctr &= ~1;
			}
			if ((fixed_ctr & 0x2) && pfm_core_is_fixed(e+i, 1)) {
				assign_pc[i] = 17;
				fixed_ctr &= ~2;
			}
			if ((fixed_ctr & 0x4) && pfm_core_is_fixed(e+i, 2)) {
				assign_pc[i] = 18;
				fixed_ctr &= ~4;
			}
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
	j = 0;

	/* setup fixed counters */
	reg.val = 0;
	k = 0;
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
		val |= 1ULL << 3;	 /* force APIC int (kernel may force it anyway) */

		reg.val |= val << ((assign_pc[i]-16)<<2);
	}

	if (reg.val) {
		pc[npc].reg_num   = 16;
		pc[npc].reg_value = reg.val;
		pc[npc].reg_addr  = 0x38D;
		pc[npc].reg_alt_addr  = 0x38D;

		__pfm_vbprintf("[FIXED_CTRL(pmc%u)=0x%"PRIx64" pmi0=1 en0=0x%"PRIx64" pmi1=1 en1=0x%"PRIx64" pmi2=1 en2=0x%"PRIx64"] ",
				pc[npc].reg_num,
				reg.val,
				reg.val & 0x3ULL,
				(reg.val>>4) & 0x3ULL,
				(reg.val>>8) & 0x3ULL);

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

	for (i=0; i < n ; i++ ) {
		/* skip fixed counters */
		if (is_fixed_pmc(assign_pc[i]))
			continue;

		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;

		val = core_pe[e[i].event].pme_code;

		reg.sel_event_select = val  & 0xff;

		ucode = (val >> 8) & 0xff;

		for(k=0; k < e[i].num_masks; k++) {
			ucode |= core_pe[e[i].event].pme_umasks[e[i].unit_masks[k]].pme_ucode;
		}

		/*
		 * for events supporting Core specificity (self, both), a value
		 * of 0 for bits 15:14 (7:6 in our umask) is reserved, therefore we
		 * force to SELF if user did not specify anything
		 */
		if ((core_pe[e[i].event].pme_flags & PFMLIB_CORE_CSPEC)
		    && ((ucode & (0x3 << 6)) == 0)) {
			ucode |= 1 << 6;
		}
		/*
		 * for events supporting MESI, a value
		 * of 0 for bits 11:8 (0-3 in our umask) means nothing will be
		 * counted. Therefore, we force a default of 0xf (M,E,S,I).
		 */
		if ((core_pe[e[i].event].pme_flags & PFMLIB_CORE_MESI)
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
				reg.sel_edge = cntrs[i].flags & PFM_CORE_SEL_EDGE ? 1 : 0;
			if (!reg.sel_inv)
				reg.sel_inv = cntrs[i].flags & PFM_CORE_SEL_INV ? 1 : 0;
		}

		pc[npc].reg_num     = assign_pc[i];
		pc[npc].reg_value   = reg.val;
		pc[npc].reg_addr    = CORE_SEL_BASE+assign_pc[i];
		pc[npc].reg_alt_addr= CORE_SEL_BASE+assign_pc[i];

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
				core_pe[e[i].event].pme_name);

		__pfm_vbprintf("[PMC%u(pmd%u)]\n",
				pc[npc].reg_num,
				pc[npc].reg_num);

		npc++;
	}
	/*
	 * setup pmds: must be in the same order as the events
	 */
	for (i=0; i < n ; i++) {
		if (is_fixed_pmc(assign_pc[i])) {
			/* setup pd array */
			pd[i].reg_num = assign_pc[i];
			pd[i].reg_addr = FIXED_CTR_BASE+assign_pc[i]-16;
			pd[i].reg_alt_addr = 0x40000000+assign_pc[i]-16;
		} else {
			pd[i].reg_num  = assign_pc[i];
			pd[i].reg_addr = CORE_CTR_BASE+assign_pc[i];
			/* index to use with RDPMC */
			pd[i].reg_alt_addr  = assign_pc[i];
		}
	}
	outp->pfp_pmd_count = i;

	/*
	 * setup PEBS_ENABLE
	 */
	if (use_pebs && done_pebs) {
		/*
		 * check that PEBS_ENABLE is available
		 */
		if (pfm_regmask_isset(r_pmcs, 17))
			return PFMLIB_ERR_NOASSIGN;
		pc[npc].reg_num   = 17;
		pc[npc].reg_value = 1ULL;
		pc[npc].reg_addr  = 0x3f1; /* IA32_PEBS_ENABLE */
		pc[npc].reg_alt_addr  = 0x3f1; /* IA32_PEBS_ENABLE */

		__pfm_vbprintf("[PEBS_ENABLE(pmc%u)=0x%"PRIx64" ena=%d]\n",
				pc[npc].reg_num,
				pc[npc].reg_value,
				pc[npc].reg_value & 0x1ull);

		npc++;

	}
	outp->pfp_pmc_count = npc;

	return PFMLIB_SUCCESS;
}

#if 0
static int
pfm_core_dispatch_pebs(pfmlib_input_param_t *inp, pfmlib_core_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_event_t *e;
	pfm_core_sel_reg_t reg;
	unsigned int umask, npc, npd, k, plm;
	pfmlib_regmask_t *r_pmcs;
	pfmlib_reg_t *pc, *pd;
	int event;

	npc = outp->pfp_pmc_count;
	npd = outp->pfp_pmd_count;
	pc  = outp->pfp_pmcs;
	pd  = outp->pfp_pmds;
	r_pmcs = &inp->pfp_unavail_pmcs;

	e = inp->pfp_events;
	/*
	 * check for valid flags
	 */
	if (e[0].flags & ~PFMLIB_CORE_ALL_FLAGS)
		return PFMLIB_ERR_INVAL;

	/*
	 * check event supports PEBS
	 */
	if (pfm_core_is_pebs(e) == 0)
		return PFMLIB_ERR_FEATCOMB;

	/*
	 * check that PMC0 is available
	 * PEBS works only on PMC0
	 * Some PEBS at-retirement events do require PMC0 anyway
	 */
	if (pfm_regmask_isset(r_pmcs, 0))
		return PFMLIB_ERR_NOASSIGN;

	/*
	 * check that PEBS_ENABLE is available
	 */
	if (pfm_regmask_isset(r_pmcs, 17))
		return PFMLIB_ERR_NOASSIGN;

	reg.val = 0; /* assume reserved bits are zerooed */

	event = e[0].event;

	/* if plm is 0, then assume not specified per-event and use default */
	plm = e[0].plm ? e[0].plm : inp->pfp_dfl_plm;

	reg.sel_event_select = core_pe[event].pme_code & 0xff;

	umask = (core_pe[event].pme_code >> 8) & 0xff;

	for(k=0; k < e[0].num_masks; k++) {
		umask |= core_pe[event].pme_umasks[e[0].unit_masks[k]].pme_ucode;
	}
	reg.sel_unit_mask  = umask;
	reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
	reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
	reg.sel_en         = 1; /* force enable bit to 1 */
	reg.sel_int        = 0; /* not INT for PEBS counter */

	reg.sel_cnt_mask = mod_in->pfp_core_counters[0].cnt_mask;
	reg.sel_edge	 = mod_in->pfp_core_counters[0].flags & PFM_CORE_SEL_EDGE ? 1 : 0;
	reg.sel_inv	 = mod_in->pfp_core_counters[0].flags & PFM_CORE_SEL_INV ? 1 : 0;

	pc[npc].reg_num     = 0;
	pc[npc].reg_value   = reg.val;
	pc[npc].reg_addr    = CORE_SEL_BASE;
	pc[npc].reg_alt_addr= CORE_SEL_BASE;

	pd[npd].reg_num  = 0;
	pd[npd].reg_addr = CORE_CTR_BASE;
	pd[npd].reg_alt_addr = 0;

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
		core_pe[e[0].event].pme_name);

	__pfm_vbprintf("[PMC%u(pmd%u)]\n",
			pd[npd].reg_num,
			pd[npd].reg_num);

	npc++;
	npd++;
	/*
	 * setup PEBS_ENABLE
	 */
	pc[npc].reg_num   = 17;
	pc[npc].reg_value = 1ULL;
	pc[npc].reg_addr  = 0x3f1; /* IA32_PEBS_ENABLE */
	pc[npc].reg_alt_addr = 0x3f1; /* IA32_PEBS_ENABLE */

	__pfm_vbprintf("[PEBS_ENABLE(pmc%u)=0x%"PRIx64" ena=%d]\n",
			pc[npc].reg_num,
			pc[npc].reg_value,
			pc[npc].reg_value & 0x1ull);

	npc++;

	/* number of evtsel/ctr registers programmed */
	outp->pfp_pmc_count = npc;
	outp->pfp_pmd_count = npd;

	return PFMLIB_SUCCESS;
}
#endif

static int
pfm_core_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_core_input_param_t *mod_in  = (pfmlib_core_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT("invalid plm=%x\n", inp->pfp_dfl_plm);
		return PFMLIB_ERR_INVAL;
	}
	return pfm_core_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_core_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST
	    && (cnt > highest_counter ||
		!pfm_regmask_isset(&core_impl_pmds, cnt)))
		return PFMLIB_ERR_INVAL;

	*code = core_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_core_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int n, i;
	unsigned int has_f0, has_f1, has_f2;

	memset(counters, 0, sizeof(*counters));

	n = core_pe[j].pme_numasks;
	has_f0 = has_f1 = has_f2 = 0;

	for (i=0; i < n; i++) {
		if (core_pe[j].pme_umasks[i].pme_flags & PFMLIB_CORE_FIXED0)
			has_f0 = 1;
		if (core_pe[j].pme_umasks[i].pme_flags & PFMLIB_CORE_FIXED1)
			has_f1 = 1;
		if (core_pe[j].pme_umasks[i].pme_flags & PFMLIB_CORE_FIXED2_ONLY)
			has_f2 = 1;
	}

	if (has_f0 == 0)
		has_f0 = core_pe[j].pme_flags & PFMLIB_CORE_FIXED0;
	if (has_f1 == 0)
		has_f1 = core_pe[j].pme_flags & PFMLIB_CORE_FIXED1;
	if (has_f2 == 0)
		has_f2 = core_pe[j].pme_flags & PFMLIB_CORE_FIXED2_ONLY;

	if (has_f0)
		pfm_regmask_set(counters, 16);
	if (has_f1)
		pfm_regmask_set(counters, 17);
	if (has_f2)
		pfm_regmask_set(counters, 18);

	/* the event on FIXED_CTR2 is exclusive CPU_CLK_UNHALTED:REF */
	if (!has_f2) {
		pfm_regmask_set(counters, 0);
		pfm_regmask_set(counters, 1);

		if (core_pe[j].pme_flags & PFMLIB_CORE_PMC0)
			pfm_regmask_clr(counters, 1);
		if (core_pe[j].pme_flags & PFMLIB_CORE_PMC1)
			pfm_regmask_clr(counters, 0);
	}
}

static void
pfm_core_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	*impl_pmcs = core_impl_pmcs;
}

static void
pfm_core_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	*impl_pmds = core_impl_pmds;
}

static void
pfm_core_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	pfm_regmask_set(impl_counters, 0);
	pfm_regmask_set(impl_counters, 1);
	pfm_regmask_set(impl_counters, 16);
	pfm_regmask_set(impl_counters, 17);
	pfm_regmask_set(impl_counters, 18);
}

/*
 * Even though, CPUID 0xa returns in eax the actual counter
 * width, the architecture specifies that writes are limited
 * to lower 32-bits. As such, only the lower 32-bit have full
 * degree of freedom. That is the "useable" counter width.
 */
#define PMU_CORE_COUNTER_WIDTH       32

static void
pfm_core_get_hw_counter_width(unsigned int *width)
{
	/*
	 * Even though, CPUID 0xa returns in eax the actual counter
	 * width, the architecture specifies that writes are limited
	 * to lower 32-bits. As such, only the lower 31 bits have full
	 * degree of freedom. That is the "useable" counter width.
	 */
	*width = PMU_CORE_COUNTER_WIDTH;
}

static char *
pfm_core_get_event_name(unsigned int i)
{
	return core_pe[i].pme_name;
}

static int
pfm_core_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = core_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static char *
pfm_core_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return core_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_core_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = core_pe[ev].pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_core_get_num_event_masks(unsigned int ev)
{
	return core_pe[ev].pme_numasks;
}

static int
pfm_core_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code =core_pe[ev].pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_core_get_cycle_event(pfmlib_event_t *e)
{
	e->event = PME_CORE_UNHALTED_CORE_CYCLES;
	return PFMLIB_SUCCESS;
}

static int
pfm_core_get_inst_retired(pfmlib_event_t *e)
{
	e->event = PME_CORE_INSTRUCTIONS_RETIRED;
	return PFMLIB_SUCCESS;
}

int
pfm_core_is_pebs(pfmlib_event_t *e)
{
	unsigned int i, n=0;

	if (e == NULL || e->event >= PME_CORE_EVENT_COUNT)
		return 0;

	if (core_pe[e->event].pme_flags & PFMLIB_CORE_PEBS)
		return 1;

	/*
	 * ALL unit mask must support PEBS for this test to return true
	 */
	for(i=0; i < e->num_masks; i++) {
		/* check for valid unit mask */
		if (e->unit_masks[i] >= core_pe[e->event].pme_numasks)
			return 0;
		if (core_pe[e->event].pme_umasks[e->unit_masks[i]].pme_flags & PFMLIB_CORE_PEBS)
			n++;
	}
	return n > 0 && n == e->num_masks;
}

pfm_pmu_support_t core_support={
	.pmu_name		= "Intel Core",
	.pmu_type		= PFMLIB_CORE_PMU,
	.pme_count		= PME_CORE_EVENT_COUNT,
	.pmc_count		= 4,
	.pmd_count		= 14,
	.num_cnt		= 5,
	.get_event_code		= pfm_core_get_event_code,
	.get_event_name		= pfm_core_get_event_name,
	.get_event_counters	= pfm_core_get_event_counters,
	.dispatch_events	= pfm_core_dispatch_events,
	.pmu_detect		= pfm_core_detect,
	.pmu_init		= pfm_core_init,
	.get_impl_pmcs		= pfm_core_get_impl_pmcs,
	.get_impl_pmds		= pfm_core_get_impl_pmds,
	.get_impl_counters	= pfm_core_get_impl_counters,
	.get_hw_counter_width	= pfm_core_get_hw_counter_width,
	.get_event_desc         = pfm_core_get_event_description,
	.get_num_event_masks	= pfm_core_get_num_event_masks,
	.get_event_mask_name	= pfm_core_get_event_mask_name,
	.get_event_mask_code	= pfm_core_get_event_mask_code,
	.get_event_mask_desc	= pfm_core_get_event_mask_desc,
	.get_cycle_event	= pfm_core_get_cycle_event,
	.get_inst_retired_event = pfm_core_get_inst_retired
};
