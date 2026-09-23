/*
 * pfmlib_gen_ia32.c : Intel architectural PMU v1, v2, v3
 *
 * The file provides support for the Intel architectural PMU v1 and v2.
 *
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
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
 * This file implements supports for the IA-32 architectural PMU as specified
 * in the following document:
 * 	"IA-32 Intel Architecture Software Developer's Manual - Volume 3B: System
 * 	Programming Guide"
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_gen_ia32.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_gen_ia32_priv.h"		/* architecture private */

#include "gen_ia32_events.h"			/* architected event table */

/* let's define some handy shortcuts! */
#define sel_event_select perfevtsel.sel_event_select
#define sel_unit_mask	 perfevtsel.sel_unit_mask
#define sel_usr		 perfevtsel.sel_usr
#define sel_os		 perfevtsel.sel_os
#define sel_edge	 perfevtsel.sel_edge
#define sel_pc		 perfevtsel.sel_pc
#define sel_int		 perfevtsel.sel_int
#define sel_any		 perfevtsel.sel_any
#define sel_en		 perfevtsel.sel_en
#define sel_inv		 perfevtsel.sel_inv
#define sel_cnt_mask	 perfevtsel.sel_cnt_mask

pfm_pmu_support_t *gen_support;

/*
 * Description of the PMC/PMD register mappings use by
 * this module (as reported in pfmlib_reg_t.reg_num)
 *
 * For V1 (up to 16 generic counters 0-15):
 *
 * 	0 -> PMC0 -> PERFEVTSEL0 -> MSR @ 0x186
 * 	1 -> PMC1 -> PERFEVTSEL1 -> MSR @ 0x187
 * 	...
 * 	n -> PMCn -> PERFEVTSELn -> MSR @ 0x186+n
 *
 * 	0 -> PMD0 -> IA32_PMC0   -> MSR @ 0xc1
 * 	1 -> PMD1 -> IA32_PMC1   -> MSR @ 0xc2
 * 	...
 * 	n -> PMDn -> IA32_PMCn   -> MSR @ 0xc1+n
 *
 * For V2 (up to 16 generic and 16 fixed counters):
 *
 * 	0 -> PMC0 -> PERFEVTSEL0 -> MSR @ 0x186
 * 	1 -> PMC1 -> PERFEVTSEL1 -> MSR @ 0x187
 * 	...
 * 	15 -> PMC15 -> PERFEVTSEL15 -> MSR @ 0x186+15
 *
 * 	16 -> PMC16 -> IA32_FIXED_CTR_CTRL -> MSR @ 0x38d
 *
 * 	0 -> PMD0 -> IA32_PMC0   -> MSR @ 0xc1
 * 	1 -> PMD1 -> IA32_PMC1   -> MSR @ 0xc2
 * 	...
 * 	15 -> PMD15 -> IA32_PMC15   -> MSR @ 0xc1+15
 *
 * 	16 -> PMD16 -> IA32_FIXED_CTR0 -> MSR @ 0x309
 * 	17 -> PMD17 -> IA32_FIXED_CTR1 -> MSR @ 0x30a
 * 	...
 * 	n -> PMDn -> IA32_FIXED_CTRn -> MSR @ 0x309+n
 */
#define GEN_IA32_SEL_BASE	  0x186
#define GEN_IA32_CTR_BASE	  0xc1
#define GEN_IA32_FIXED_CTR_BASE	  0x309

#define FIXED_PMD_BASE		16

#define PFMLIB_GEN_IA32_ALL_FLAGS \
	(PFM_GEN_IA32_SEL_INV|PFM_GEN_IA32_SEL_EDGE|PFM_GEN_IA32_SEL_ANYTHR)

static char * pfm_gen_ia32_get_event_name(unsigned int i);

static pme_gen_ia32_entry_t *gen_ia32_pe;

static int gen_ia32_cycle_event, gen_ia32_inst_retired_event;
static unsigned int num_fixed_cnt, num_gen_cnt, pmu_version;

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

static pfmlib_regmask_t gen_ia32_impl_pmcs, gen_ia32_impl_pmds;

/*
 * create architected event table
 */
static int
create_arch_event_table(unsigned int mask)
{
	pme_gen_ia32_entry_t *pe;
	unsigned int i, num_events = 0;
	unsigned int m;

	/*
	 * first pass: count the number of supported events
	 */
	m = mask;
	for(i=0; i < 7; i++, m>>=1) {
		if ((m & 0x1)  == 0)
			num_events++;
	}
	gen_ia32_support.pme_count = num_events;

	gen_ia32_pe = calloc(num_events, sizeof(pme_gen_ia32_entry_t));
	if (gen_ia32_pe == NULL)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * second pass: populate the table
	 */
	gen_ia32_cycle_event = gen_ia32_inst_retired_event = -1;
	m = mask;
	for(i=0, pe = gen_ia32_pe; i < 7; i++, m>>=1) {
		if ((m & 0x1)  == 0) {
			*pe = gen_ia32_all_pe[i];
			/*
			 * setup default event: cycles and inst_retired
			 */
			if (i == PME_GEN_IA32_UNHALTED_CORE_CYCLES)
				gen_ia32_cycle_event = pe - gen_ia32_pe;
			if (i == PME_GEN_IA32_INSTRUCTIONS_RETIRED)
				gen_ia32_inst_retired_event = pe - gen_ia32_pe;
			pe++;
		}
	}
	return PFMLIB_SUCCESS;
}

static int
check_arch_pmu(int family)
{
	union {
		unsigned int val;
		pmu_eax_t eax;
		pmu_edx_t edx;
	} eax, ecx, edx, ebx;

	/*
	 * check family number to reject for processors
	 * older than Pentium (family=5). Those processors
	 * did not have the CPUID instruction
	 */
	if (family < 5)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * check if CPU supports 0xa function of CPUID
	 * 0xa started with Core Duo. Needed to detect if
	 * architected PMU is present
	 */
	cpuid(0x0, &eax.val, &ebx.val, &ecx.val, &edx.val);
	if (eax.val < 0xa)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * extract architected PMU information
	 */
	cpuid(0xa, &eax.val, &ebx.val, &ecx.val, &edx.val);

	/*
	 * version must be greater than zero
	 */
	return eax.eax.version < 1 ? PFMLIB_ERR_NOTSUPP : PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_detect(void)
{
	int ret, family;
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

	return check_arch_pmu(family);
}

static int
pfm_gen_ia32_init(void)
{
	union {
		unsigned int val;
		pmu_eax_t eax;
		pmu_edx_t edx;
	} eax, ecx, edx, ebx;
	unsigned int num_cnt, i;
	int ret;

	/*
	 * extract architected PMU information
	 */
	if (forced_pmu == PFMLIB_NO_PMU) {
		cpuid(0xa, &eax.val, &ebx.val, &ecx.val, &edx.val);
	} else {
		/*
		 * when forced, simulate v2
		 * with 2 generic and 3 fixed counters
		 */
		eax.eax.version = 3;
		eax.eax.num_cnt = 2;
		eax.eax.cnt_width = 40;
		eax.eax.ebx_length = 0; /* unused */
		ebx.val = 0;
		edx.edx.num_cnt = 3;
		edx.edx.cnt_width = 40;
	}

	num_cnt = eax.eax.num_cnt;
	pmu_version = eax.eax.version;

	/* 
	 * populate impl_pm* bitmasks for generic counters
	 */
	for(i=0; i < num_cnt; i++) {
		pfm_regmask_set(&gen_ia32_impl_pmcs, i);
		pfm_regmask_set(&gen_ia32_impl_pmds, i);
	}

	/* check for fixed counters */
	if (pmu_version >= 2) {
		/*
		 * As described in IA-32 Developer's manual vol 3b
		 * in section 18.12.2.1, early processors supporting
		 * V2 may report invalid information concerning the fixed
		 * counters. So we compensate for this here by forcing
		 * num_cnt to 3.
		 */
		if (edx.edx.num_cnt == 0)
			edx.edx.num_cnt = 3;

		for(i=0; i < edx.edx.num_cnt; i++)
			pfm_regmask_set(&gen_ia32_impl_pmds, FIXED_PMD_BASE+i);
		if (i)
			pfm_regmask_set(&gen_ia32_impl_pmcs, 16);

	}

	num_gen_cnt = eax.eax.num_cnt;
	num_fixed_cnt = edx.edx.num_cnt;

	gen_ia32_support.pmc_count = num_gen_cnt + (num_fixed_cnt > 0);
	gen_ia32_support.pmd_count = num_gen_cnt + num_fixed_cnt;
	gen_ia32_support.num_cnt   = num_gen_cnt + num_fixed_cnt;

	__pfm_vbprintf("Intel architected PMU: version=%d num_gen=%u num_fixed=%u pmc=%u pmd=%d\n",
		pmu_version,
		num_gen_cnt,num_fixed_cnt,
		gen_ia32_support.pmc_count,
		gen_ia32_support.pmd_count);

	ret = create_arch_event_table(ebx.val);
	if (ret != PFMLIB_SUCCESS)
		return ret;

	gen_support = &gen_ia32_support;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_dispatch_counters_v1(pfmlib_input_param_t *inp, pfmlib_gen_ia32_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_gen_ia32_input_param_t *param = mod_in;
	pfmlib_gen_ia32_counter_t *cntrs;
	pfm_gen_ia32_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, cnt, k, ucode, val;
	unsigned int assign[PMU_GEN_IA32_MAX_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_gen_ia32_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT("ev[%d]=%s\n", j, gen_ia32_pe[e[j].event].pme_name);
		}
	}

	if (cnt > gen_support->pmd_count)
		return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++) {
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT("event=%d invalid plm=%d\n", e[j].event, e[j].plm);
			return PFMLIB_ERR_INVAL;
		}

		if (e[j].flags & ~PFMLIB_GEN_IA32_ALL_FLAGS) {
			DPRINT("event=%d invalid flags=0x%lx\n", e[j].event, e[j].flags);
			return PFMLIB_ERR_INVAL;
		}

		if (cntrs && pmu_version != 3 && (cntrs[j].flags & PFM_GEN_IA32_SEL_ANYTHR)) {
			DPRINT("event=%d anythread requires architectural perfmon v3", e[j].event);
			return PFMLIB_ERR_INVAL;
		}
		/*
		 * exclude restricted registers from assignment
		 */
		while(i < gen_support->pmc_count && pfm_regmask_isset(r_pmcs, i)) i++;

		if (i == gen_support->pmc_count)
			return PFMLIB_ERR_TOOMANY;

		/*
		 * events can be assigned to any counter
		 */
		assign[j] = i++;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		val = gen_ia32_pe[e[j].event].pme_code;

		reg.sel_event_select = val & 0xff;

		ucode = (val >> 8) & 0xff;

		for(k=0; k < e[j].num_masks; k++)
			ucode |= gen_ia32_pe[e[j].event].pme_umasks[e[j].unit_masks[k]].pme_ucode;

		val |= ucode << 8;

		reg.sel_unit_mask  = ucode; /* use 8 least significant bits */
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */

		reg.sel_cnt_mask = val >>24;
		reg.sel_inv = val >> 23;
		reg.sel_any = val >> 21;;
		reg.sel_edge = val >> 18;

		if (cntrs) {
			if (!reg.sel_cnt_mask) {
				/*
			 	 * counter mask is 8-bit wide, do not silently
			 	 * wrap-around
			 	 */
				if (cntrs[i].cnt_mask > 255)
					return PFMLIB_ERR_INVAL;
				reg.sel_cnt_mask = cntrs[j].cnt_mask;
			}

			if (!reg.sel_edge)
				reg.sel_edge = cntrs[j].flags & PFM_GEN_IA32_SEL_EDGE ? 1 : 0;
			if (!reg.sel_inv)
				reg.sel_inv = cntrs[j].flags & PFM_GEN_IA32_SEL_INV ? 1 : 0;
		}

		pc[j].reg_num     = assign[j];
		pc[j].reg_addr    = GEN_IA32_SEL_BASE+assign[j];
		pc[j].reg_value   = reg.val;

		pd[j].reg_num  = assign[j];
		pd[j].reg_addr = GEN_IA32_CTR_BASE+assign[j];

		__pfm_vbprintf("[PERFEVTSEL%u(pmc%u)=0x%llx event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
			assign[j],
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
			gen_ia32_pe[e[j].event].pme_name);

		__pfm_vbprintf("[PMC%u(pmd%u)]\n", pd[j].reg_num, pd[j].reg_num);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;
	outp->pfp_pmd_count = cnt;

	return PFMLIB_SUCCESS;
}

static const char *fixed_event_names[]={ "INSTRUCTIONS_RETIRED", "UNHALTED_CORE_CYCLES ", "UNHALTED_REFERENCE_CYCLES " };
#define MAX_EVENT_NAMES (sizeof(fixed_event_names)/sizeof(char *))

static int
pfm_gen_ia32_dispatch_counters_v23(pfmlib_input_param_t *inp, pfmlib_gen_ia32_input_param_t *param, pfmlib_output_param_t *outp)
{
#define HAS_OPTIONS(x)	(cntrs && (cntrs[i].flags || cntrs[i].cnt_mask))
#define is_fixed_pmc(a) (a > 15)

	pfmlib_gen_ia32_counter_t *cntrs;
	pfm_gen_ia32_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	uint64_t val;
	unsigned long plm;
	unsigned int fixed_ctr_mask;
	unsigned int npc = 0;
	unsigned int i, j, n, k, ucode;
	unsigned int assign[PMU_GEN_IA32_MAX_COUNTERS];
	unsigned int next_gen, last_gen;

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	n      = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_gen_ia32_counters : NULL;

	if (n > gen_support->pmd_count)
		return PFMLIB_ERR_TOOMANY;

	/*
	 * initilize to empty
	 */
	for(i=0; i < n; i++)
		assign[i] = -1;

	/*
	 * error checking
	 */
	for(j=0; j < n; j++) {
		/*
		 * only supports two priv levels for perf counters
		 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2))
			return PFMLIB_ERR_INVAL;

		/*
		 * check for valid flags
		 */
		if (cntrs && cntrs[j].flags & ~PFMLIB_GEN_IA32_ALL_FLAGS)
			return PFMLIB_ERR_INVAL;

		if (cntrs && pmu_version != 3 && (cntrs[j].flags & PFM_GEN_IA32_SEL_ANYTHR)) {
			DPRINT("event=%d anythread requires architectural perfmon v3", e[j].event);
			return PFMLIB_ERR_INVAL;
		}
	}

	next_gen = 0; /* first generic counter */
	last_gen = num_gen_cnt - 1; /* last generic counter */

	fixed_ctr_mask = (1 << num_fixed_cnt) - 1;
	/*
	 * first constraint: fixed counters (try using them first)
	 */
	if (fixed_ctr_mask) {
		for(i=0; i < n; i++) {
			/* fixed counters do not support event options (filters) */
			if (HAS_OPTIONS(i)) {
				if (pmu_version != 3)
					continue;
				if (cntrs[i].flags != PFM_GEN_IA32_SEL_ANYTHR)
					continue;
				/* ok for ANYTHR */
			}
			for(j=0; j < num_fixed_cnt; j++) {
				if ((fixed_ctr_mask & (1<<j)) && gen_ia32_pe[e[i].event].pme_fixed == (FIXED_PMD_BASE+j)) {
					assign[i] = FIXED_PMD_BASE+j;
					fixed_ctr_mask &= ~(1<<j);
					break;
				}
			}
		}
	}
	/*
	 * assign what is left
	 */
	for(i=0; i < n; i++) {
		if (assign[i] == -1) {
			for(; next_gen <= last_gen; next_gen++) {
				if (!pfm_regmask_isset(r_pmcs, next_gen))
					break;
			}
			if (next_gen <= last_gen)
				assign[i] = next_gen++;
			else
				return PFMLIB_ERR_NOASSIGN;
		}
	}
	j = 0;

	/* setup fixed counters */
	reg.val = 0;
	k = 0;
	for (i=0; i < n ; i++ ) {
		if (!is_fixed_pmc(assign[i]))
			continue;
		val = 0;
		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;
		if (plm & PFM_PLM0)
			val |= 1ULL;
		if (plm & PFM_PLM3)
			val |= 2ULL;

		/* only possible for v3 */
		if (cntrs && cntrs[i].flags & PFM_GEN_IA32_SEL_ANYTHR)
			val |= 4ULL;

		val |= 1ULL << 3;	 /* force APIC int (kernel may force it anyway) */

		reg.val |= val << ((assign[i]-FIXED_PMD_BASE)<<2);

		/* setup pd array */
		pd[i].reg_num = assign[i];
		pd[i].reg_addr = GEN_IA32_FIXED_CTR_BASE+assign[i]-FIXED_PMD_BASE;
	}

	if (reg.val) {
		pc[npc].reg_num   = 16;
		pc[npc].reg_value = reg.val;
		pc[npc].reg_addr  = 0x38D;

		__pfm_vbprintf("[FIXED_CTRL(pmc%u)=0x%"PRIx64,
				pc[npc].reg_num,
				reg.val);

		for(i=0; i < num_fixed_cnt; i++) {
			if (pmu_version != 3) 
				__pfm_vbprintf(" pmi%d=1 en%d=0x%"PRIx64,
					i, i,
					(reg.val >> (i*4)) & 0x3ULL);
			else
				__pfm_vbprintf(" pmi%d=1 en%d=0x%"PRIx64 " any%d=%"PRId64,
					i, i,
					(reg.val >> (i*4)) & 0x3ULL,
					i,
					!!((reg.val >> (i*4)) & 0x4ULL));
		}

		__pfm_vbprintf("] ");
		for(i=0; i < num_fixed_cnt; i++) {
			if ((fixed_ctr_mask & (0x1 << i)) == 0) {
				if (i < MAX_EVENT_NAMES)
					__pfm_vbprintf("%s ", fixed_event_names[i]);
				else
					__pfm_vbprintf("??? ");
			}
		}
		__pfm_vbprintf("\n");

		npc++;

		for (i=0; i < n ; i++ ) {
			if (!is_fixed_pmc(assign[i]))
				continue;
			__pfm_vbprintf("[FIXED_CTR%u(pmd%u)]\n", pd[i].reg_num, pd[i].reg_num);
		}
	}

	for (i=0; i < n ; i++ ) {
		/* skip fixed counters */
		if (is_fixed_pmc(assign[i]))
			continue;

		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[i].plm ? e[i].plm : inp->pfp_dfl_plm;

		val = gen_ia32_pe[e[i].event].pme_code;

		reg.sel_event_select = val & 0xff;

		ucode = (val >> 8) & 0xff;

		for(k=0; k < e[i].num_masks; k++)
			ucode |= gen_ia32_pe[e[i].event].pme_umasks[e[i].unit_masks[k]].pme_ucode;

		val |= ucode << 8;

		reg.sel_unit_mask  = ucode;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */

		reg.sel_cnt_mask = val >>24;
		reg.sel_inv = val >> 23;
		reg.sel_any = val >> 21;;
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
				reg.sel_edge = cntrs[i].flags & PFM_GEN_IA32_SEL_EDGE ? 1 : 0;
			if (!reg.sel_inv)
				reg.sel_inv = cntrs[i].flags & PFM_GEN_IA32_SEL_INV ? 1 : 0;
			if (!reg.sel_any)
				reg.sel_any = cntrs[i].flags & PFM_GEN_IA32_SEL_ANYTHR? 1 : 0;
		}

		pc[npc].reg_num     = assign[i];
		pc[npc].reg_value   = reg.val;
		pc[npc].reg_addr    = GEN_IA32_SEL_BASE+assign[i];
		pd[i].reg_num  = assign[i];
		pd[i].reg_addr = GEN_IA32_CTR_BASE+assign[i];

		if (pmu_version < 3)
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
					gen_ia32_pe[e[i].event].pme_name);
		else
			__pfm_vbprintf("[PERFEVTSEL%u(pmc%u)=0x%"PRIx64" event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d anythr=%d] %s\n",
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
					reg.sel_any,
					gen_ia32_pe[e[i].event].pme_name);

		__pfm_vbprintf("[PMC%u(pmd%u)]\n",
				pd[i].reg_num,
				pd[i].reg_num);

		npc++;
	}
	/* number of evtsel/ctr registers programmed */
	outp->pfp_pmc_count = npc;
	outp->pfp_pmd_count = n;
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_gen_ia32_input_param_t *mod_in  = model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT("invalid plm=%x\n", inp->pfp_dfl_plm);
		return PFMLIB_ERR_INVAL;
	}

	/* simplfied v1 (no fixed counters */
	if (pmu_version == 1)
		return pfm_gen_ia32_dispatch_counters_v1(inp, mod_in, outp);
	/* v2 or above */
	return pfm_gen_ia32_dispatch_counters_v23(inp, mod_in, outp);
}

static int
pfm_gen_ia32_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > gen_support->pmc_count)
		return PFMLIB_ERR_INVAL;

	*code = gen_ia32_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_gen_ia32_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));
	for(i=0; i < num_gen_cnt; i++)
		pfm_regmask_set(counters, i);

	for(i=0; i < num_fixed_cnt; i++) {
		if (gen_ia32_pe[j].pme_fixed == (FIXED_PMD_BASE+i))
			pfm_regmask_set(counters, FIXED_PMD_BASE+i);
	}
}

static void
pfm_gen_ia32_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	*impl_pmcs = gen_ia32_impl_pmcs;
}

static void
pfm_gen_ia32_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	*impl_pmds = gen_ia32_impl_pmds;
}

static void
pfm_gen_ia32_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	/* all pmds are counters */
	*impl_counters = gen_ia32_impl_pmds;
}

static void
pfm_gen_ia32_get_hw_counter_width(unsigned int *width)
{
	/*
	 * Even though, CPUID 0xa returns in eax the actual counter
	 * width, the architecture specifies that writes are limited
	 * to lower 32-bits. As such, only the lower 31 bits have full
	 * degree of freedom. That is the "useable" counter width.
	 */
	*width = PMU_GEN_IA32_COUNTER_WIDTH;
}

static char *
pfm_gen_ia32_get_event_name(unsigned int i)
{
	return gen_ia32_pe[i].pme_name;
}

static int
pfm_gen_ia32_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = gen_ia32_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static char *
pfm_gen_ia32_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return gen_ia32_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_gen_ia32_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = gen_ia32_pe[ev].pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_gen_ia32_get_num_event_masks(unsigned int ev)
{
	return gen_ia32_pe[ev].pme_numasks;
}

static int
pfm_gen_ia32_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code =gen_ia32_pe[ev].pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_get_cycle_event(pfmlib_event_t *e)
{
	if (gen_ia32_cycle_event == -1)
		return PFMLIB_ERR_NOTSUPP;

	e->event = gen_ia32_cycle_event;
	return PFMLIB_SUCCESS;

}

static int
pfm_gen_ia32_get_inst_retired(pfmlib_event_t *e)
{
	if (gen_ia32_inst_retired_event == -1)
		return PFMLIB_ERR_NOTSUPP;

	e->event = gen_ia32_inst_retired_event;
	return PFMLIB_SUCCESS;
}

/* architected PMU */
pfm_pmu_support_t gen_ia32_support={
	.pmu_name		= "Intel architectural PMU",
	.pmu_type		= PFMLIB_GEN_IA32_PMU,
	.pme_count		= 0,
	.pmc_count		= 0,
	.pmd_count		= 0,
	.num_cnt		= 0,
	.get_event_code		= pfm_gen_ia32_get_event_code,
	.get_event_name		= pfm_gen_ia32_get_event_name,
	.get_event_counters	= pfm_gen_ia32_get_event_counters,
	.dispatch_events	= pfm_gen_ia32_dispatch_events,
	.pmu_detect		= pfm_gen_ia32_detect,
	.pmu_init		= pfm_gen_ia32_init,
	.get_impl_pmcs		= pfm_gen_ia32_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_ia32_get_impl_pmds,
	.get_impl_counters	= pfm_gen_ia32_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_ia32_get_hw_counter_width,
	.get_event_desc         = pfm_gen_ia32_get_event_description,
	.get_cycle_event	= pfm_gen_ia32_get_cycle_event,
	.get_inst_retired_event = pfm_gen_ia32_get_inst_retired,
	.get_num_event_masks	= pfm_gen_ia32_get_num_event_masks,
	.get_event_mask_name	= pfm_gen_ia32_get_event_mask_name,
	.get_event_mask_code	= pfm_gen_ia32_get_event_mask_code,
	.get_event_mask_desc	= pfm_gen_ia32_get_event_mask_desc
};
