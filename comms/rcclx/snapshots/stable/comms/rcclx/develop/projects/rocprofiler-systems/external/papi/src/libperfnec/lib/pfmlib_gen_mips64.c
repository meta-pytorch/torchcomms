/*
 * pfmlib_gen_mips64.c : support for the generic MIPS64 PMU family
 *
 * Contributed by Philip Mucci <mucci@cs.utk.edu> based on code from
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* public headers */
#include <perfmon/pfmlib_gen_mips64.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_gen_mips64_priv.h"		/* architecture private */
#include "gen_mips64_events.h"		/* PMU private */

/* let's define some handy shortcuts! */
#define sel_event_mask	perfsel.sel_event_mask
#define sel_exl		perfsel.sel_exl
#define sel_os		perfsel.sel_os
#define sel_usr		perfsel.sel_usr
#define sel_sup		perfsel.sel_sup
#define sel_int		perfsel.sel_int

static pme_gen_mips64_entry_t *gen_mips64_pe = NULL;

pfm_pmu_support_t generic_mips64_support;

static int
pfm_gen_mips64_detect(void)
{
	static char mips_name[64] = "";
	int ret;
	char buffer[128];

	ret = __pfm_getcpuinfo_attr("cpu model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	generic_mips64_support.pmu_name = mips_name;
	generic_mips64_support.num_cnt = 0;
	if (strstr(buffer,"MIPS 20Kc"))
	  {
		gen_mips64_pe = gen_mips64_20K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS20KC"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_20K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 1;
		generic_mips64_support.pmd_count = 1;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_20KC_PMU;
	  }
	else if (strstr(buffer,"MIPS 24K"))
	  {
		gen_mips64_pe = gen_mips64_24K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS24K"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_24K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_24K_PMU;
	  }
	else if (strstr(buffer,"MIPS 25Kf"))
	  {
		gen_mips64_pe = gen_mips64_25K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS25KF"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_25K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_25KF_PMU;
	  }
	else if (strstr(buffer,"MIPS 34K"))
	  {
		gen_mips64_pe = gen_mips64_34K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS34K"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_34K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 4;
		generic_mips64_support.pmd_count = 4;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_34K_PMU;
	  }
	else if (strstr(buffer,"MIPS 5Kc"))
	  {
		gen_mips64_pe = gen_mips64_5K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS5KC"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_5K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_5KC_PMU;
	}
#if 0
	else if (strstr(buffer,"MIPS 74K"))
	  {
		gen_mips64_pe = gen_mips64_74K_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPS74K"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_74K_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 4;
		generic_mips64_support.pmd_count = 4;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_74K_PMU;
	}
#endif
	else if (strstr(buffer,"R10000"))
	  {
		gen_mips64_pe = gen_mips64_r10000_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPSR10000"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_r10000_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_R10000_PMU;
	  }
	else if (strstr(buffer,"R12000"))
	  {
		gen_mips64_pe = gen_mips64_r12000_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPSR12000"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_r12000_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 4;
		generic_mips64_support.pmd_count = 4;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_R12000_PMU;
	  }
	else if (strstr(buffer,"RM7000"))
	  {
		gen_mips64_pe = gen_mips64_rm7000_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPSRM7000"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_rm7000_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_RM7000_PMU;
	  }
	else if (strstr(buffer,"RM9000"))
	  {
		gen_mips64_pe = gen_mips64_rm9000_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPSRM9000"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_rm9000_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_RM9000_PMU;
	  }
	else if (strstr(buffer,"SB1"))
	  {
		gen_mips64_pe = gen_mips64_sb1_pe;
		strcpy(generic_mips64_support.pmu_name,"MIPSSB1"),
			generic_mips64_support.pme_count = (sizeof(gen_mips64_sb1_pe)/sizeof(pme_gen_mips64_entry_t));
		generic_mips64_support.pmc_count = 4;
		generic_mips64_support.pmd_count = 4;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_SB1_PMU;
	  }
	else if (strstr(buffer,"VR5432"))
	  {
		gen_mips64_pe = gen_mips64_vr5432_pe;
		generic_mips64_support.pme_count = (sizeof(gen_mips64_vr5432_pe)/sizeof(pme_gen_mips64_entry_t));
		strcpy(generic_mips64_support.pmu_name,"MIPSVR5432"),
			generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_VR5432_PMU;
	  }
	else if (strstr(buffer,"VR5500"))
	  {
		gen_mips64_pe = gen_mips64_vr5500_pe;
		generic_mips64_support.pme_count = (sizeof(gen_mips64_vr5500_pe)/sizeof(pme_gen_mips64_entry_t));
		strcpy(generic_mips64_support.pmu_name,"MIPSVR5500"),
			generic_mips64_support.pmc_count = 2;
		generic_mips64_support.pmd_count = 2;
		generic_mips64_support.pmu_type = PFMLIB_MIPS_VR5500_PMU;
	}
	else
		return PFMLIB_ERR_NOTSUPP;

	if (generic_mips64_support.num_cnt == 0)
	generic_mips64_support.num_cnt = generic_mips64_support.pmd_count;

	return PFMLIB_SUCCESS;
}

static void stuff_regs(pfmlib_event_t *e, int plm, pfmlib_reg_t *pc, pfmlib_reg_t *pd, int cntr, int j, pfmlib_gen_mips64_input_param_t *mod_in)
{
	pfm_gen_mips64_sel_reg_t reg;

	reg.val    = 0; /* assume reserved bits are zerooed */

	/* if plm is 0, then assume not specified per-event and use default */
	plm = e[j].plm ? e[j].plm : plm;
	reg.sel_usr = plm & PFM_PLM3 ? 1 : 0;
	reg.sel_os  = plm & PFM_PLM0 ? 1 : 0;
	reg.sel_sup = plm & PFM_PLM1 ? 1 : 0;
	reg.sel_exl = plm & PFM_PLM2 ? 1 : 0;
	reg.sel_int = 1; /* force int to 1 */

	reg.sel_event_mask = (gen_mips64_pe[e[j].event].pme_code >> (cntr*8)) & 0xff;
	pc[j].reg_value   = reg.val;
	pc[j].reg_addr    = cntr*2;
        pc[j].reg_num     = cntr;

	__pfm_vbprintf("[CP0_25_%"PRIx64"(pmc%u)=0x%"PRIx64" event_mask=0x%x usr=%d os=%d sup=%d exl=%d int=1] %s\n",
			pc[j].reg_addr,
			pc[j].reg_num,
			pc[j].reg_value,
			reg.sel_event_mask,
			reg.sel_usr,
			reg.sel_os,
			reg.sel_sup,
			reg.sel_exl,
			gen_mips64_pe[e[j].event].pme_name);

	pd[j].reg_num  = cntr;
	pd[j].reg_addr = cntr*2 + 1;

	__pfm_vbprintf("[CP0_25_%u(pmd%u)]\n",
			pc[j].reg_addr,
			pc[j].reg_num);
}
/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_regt structure is ready to be submitted to kernel
 */
static int
pfm_gen_mips64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_gen_mips64_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
        /* pfmlib_gen_mips64_input_param_t *param = mod_in; */
	pfmlib_event_t *e = inp->pfp_events;
	pfmlib_reg_t *pc, *pd;
	unsigned int i, j, cnt = inp->pfp_event_count;
	unsigned int used = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	pc = outp->pfp_pmcs;
	pd = outp->pfp_pmds;

	/* Degree 2 rank based allocation */
	if (cnt > generic_mips64_support.pmc_count) return PFMLIB_ERR_TOOMANY;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT("ev[%d]=%s, counters=0x%x\n", j, gen_mips64_pe[e[j].event].pme_name,gen_mips64_pe[e[j].event].pme_counters);
		}
	}

	/* Do rank based allocation, counters that live on 1 reg 
	   before counters that live on 2 regs etc. */
	for (i=1;i<=PMU_GEN_MIPS64_NUM_COUNTERS;i++)
	  {
	    for (j=0; j < cnt;j++) 
	      {
		unsigned int cntr, avail;
		if (pfmlib_popcnt(gen_mips64_pe[e[j].event].pme_counters) == i)
		  {
				/* These counters can be used for this event */
				avail = ~used & gen_mips64_pe[e[j].event].pme_counters;
		    DPRINT("Rank %d: Counters available 0x%x\n",i,avail);
				if (avail == 0x0)
					return PFMLIB_ERR_NOASSIGN;

				/* Pick one, mark as used*/
				cntr = ffs(avail) - 1;
		    DPRINT("Rank %d: Chose counter %d\n",i,cntr);

				/* Update registers */
				stuff_regs(e,inp->pfp_dfl_plm,pc,pd,cntr,j,mod_in);

				used |= (1 << cntr);
		    DPRINT("%d: Used counters 0x%x\n",i, used);
			}
		}
	}

	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;
	outp->pfp_pmd_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_mips64_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_gen_mips64_input_param_t *mod_in  = (pfmlib_gen_mips64_input_param_t *)model_in;

	return pfm_gen_mips64_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_gen_mips64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	extern pfm_pmu_support_t generic_mips64_support;

	/* check validity of counter index */
	if (cnt != PFMLIB_CNT_FIRST) {
		if (cnt < 0 || cnt >= generic_mips64_support.pmc_count)
	    return PFMLIB_ERR_INVAL; }
	else 	  {
		cnt = ffs(gen_mips64_pe[i].pme_counters)-1;
		if (cnt == -1)
			return(PFMLIB_ERR_INVAL);
	}

	/* if cnt == 1, shift right by 0, if cnt == 2, shift right by 8 */
	/* Works on both 5k anf 20K */

	if (gen_mips64_pe[i].pme_counters & (1<< cnt))
		*code = 0xff & (gen_mips64_pe[i].pme_code >> (cnt*8));
	else
		return PFMLIB_ERR_INVAL;

	return PFMLIB_SUCCESS;
}

static void
pfm_gen_mips64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	extern pfm_pmu_support_t generic_mips64_support;
	unsigned int tmp;

	memset(counters, 0, sizeof(*counters));
	tmp = gen_mips64_pe[j].pme_counters;

	while (tmp)
	       {
		int t = ffs(tmp) - 1;
			pfm_regmask_set(counters, t);
			tmp = tmp ^ (1 << t);
		}
	}

static void
pfm_gen_mips64_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	/* all pmcs are contiguous */
	for(i=0; i < generic_mips64_support.pmc_count; i++) pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_gen_mips64_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	/* all pmds are contiguous */
	for(i=0; i < generic_mips64_support.pmd_count; i++) pfm_regmask_set(impl_pmds, i);
}

static void
pfm_gen_mips64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	for(i=0; i < generic_mips64_support.pmc_count; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_gen_mips64_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_GEN_MIPS64_COUNTER_WIDTH;
}

static char *
pfm_gen_mips64_get_event_name(unsigned int i)
{
	return gen_mips64_pe[i].pme_name;
}

static int
pfm_gen_mips64_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = gen_mips64_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_mips64_get_cycle_event(pfmlib_event_t *e)
{
	return pfm_find_full_event("CYCLES",e);
}

static int
pfm_gen_mips64_get_inst_retired(pfmlib_event_t *e)
{
	if (pfm_current == NULL)
		return(PFMLIB_ERR_NOINIT);

  switch (pfm_current->pmu_type)
    {
		case PFMLIB_MIPS_20KC_PMU:
			return pfm_find_full_event("INSNS_COMPLETED",e);
		case PFMLIB_MIPS_24K_PMU:
			return pfm_find_full_event("INSTRUCTIONS",e); 
		case PFMLIB_MIPS_25KF_PMU:
			return pfm_find_full_event("INSNS_COMPLETE",e);
		case PFMLIB_MIPS_34K_PMU:
			return pfm_find_full_event("INSTRUCTIONS",e); 
		case PFMLIB_MIPS_5KC_PMU:
			return pfm_find_full_event("INSNS_EXECD",e); 
		case PFMLIB_MIPS_R10000_PMU:
		case PFMLIB_MIPS_R12000_PMU:
			return pfm_find_full_event("INSTRUCTIONS_GRADUATED",e);
		case PFMLIB_MIPS_RM7000_PMU:
		case PFMLIB_MIPS_RM9000_PMU:
			return pfm_find_full_event("INSTRUCTIONS_ISSUED",e); 
		case PFMLIB_MIPS_VR5432_PMU:
		case PFMLIB_MIPS_VR5500_PMU:
			return pfm_find_full_event("INSTRUCTIONS_EXECUTED",e); 
		case PFMLIB_MIPS_SB1_PMU:
			return pfm_find_full_event("INSN_SURVIVED_STAGE7",e); 
		default:
			return(PFMLIB_ERR_NOTFOUND);
	}
}

/* SiCortex specific functions */

pfm_pmu_support_t generic_mips64_support = {
	.pmu_name		= NULL,
	.pmu_type		= PFMLIB_UNKNOWN_PMU,
	.pme_count		= 0,
	.pmc_count		= 0,
	.pmd_count		= 0,
	.num_cnt		= 0,
	.flags			= PFMLIB_MULT_CODE_EVENT,
	.get_event_code		= pfm_gen_mips64_get_event_code,
	.get_event_name		= pfm_gen_mips64_get_event_name,
	.get_event_counters	= pfm_gen_mips64_get_event_counters,
	.dispatch_events	= pfm_gen_mips64_dispatch_events,
	.pmu_detect		= pfm_gen_mips64_detect,
	.get_impl_pmcs		= pfm_gen_mips64_get_impl_perfsel,
	.get_impl_pmds		= pfm_gen_mips64_get_impl_perfctr,
	.get_impl_counters	= pfm_gen_mips64_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_mips64_get_hw_counter_width,
	.get_event_desc         = pfm_gen_mips64_get_event_description,
	.get_cycle_event	= pfm_gen_mips64_get_cycle_event,
	.get_inst_retired_event = pfm_gen_mips64_get_inst_retired
};
