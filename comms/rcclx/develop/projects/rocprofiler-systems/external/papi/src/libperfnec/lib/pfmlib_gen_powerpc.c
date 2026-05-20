/*
 * Copyright (C) IBM Corporation, 2007.  All rights reserved.
 * Contributed by Corey Ashford (cjashfor@us.ibm.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * pfmlib_gen_powerpc.c
 *
 * Support for libpfm for the PowerPC970, POWER4,4+,5,5+,6 processors.
 */

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE /* for getline */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


/* private headers */
#include "powerpc_reg.h"

#include "pfmlib_priv.h"

#include "pfmlib_power_priv.h"
#include "pfmlib_ppc970_priv.h"
#include "pfmlib_ppc970mp_priv.h"
#include "pfmlib_power4_priv.h"
#include "pfmlib_power5_priv.h"
#include "pfmlib_power5+_priv.h"
#include "pfmlib_power6_priv.h"
#include "pfmlib_power7_priv.h"

#include "ppc970_events.h"
#include "ppc970mp_events.h"
#include "power4_events.h"
#include "power5_events.h"
#include "power5+_events.h"
#include "power6_events.h"
#include "power7_events.h"

#define FIRST_POWER_PMU PFMLIB_PPC970_PMU

static const int num_group_vec[] = {
	[PFMLIB_PPC970_PMU - FIRST_POWER_PMU] = PPC970_NUM_GROUP_VEC,
	[PFMLIB_PPC970MP_PMU - FIRST_POWER_PMU] = PPC970MP_NUM_GROUP_VEC,
	[PFMLIB_POWER4_PMU - FIRST_POWER_PMU] = POWER4_NUM_GROUP_VEC,
	[PFMLIB_POWER5_PMU - FIRST_POWER_PMU] = POWER5_NUM_GROUP_VEC,
	[PFMLIB_POWER5p_PMU - FIRST_POWER_PMU] = POWER5p_NUM_GROUP_VEC,
	[PFMLIB_POWER6_PMU - FIRST_POWER_PMU] = POWER6_NUM_GROUP_VEC,
	[PFMLIB_POWER7_PMU - FIRST_POWER_PMU] = POWER7_NUM_GROUP_VEC
};

static const int event_count[] = {
	[PFMLIB_PPC970_PMU - FIRST_POWER_PMU] = PPC970_PME_EVENT_COUNT,
	[PFMLIB_PPC970MP_PMU - FIRST_POWER_PMU] = PPC970MP_PME_EVENT_COUNT,
	[PFMLIB_POWER5_PMU - FIRST_POWER_PMU] = POWER5_PME_EVENT_COUNT,
	[PFMLIB_POWER5p_PMU - FIRST_POWER_PMU] = POWER5p_PME_EVENT_COUNT,
	[PFMLIB_POWER6_PMU - FIRST_POWER_PMU] = POWER6_PME_EVENT_COUNT,
	[PFMLIB_POWER7_PMU - FIRST_POWER_PMU] = POWER7_PME_EVENT_COUNT
};

unsigned *pmd_priv_vec;

static unsigned long long mmcr0_fc5_6_mask;
static unsigned long long *mmcr0_counter_mask;
static unsigned long long *mmcr1_counter_mask;
static unsigned long long *mmcr0_counter_off_val;
static unsigned long long *mmcr1_counter_off_val;

static const pme_power_entry_t *pe;
static const pmg_power_group_t *groups;

static inline int get_num_event_counters() {
	return gen_powerpc_support.pmd_count;
}

static inline int get_num_control_regs() {
	return gen_powerpc_support.pmc_count;
}

static inline const unsigned long long *get_group_vector(int event) {
	return pe[event].pme_group_vector;
}

static inline int get_event_id(int event, int counter) {
	return pe[event].pme_event_ids[counter];
}

static inline char *get_event_name(int event) {
	return pe[event].pme_name;
}

static inline char *get_long_desc(int event) {
	return pe[event].pme_long_desc;
}

static inline int get_group_event_id(int group, int counter) {
	return groups[group].pmg_event_ids[counter];
}

static inline unsigned long long get_mmcr0(int group) {
	return groups[group].pmg_mmcr0;
}

static inline unsigned long long get_mmcr1(int group) {
	return groups[group].pmg_mmcr1;
}

static inline unsigned long long get_mmcra(int group) {
	return groups[group].pmg_mmcra;
}
   

/**
 * pfm_gen_powerpc_get_event_code
 *
 * Return the event-select value for the specified event as
 * needed for the specified PMD counter.
 **/
static int pfm_gen_powerpc_get_event_code(unsigned int event,
				   unsigned int pmd,
				   int *code)
{
	if (event < event_count[gen_powerpc_support.pmu_type - FIRST_POWER_PMU]) {
		*code = pe[event].pme_code;
		return PFMLIB_SUCCESS;
	} else
		return PFMLIB_ERR_INVAL;
}

/**
 * pfm_gen_powerpc_get_event_name
 *
 * Return the name of the specified event.
 **/
static char *pfm_gen_powerpc_get_event_name(unsigned int event)
{
	return get_event_name(event);
}

/**
 * pfm_gen_powerpc_get_event_mask_name
 *
 * Return the name of the specified event-mask.
 **/
static char *pfm_gen_powerpc_get_event_mask_name(unsigned int event, unsigned int mask)
{
	return "";
}

/**
 * pfm_gen_powerpc_get_event_counters
 *
 * Fill in the 'counters' bitmask with all possible PMDs that could be
 * used to count the specified event.
 **/
static void pfm_gen_powerpc_get_event_counters(unsigned int event,
					pfmlib_regmask_t *counters)
{
	int i;

	counters->bits[0] = 0;	
	for (i = 0; i < get_num_event_counters(); i++) {
		if (get_event_id(event, i) != -1) {
			counters->bits[0] |= (1 << i);
		}
	}
}

/**
 * pfm_gen_powerpc_get_num_event_masks
 *
 * Count the number of available event-masks for the specified event.
 **/
static unsigned int pfm_gen_powerpc_get_num_event_masks(unsigned int event)
{
        /* POWER arch doesn't use event masks */
	return 0;
}

static void remove_group(unsigned long long *group_vec, int group)
{
        group_vec[group / 64] &= ~(1ULL << (group % 64));
}

static void intersect_groups(unsigned long long *result, const unsigned long long *operand)
{
        int i;

        for (i = 0; i < num_group_vec[gen_powerpc_support.pmu_type - FIRST_POWER_PMU]; i++) {
                result[i] &= operand[i];
        }
}

static int first_group(unsigned long long *group_vec)
{
        int i, bit;

        for (i = 0; i < num_group_vec[gen_powerpc_support.pmu_type - FIRST_POWER_PMU]; i++) {
                bit = ffsll(group_vec[i]);
                if (bit) {
                        return (bit - 1) + (i * 64);
                }
        }
        /* There were no groups */
        return -1;
}


static unsigned gq_pmd_priv_vec[8] = {
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e
};

static unsigned gr_pmd_priv_vec[6] = {
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
};

static unsigned gs_pmd_priv_vec[6] = {
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0f0e,
	0x0800,
	0x0800,
};

/* These masks are used on the PPC970*, and POWER4,4+ chips */
static unsigned long long power4_mmcr0_counter_mask[POWER4_NUM_EVENT_COUNTERS] = {
	0x1fUL << (63 - 55), /* PMC1 */
	0x1fUL << (63 - 62), /* PMC2 */
	0,
	0,
	0,
	0,
	0,
	0
};
static unsigned long long power4_mmcr1_counter_mask[POWER4_NUM_EVENT_COUNTERS] = {
	0,
	0,
	0x1fUL << (63 - 36), /* PMC3 */
	0x1fUL << (63 - 41), /* PMC4 */
	0x1fUL << (63 - 46), /* PMC5 */
	0x1fUL << (63 - 51), /* PMC6 */
	0x1fUL << (63 - 56), /* PMC7 */
	0x1fUL << (63 - 61)  /* PMC8 */
};

static unsigned long long power4_mmcr0_counter_off_val[POWER4_NUM_EVENT_COUNTERS] = {
	0, /* PMC1 */
	0, /* PMC2 */
	0,
	0,
	0,
	0,
	0,
	0
};
static unsigned long long power4_mmcr1_counter_off_val[POWER4_NUM_EVENT_COUNTERS] = {
	0,
	0,
	0, /* PMC3 */
	0, /* PMC4 */
	0, /* PMC5 */
	0, /* PMC6 */
	0, /* PMC7 */
	0  /* PMC8 */
};

static unsigned long long ppc970_mmcr0_counter_off_val[POWER4_NUM_EVENT_COUNTERS] = {
	0x8UL << (63 - 55), /* PMC1 */
	0x8UL << (63 - 62), /* PMC2 */
	0,
	0,
	0,
	0,
	0,
	0
};
static unsigned long long ppc970_mmcr1_counter_off_val[POWER4_NUM_EVENT_COUNTERS] = {
	0,
	0,
	0x8UL << (63 - 36), /* PMC3 */
	0x8UL << (63 - 41), /* PMC4 */
	0x8UL << (63 - 46), /* PMC5 */
	0x8UL << (63 - 51), /* PMC6 */
	0x8UL << (63 - 56), /* PMC7 */
	0x8UL << (63 - 61)  /* PMC8 */
};

/* These masks are used on POWER5,5+,5++,6,7 */
static unsigned long long power5_mmcr0_counter_mask[POWER5_NUM_EVENT_COUNTERS] = {
	0,
	0,
	0,
	0,
	0,
	0
};
static unsigned long long power5_mmcr1_counter_mask[POWER5_NUM_EVENT_COUNTERS] = {
	0xffUL << (63 - 39), /* PMC1 */
	0xffUL << (63 - 47), /* PMC2 */
	0xffUL << (63 - 55), /* PMC3 */
	0xffUL << (63 - 63), /* PMC4 */
	0,
	0
};

static unsigned long long power5_mmcr0_counter_off_val[POWER5_NUM_EVENT_COUNTERS] = {
	0,
	0,
	0,
	0,
	0,
	0
};

static unsigned long long power5_mmcr1_counter_off_val[POWER5_NUM_EVENT_COUNTERS] = {
	0, /* PMC1 */
	0, /* PMC2 */
	0, /* PMC3 */
	0, /* PMC4 */
	0,
	0
};


/**
 * pfm_gen_powerpc_dispatch_events
 *
 * Examine each desired event specified in "input" and find an appropriate
 * set of PMCs and PMDs to count them.
 **/
static int pfm_gen_powerpc_dispatch_events(pfmlib_input_param_t *input,
				   void *model_input,
				   pfmlib_output_param_t *output,

				   void *model_output)
{
	/* model_input and model_output are unused on POWER */

	int i, j, group;
	int counters_used = 0;
	unsigned long long mmcr0_val, mmcr1_val;
	unsigned long long group_vector[num_group_vec[gen_powerpc_support.pmu_type - FIRST_POWER_PMU]];
	unsigned int plm;

	plm  = (input->pfp_events[0].plm != 0) ? input->pfp_events[0].plm : input->pfp_dfl_plm;

        /*
         * Verify that all of the privilege level masks are identical, as
         * we cannot have mixed levels on POWER
         */

	for (i = 1; i < input->pfp_event_count; i++) {
		if (input->pfp_events[i].plm == 0) {
			/* it's ok if the default is the same as plm */
			if (plm != input->pfp_dfl_plm) 
				return PFMLIB_ERR_NOASSIGN;
		} else {
			if (plm != input->pfp_events[i].plm)
				return PFMLIB_ERR_NOASSIGN;
		}
	}

	/* start by setting all of the groups as available */
	memset(group_vector, 0xff, sizeof(unsigned long long) * num_group_vec[gen_powerpc_support.pmu_type - FIRST_POWER_PMU]);

	for (i = 0; i < input->pfp_event_count; i++) {
		mmcr0_val |= mmcr0_counter_off_val[i];	
		intersect_groups(group_vector, get_group_vector(input->pfp_events[i].event));
		mmcr1_val |= mmcr1_counter_off_val[i];	
	}
	group = first_group(group_vector);
	while (group != -1) {
		/* find out if the the privilege levels are compatible with each counter */
		for (i = 0; i < input->pfp_event_count; i++) {
			/* find event counter in group */
			for (j = 0; j < get_num_event_counters(); j++) {
				if (get_event_id(input->pfp_events[i].event,j) == get_group_event_id(group, j)) {
					/* found counter */
					if (input->pfp_events[i].plm != 0) {
						if (! (pmd_priv_vec[j] & (1 << input->pfp_events[0].plm))) {
							remove_group(group_vector, group);
							group = first_group(group_vector);
							goto try_next_group;
						}
					} else {
						if (! (pmd_priv_vec[j] & (1 << input->pfp_dfl_plm))) {
							remove_group(group_vector, group);
							group = first_group(group_vector);
							goto try_next_group;
						}
					}
					/* We located this counter and its privilege checks out ok. */
					counters_used |= (1 << j);
					output->pfp_pmds[i].reg_value = 0;
					output->pfp_pmds[i].reg_addr = 0;
					output->pfp_pmds[i].reg_alt_addr = 0;
					output->pfp_pmds[i].reg_num = j + 1;
					output->pfp_pmds[i].reg_reserved1 = 0;
					output->pfp_pmd_count = i + 1;

					/* Find the next counter */
					break;
				}
			}
			if (j == get_num_event_counters()) {
				printf ("libpfm: Internal error.  Unable to find counter in group.\n");
			}
		}
                /*
                 * Success!  We found a group (group) that meets the
                 * privilege constraints
                 */
		break; 
try_next_group: ;
	}
	if (group == -1)
		/* We did not find a group that meets the constraints */
		return PFMLIB_ERR_NOASSIGN;

	/* We now have a group that meets the constraints */

	mmcr0_val = get_mmcr0(group);
	mmcr1_val = get_mmcr1(group);
	for (i = 0; i < get_num_event_counters(); i++) {
		if (! (counters_used & (1 << i))) {	
			/*
			 * This counter is not used, so set that
			 * selector to its off value.
			 */
			mmcr0_val &= ~mmcr0_counter_mask[i];	
			mmcr0_val |= mmcr0_counter_off_val[i];	
			mmcr1_val &= ~mmcr1_counter_mask[i];	
			mmcr1_val |= mmcr1_counter_off_val[i];	
		}
	}
	/*
	 * As a special case for PMC5 and PMC6 on POWER5/5+, freeze these
	 * two counters if neither are used.  Note that the
	 * mmcr0_fc5_6_mask is zero for all processors except POWER5/5+
	 */
	if ((counters_used & ((1 << (5 - 1)) | (1 << (6 - 1)))) == 0)
		mmcr0_val |= mmcr0_fc5_6_mask;

        /*
         * Enable counter "exception on negative" and performance monitor
         * exceptions
         */
	mmcr0_val |= MMCR0_PMXE | MMCR0_PMC1CE | MMCR0_PMCjCE;

	/* Start with the counters frozen in every state, then selectively
           enable them */

	mmcr0_val |= MMCR0_FCP | MMCR0_FCS | MMCR0_FCHV;

	if (plm & PFM_PLM3) {
		/* user */
		mmcr0_val &= ~MMCR0_FCP;
	}
	if (plm & PFM_PLM0) {
		/* kernel */
		mmcr0_val &= ~MMCR0_FCS;
	}
	if (plm & PFM_PLM1) {
		/* hypervisor */
		mmcr0_val &= ~MMCR0_FCHV;
	}
        /* PFM_PLM2 is not supported */

	output->pfp_pmcs[0].reg_value = mmcr0_val;
	output->pfp_pmcs[0].reg_addr = 0;
	output->pfp_pmcs[0].reg_alt_addr = 0;
	output->pfp_pmcs[0].reg_num = 0;
	output->pfp_pmcs[0].reg_reserved1 = 0;

	output->pfp_pmcs[1].reg_value = mmcr1_val;
	output->pfp_pmcs[1].reg_addr = 0;
	output->pfp_pmcs[1].reg_alt_addr = 0;
	output->pfp_pmcs[1].reg_num = 1;
	output->pfp_pmcs[1].reg_reserved1 = 0;

	output->pfp_pmcs[2].reg_value = get_mmcra(group);
	output->pfp_pmcs[2].reg_addr = 0;
	output->pfp_pmcs[2].reg_alt_addr = 0;
	output->pfp_pmcs[2].reg_num = 2;
	output->pfp_pmcs[2].reg_reserved1 = 0;

	/* We always use the same number of control regs */
	output->pfp_pmc_count = get_num_control_regs();

	return PFMLIB_SUCCESS;
}


/**
 * pfm_gen_powerpc_pmu_detect
 *
 * Determine which POWER processor, if any, we are running on.
 *
 **/


/**
 * These should be defined in more recent versions of
 * /usr/include/asm-ppc64/reg.h.  It isn't pretty to have these here, but
 * maybe we can remove them someday.
 **/

static int pfm_gen_powerpc_pmu_detect(void)
{
	if (__is_processor(PV_970) || __is_processor(PV_970FX) || __is_processor(PV_970GX)) {
		gen_powerpc_support.pmu_type = PFMLIB_PPC970_PMU;          
		gen_powerpc_support.pmu_name = "PPC970";          
		gen_powerpc_support.pme_count = PPC970_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = PPC970_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = PPC970_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = PPC970_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = 0;
		mmcr0_counter_mask = power4_mmcr0_counter_mask;
		mmcr1_counter_mask = power4_mmcr1_counter_mask;
		mmcr0_counter_off_val = ppc970_mmcr0_counter_off_val;
		mmcr1_counter_off_val = ppc970_mmcr1_counter_off_val;
		pmd_priv_vec = gq_pmd_priv_vec;
		pe = ppc970_pe;
		groups = ppc970_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_970MP)) {
		gen_powerpc_support.pmu_type = PFMLIB_PPC970MP_PMU;          
		gen_powerpc_support.pmu_name = "PPC970MP";          
		gen_powerpc_support.pme_count = PPC970MP_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = PPC970MP_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = PPC970MP_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = PPC970MP_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = 0;
		mmcr0_counter_mask = power4_mmcr0_counter_mask;
		mmcr1_counter_mask = power4_mmcr1_counter_mask;
		mmcr0_counter_off_val = ppc970_mmcr0_counter_off_val;
		mmcr1_counter_off_val = ppc970_mmcr1_counter_off_val;
		pmd_priv_vec = gq_pmd_priv_vec;
		pe = ppc970mp_pe;
		groups = ppc970mp_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_POWER4) || __is_processor(PV_POWER4p)) {
		gen_powerpc_support.pmu_type = PFMLIB_PPC970_PMU;          
		gen_powerpc_support.pmu_name = "POWER4";          
		gen_powerpc_support.pme_count = POWER4_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = POWER4_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = POWER4_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = POWER4_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = 0;
		mmcr0_counter_mask = power4_mmcr0_counter_mask;
		mmcr1_counter_mask = power4_mmcr1_counter_mask;
		mmcr0_counter_off_val = ppc970_mmcr0_counter_off_val;
		mmcr1_counter_off_val = ppc970_mmcr1_counter_off_val;
		mmcr0_counter_off_val = power4_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power4_mmcr1_counter_off_val;
		pmd_priv_vec = gq_pmd_priv_vec;
		pe = power4_pe;
		groups = power4_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_POWER5)) {
		gen_powerpc_support.pmu_type = PFMLIB_POWER5_PMU;          
		gen_powerpc_support.pmu_name = "POWER5";          
		gen_powerpc_support.pme_count = POWER5_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = POWER5_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = POWER5_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = POWER5_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = MMCR0_FC5_6;
		mmcr0_counter_off_val = ppc970_mmcr0_counter_off_val;
		mmcr1_counter_off_val = ppc970_mmcr1_counter_off_val;
		mmcr0_counter_mask = power5_mmcr0_counter_mask;
		mmcr1_counter_mask = power5_mmcr1_counter_mask;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		pmd_priv_vec = gr_pmd_priv_vec;
		pe = power5_pe;
		groups = power5_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_POWER5p)) {
		gen_powerpc_support.pmu_type = PFMLIB_POWER5p_PMU;          
		gen_powerpc_support.pmu_name = "POWER5+";          
		gen_powerpc_support.pme_count = POWER5p_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = POWER5p_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = POWER5p_NUM_CONTROL_REGS;
		mmcr0_counter_off_val = power4_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power4_mmcr1_counter_off_val;
		gen_powerpc_support.num_cnt = POWER5p_NUM_EVENT_COUNTERS;
		mmcr0_counter_mask = power5_mmcr0_counter_mask;
		mmcr1_counter_mask = power5_mmcr1_counter_mask;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		if (PVR_VER(mfspr(SPRN_PVR)) >= 0x300) {
			/* this is a newer, GS model POWER5+ */
			mmcr0_fc5_6_mask = 0;
			pmd_priv_vec = gs_pmd_priv_vec;
		} else {
			mmcr0_fc5_6_mask = MMCR0_FC5_6;
			pmd_priv_vec = gr_pmd_priv_vec;
		}
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		pe = power5p_pe;
		groups = power5p_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_POWER6)) {
		gen_powerpc_support.pmu_type = PFMLIB_POWER6_PMU;          
		gen_powerpc_support.pmu_name = "POWER6";          
		gen_powerpc_support.pme_count = POWER6_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = POWER6_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = POWER6_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = POWER6_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = 0;
		mmcr0_counter_mask = power5_mmcr0_counter_mask;
		mmcr1_counter_mask = power5_mmcr1_counter_mask;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		pmd_priv_vec = gs_pmd_priv_vec;
		pe = power6_pe;
		groups = power6_groups;
		return PFMLIB_SUCCESS;
	}
	if (__is_processor(PV_POWER7)) {
		gen_powerpc_support.pmu_type = PFMLIB_POWER7_PMU;
		gen_powerpc_support.pmu_name = "POWER7";
		gen_powerpc_support.pme_count = POWER7_PME_EVENT_COUNT;
		gen_powerpc_support.pmd_count = POWER7_NUM_EVENT_COUNTERS;
		gen_powerpc_support.pmc_count = POWER7_NUM_CONTROL_REGS;
		gen_powerpc_support.num_cnt = POWER7_NUM_EVENT_COUNTERS;
		mmcr0_fc5_6_mask = 0;
		mmcr0_counter_mask = power5_mmcr0_counter_mask;
		mmcr1_counter_mask = power5_mmcr1_counter_mask;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		mmcr0_counter_off_val = power5_mmcr0_counter_off_val;
		mmcr1_counter_off_val = power5_mmcr1_counter_off_val;
		pmd_priv_vec = gr_pmd_priv_vec;
		pe = power7_pe;
		groups = power7_groups;
		return PFMLIB_SUCCESS;
	}

	return PFMLIB_ERR_NOTSUPP;
}

/**
 * pfm_gen_powerpc_get_impl_pmcs
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMC that's
 * available on power4.
 **/
static void pfm_gen_powerpc_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
        impl_pmcs->bits[0] = (0xffffffff >> (32 - get_num_control_regs()));
}

/**
 * pfm_gen_powerpc_get_impl_pmds
 *

 * Set the appropriate bit in the impl_pmcs bitmask for each PMD that's
 * available.
 **/
static void pfm_gen_powerpc_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
        impl_pmds->bits[0] = (0xffffffff >> (32 - get_num_event_counters()));
}

/**
 * pfm_gen_powerpc_get_impl_counters
 *
 * Set the appropriate bit in the impl_counters bitmask for each counter
 * that's available on power4.
 *
 * For now, all PMDs are counters, so just call get_impl_pmds().
 **/
static void pfm_gen_powerpc_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	pfm_gen_powerpc_get_impl_pmds(impl_counters);
}

/**
 * pfm_gen_powerpc_get_hw_counter_width
 *
 * Return the number of usable bits in the PMD counters.
 **/
static void pfm_gen_powerpc_get_hw_counter_width(unsigned int *width)
{
	*width = 64;
}

/**
 * pfm_gen_powerpc_get_event_desc
 *
 * Return the description for the specified event (if it has one).
 **/
static int pfm_gen_powerpc_get_event_desc(unsigned int event, char **desc)
{
	*desc = strdup(get_long_desc(event));
	return 0;
}

/**
 * pfm_gen_powerpc_get_event_mask_desc
 *
 * Return the description for the specified event-mask (if it has one).
 **/
static int pfm_gen_powerpc_get_event_mask_desc(unsigned int event,
					unsigned int mask, char **desc)
{
	*desc = strdup("");
	return 0;
}

static int pfm_gen_powerpc_get_event_mask_code(unsigned int event,
				 	unsigned int mask, unsigned int *code)
{
	*code = 0;
	return 0;
}

static int
pfm_gen_powerpc_get_cycle_event(pfmlib_event_t *e)
{
	switch (gen_powerpc_support.pmu_type) {
	case PFMLIB_PPC970_PMU:
		e->event = PPC970_PME_PM_CYC;
		break;
	case PFMLIB_PPC970MP_PMU:
		e->event = PPC970MP_PME_PM_CYC;
		break;
	case PFMLIB_POWER4_PMU:
		e->event = POWER4_PME_PM_CYC;
		break;
	case PFMLIB_POWER5_PMU:
		e->event = POWER5_PME_PM_CYC;
		break;
	case PFMLIB_POWER5p_PMU:
		e->event = POWER5p_PME_PM_RUN_CYC;
		break;
	case PFMLIB_POWER6_PMU:
		e->event = POWER6_PME_PM_RUN_CYC;
		break;
	case PFMLIB_POWER7_PMU:
		e->event = POWER7_PME_PM_RUN_CYC;
		break;
	default:
		/* perhaps gen_powerpc_suport.pmu_type wasn't initialized? */
		return PFMLIB_ERR_NOINIT;
	}
	e->num_masks = 0;
	e->unit_masks[0] = 0;
	return PFMLIB_SUCCESS;

}

static int
pfm_gen_powerpc_get_inst_retired(pfmlib_event_t *e)
{
	switch (gen_powerpc_support.pmu_type) {
	case PFMLIB_PPC970_PMU:
		e->event = PPC970_PME_PM_INST_CMPL;
		break;
	case PFMLIB_PPC970MP_PMU:
		e->event = PPC970MP_PME_PM_INST_CMPL;
		break;
	case PFMLIB_POWER4_PMU:
		e->event = POWER4_PME_PM_INST_CMPL;
		break;
	case PFMLIB_POWER5_PMU:
		e->event = POWER5_PME_PM_INST_CMPL;
		break;
	case PFMLIB_POWER5p_PMU:
		e->event = POWER5p_PME_PM_INST_CMPL;
		break;
	case PFMLIB_POWER6_PMU:
		e->event = POWER6_PME_PM_INST_CMPL;
		break;
	case PFMLIB_POWER7_PMU:
		e->event = POWER7_PME_PM_INST_CMPL;
		break;
	default:
		/* perhaps gen_powerpc_suport.pmu_type wasn't initialized? */
		return PFMLIB_ERR_NOINIT;
	}
	e->num_masks = 0;
	e->unit_masks[0] = 0;
	return 0;
}

/**
 * gen_powerpc_support
 **/
pfm_pmu_support_t gen_powerpc_support = {
	/* the next 6 fields are initialized in pfm_gen_powerpc_pmu_detect */
	.pmu_name		= NULL,
	.pmu_type		= PFMLIB_UNKNOWN_PMU,
	.pme_count		= 0,
	.pmd_count		= 0,
	.pmc_count		= 0,
	.num_cnt		= 0,

	.get_event_code		= pfm_gen_powerpc_get_event_code,
	.get_event_name		= pfm_gen_powerpc_get_event_name,
	.get_event_mask_name	= pfm_gen_powerpc_get_event_mask_name,
	.get_event_counters	= pfm_gen_powerpc_get_event_counters,
	.get_num_event_masks	= pfm_gen_powerpc_get_num_event_masks,
	.dispatch_events	= pfm_gen_powerpc_dispatch_events,
	.pmu_detect		= pfm_gen_powerpc_pmu_detect,
	.get_impl_pmcs		= pfm_gen_powerpc_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_powerpc_get_impl_pmds,
	.get_impl_counters	= pfm_gen_powerpc_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_powerpc_get_hw_counter_width,
	.get_event_desc         = pfm_gen_powerpc_get_event_desc,
	.get_event_mask_desc	= pfm_gen_powerpc_get_event_mask_desc,
	.get_event_mask_code	= pfm_gen_powerpc_get_event_mask_code,
	.get_cycle_event	= pfm_gen_powerpc_get_cycle_event,
	.get_inst_retired_event = pfm_gen_powerpc_get_inst_retired
};
