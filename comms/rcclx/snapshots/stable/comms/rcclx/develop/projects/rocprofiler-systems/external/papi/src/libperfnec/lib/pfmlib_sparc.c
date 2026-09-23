/*
 * Copyright (C) 2007 David S. Miller (davem@davemloft.net)
 *
 * Based upon gen_powerpc code which is:
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
 * pfmlib_sparc.c
 *
 * Support for libpfm for Sparc processors.
 */

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE /* for getline */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* private headers */
#include "pfmlib_priv.h"

#include "pfmlib_sparc_priv.h"

#include "ultra12_events.h"
#include "ultra3_events.h"
#include "ultra3i_events.h"
#include "ultra3plus_events.h"
#include "ultra4plus_events.h"
#include "niagara1_events.h"
#include "niagara2_events.h"

static char *get_event_name(int event)
{
	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
		return ultra12_pe[event].pme_name;
	case PFMLIB_SPARC_ULTRA3_PMU:
		return ultra3_pe[event].pme_name;
	case PFMLIB_SPARC_ULTRA3I_PMU:
		return ultra3i_pe[event].pme_name;
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
		return ultra3plus_pe[event].pme_name;
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		return ultra4plus_pe[event].pme_name;
	case PFMLIB_SPARC_NIAGARA1_PMU:
		return niagara1_pe[event].pme_name;
	case PFMLIB_SPARC_NIAGARA2_PMU:
		return niagara2_pe[event].pme_name;
	}
	return (char *)-1;
}

static char *get_event_desc(int event)
{
	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
		return ultra12_pe[event].pme_desc;
	case PFMLIB_SPARC_ULTRA3_PMU:
		return ultra3_pe[event].pme_desc;
	case PFMLIB_SPARC_ULTRA3I_PMU:
		return ultra3i_pe[event].pme_desc;
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
		return ultra3plus_pe[event].pme_desc;
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		return ultra4plus_pe[event].pme_desc;
	case PFMLIB_SPARC_NIAGARA1_PMU:
		return niagara1_pe[event].pme_desc;
	case PFMLIB_SPARC_NIAGARA2_PMU:
		return niagara2_pe[event].pme_desc;
	}
	return (char *)-1;
}

static char get_ctrl(int event)
{
	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
		return ultra12_pe[event].pme_ctrl;
	case PFMLIB_SPARC_ULTRA3_PMU:
		return ultra3_pe[event].pme_ctrl;
	case PFMLIB_SPARC_ULTRA3I_PMU:
		return ultra3i_pe[event].pme_ctrl;
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
		return ultra3plus_pe[event].pme_ctrl;
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		return ultra4plus_pe[event].pme_ctrl;
	case PFMLIB_SPARC_NIAGARA1_PMU:
		return niagara1_pe[event].pme_ctrl;
	case PFMLIB_SPARC_NIAGARA2_PMU:
		return niagara2_pe[event].pme_ctrl;
	}
	return 0xff;
}

static int get_val(int event)
{
	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
		return ultra12_pe[event].pme_val;
	case PFMLIB_SPARC_ULTRA3_PMU:
		return ultra3_pe[event].pme_val;
	case PFMLIB_SPARC_ULTRA3I_PMU:
		return ultra3i_pe[event].pme_val;
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
		return ultra3plus_pe[event].pme_val;
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		return ultra4plus_pe[event].pme_val;
	case PFMLIB_SPARC_NIAGARA1_PMU:
		return niagara1_pe[event].pme_val;
	case PFMLIB_SPARC_NIAGARA2_PMU:
		return niagara2_pe[event].pme_val;
	}
	return -1;
}

static int pfm_sparc_get_event_code(unsigned int event,
				    unsigned int pmd,
				    int *code)
{
        *code = get_val(event);
	return 0;
}

static char *pfm_sparc_get_event_name(unsigned int event)
{
	return get_event_name(event);
}

static char *pfm_sparc_get_event_mask_name(unsigned int event,
					   unsigned int mask)
{
	pme_sparc_mask_entry_t *e;

	if (sparc_support.pmu_type != PFMLIB_SPARC_NIAGARA2_PMU)
		return "";

	e = &niagara2_pe[event];
	return e->pme_masks[mask].mask_name;
}

static void pfm_sparc_get_event_counters(unsigned int event,
					 pfmlib_regmask_t *counters)
{
	if (sparc_support.pmu_type == PFMLIB_SPARC_NIAGARA2_PMU) {
		counters->bits[0] = (1 << 0) | (1 << 1);
	} else {
		char ctrl = get_ctrl(event);

		counters->bits[0] = 0;	
		if (ctrl & PME_CTRL_S0)
			counters->bits[0] |= (1 << 0);
		if (ctrl & PME_CTRL_S1)
			counters->bits[0] |= (1 << 1);
	}
}

static unsigned int pfm_sparc_get_num_event_masks(unsigned int event)
{
	if (sparc_support.pmu_type != PFMLIB_SPARC_NIAGARA2_PMU)
		return 0;
	return (event == 0 ? 0 : EVENT_MASK_BITS);
}

/* Bits common to all PCR implementations */
#define PCR_PRIV	(0x1UL << 0)
#define PCR_SYS_TRACE	(0x1UL << 1)
#define PCR_USER_TRACE	(0x1UL << 2)

/* The S0 and S1 fields determine which events are monitored in
 * the assosciated PIC (PIC0 vs. PIC1 respectively).  For ultra12
 * these fields are 4 bits, on ultra3/3i/3+/4+ they are 6 bits.
 * For Niagara-1 there is only S0 and it is 3 bits in size.
 * Niagara-1's PIC1 is hard-coded to record retired instructions.
 */
#define PCR_S0_SHIFT		4
#define PCR_S0			(0x1fUL << PCR_S0_SHIFT)
#define PCR_S1_SHIFT		11
#define PCR_S1			(0x1fUL << PCR_S1_SHIFT)

/* Niagara-2 specific PCR bits.  It supports event masking.  */
#define PCR_N2_HYP_TRACE	(0x1UL << 3)
#define PCR_N2_TOE0		(0x1UL << 4)
#define PCR_N2_TOE1		(0x1UL << 5)
#define PCR_N2_SL0_SHIFT	14
#define PCR_N2_SL0		(0xf << PCR_N2_SL0_SHIFT)
#define PCR_N2_MASK0_SHIFT	6
#define PCR_N2_MASK0		(0xff << PCR_N2_MASK0_SHIFT)
#define PCR_N2_SL1_SHIFT	27
#define PCR_N2_SL1		(0xf << PCR_N2_SL1_SHIFT)
#define PCR_N2_MASK1_SHIFT	19
#define PCR_N2_MASK1		(0xff << PCR_N2_MASK1_SHIFT)

static int pfm_sparc_dispatch_events(pfmlib_input_param_t *input,
				     void *model_input,
				     pfmlib_output_param_t *output,
				     void *model_output)
{
	unsigned long long pcr, vals[2];
	unsigned int plm, i;
	int niagara2;
	char ctrls[2];

	if (input->pfp_event_count > 2)
		return PFMLIB_ERR_TOOMANY;

	plm = ((input->pfp_events[0].plm != 0) ?
	       input->pfp_events[0].plm :
	       input->pfp_dfl_plm);
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

	niagara2 = 0;
	if (sparc_support.pmu_type == PFMLIB_SPARC_NIAGARA2_PMU)
		niagara2 = 1;

	pcr = 0;
	if (plm & PFM_PLM3)
		pcr |= PCR_USER_TRACE;
	if (plm & PFM_PLM0)
		pcr |= PCR_SYS_TRACE;
	if (niagara2 && (plm & PFM_PLM1))
		pcr |= PCR_N2_HYP_TRACE;

	for (i = 0; i < input->pfp_event_count; i++) {
		pfmlib_event_t *e = &input->pfp_events[i];

		ctrls[i] = get_ctrl(e->event);
		vals[i] = get_val(e->event);

		if (i == 1) {
			if ((ctrls[0] & ctrls[1]) == 0)
				continue;

			if (ctrls[0] == (PME_CTRL_S0|PME_CTRL_S1)) {
				if (ctrls[1] == (PME_CTRL_S0|PME_CTRL_S1)) {
					ctrls[0] = PME_CTRL_S0;
					ctrls[1] = PME_CTRL_S1;
				} else {
					ctrls[0] &= ~ctrls[1];
				}
			} else if (ctrls[1] == (PME_CTRL_S0|PME_CTRL_S1)) {
				ctrls[1] &= ~ctrls[0];
			} else
				return PFMLIB_ERR_INVAL;
		}
	}

	if (input->pfp_event_count == 1) {
		if (ctrls[0] == (PME_CTRL_S0|PME_CTRL_S1))
			ctrls[0] = PME_CTRL_S0;
	}

	for (i = 0; i < input->pfp_event_count; i++) {
		unsigned long long val = vals[i];
		char ctrl = ctrls[i];

		switch (ctrl) {
		case PME_CTRL_S0:
			output->pfp_pmds[i].reg_num = 0;
			pcr |= (val <<
				(niagara2 ?
				 PCR_N2_SL0_SHIFT :
				 PCR_S0_SHIFT));
			break;

		case PME_CTRL_S1:
			output->pfp_pmds[i].reg_num = 1;
			pcr |= (val <<
				(niagara2 ?
				 PCR_N2_SL1_SHIFT :
				 PCR_S1_SHIFT));
			break;

		default:
			return PFMLIB_ERR_INVAL;
		}
		if (niagara2) {
			pfmlib_event_t *e = &input->pfp_events[i];
			unsigned int j, shift;

			if (ctrl == PME_CTRL_S0) {
				pcr |= PCR_N2_TOE0;
				shift = PCR_N2_MASK0_SHIFT;
			} else {
				pcr |= PCR_N2_TOE1;
				shift = PCR_N2_MASK1_SHIFT;
			}
			for (j = 0; j < e->num_masks; j++) {
				unsigned int mask;

				mask = e->unit_masks[j];
				if (mask >= EVENT_MASK_BITS)
					return PFMLIB_ERR_INVAL;
				pcr |= (1ULL << (shift + mask));
			}
		}

		output->pfp_pmds[i].reg_value = 0;
		output->pfp_pmds[i].reg_addr = 0;
		output->pfp_pmds[i].reg_alt_addr = 0;
		output->pfp_pmds[i].reg_reserved1 = 0;
		output->pfp_pmd_count = i + 1;
	}

	output->pfp_pmcs[0].reg_value = pcr;
	output->pfp_pmcs[0].reg_addr = 0;
	output->pfp_pmcs[0].reg_num = 0;
	output->pfp_pmcs[0].reg_reserved1 = 0;
	output->pfp_pmc_count = 1;
		
	return PFMLIB_SUCCESS;
}

static int pmu_name_to_pmu_type(char *name)
{
	if (!strcmp(name, "ultra12"))
		return PFMLIB_SPARC_ULTRA12_PMU;
	if (!strcmp(name, "ultra3"))
		return PFMLIB_SPARC_ULTRA3_PMU;
	if (!strcmp(name, "ultra3i"))
		return PFMLIB_SPARC_ULTRA3I_PMU;
	if (!strcmp(name, "ultra3+"))
		return PFMLIB_SPARC_ULTRA3PLUS_PMU;
	if (!strcmp(name, "ultra4+"))
		return PFMLIB_SPARC_ULTRA4PLUS_PMU;
	if (!strcmp(name, "niagara2"))
		return PFMLIB_SPARC_NIAGARA2_PMU;
	if (!strcmp(name, "niagara"))
		return PFMLIB_SPARC_NIAGARA1_PMU;
	return -1;
}

static int pfm_sparc_pmu_detect(void)
{
	int ret, pmu_type, pme_count;
	char buffer[32];

	ret = __pfm_getcpuinfo_attr("pmu", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	pmu_type = pmu_name_to_pmu_type(buffer);
	if (pmu_type == -1)
		return PFMLIB_ERR_NOTSUPP;

	switch (pmu_type) {
	default:
		return PFMLIB_ERR_NOTSUPP;

	case PFMLIB_SPARC_ULTRA12_PMU:
		pme_count = PME_ULTRA12_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_ULTRA3_PMU:
		pme_count = PME_ULTRA3_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_ULTRA3I_PMU:
		pme_count = PME_ULTRA3I_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
		pme_count = PME_ULTRA3PLUS_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		pme_count = PME_ULTRA4PLUS_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_NIAGARA1_PMU:
		pme_count = PME_NIAGARA1_EVENT_COUNT;
		break;

	case PFMLIB_SPARC_NIAGARA2_PMU:
		pme_count = PME_NIAGARA2_EVENT_COUNT;
		break;
	}

	sparc_support.pmu_type = pmu_type;
	sparc_support.pmu_name = strdup(buffer);
	sparc_support.pme_count = pme_count;

	return PFMLIB_SUCCESS;
}

static void pfm_sparc_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
        impl_pmcs->bits[0] = 0x1;
}

static void pfm_sparc_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
        impl_pmds->bits[0] = 0x3;
}

static void pfm_sparc_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	pfm_sparc_get_impl_pmds(impl_counters);
}

static void pfm_sparc_get_hw_counter_width(unsigned int *width)
{
	*width = 32;
}

static int pfm_sparc_get_event_desc(unsigned int event, char **desc)
{
	*desc = strdup(get_event_desc(event));
	return 0;
}

static int pfm_sparc_get_event_mask_desc(unsigned int event,
					 unsigned int mask, char **desc)
{
	if (sparc_support.pmu_type != PFMLIB_SPARC_NIAGARA2_PMU) {
		*desc = strdup("");
	} else {
		pme_sparc_mask_entry_t *e;

		e = &niagara2_pe[event];
		*desc = strdup(e->pme_masks[mask].mask_desc);
	}
	return 0;
}

static int pfm_sparc_get_event_mask_code(unsigned int event,
					 unsigned int mask, unsigned int *code)
{
	if (sparc_support.pmu_type != PFMLIB_SPARC_NIAGARA2_PMU)
		*code = 0;
	else
		*code = mask;
	return 0;
}

static int
pfm_sparc_get_cycle_event(pfmlib_event_t *e)
{
	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
	case PFMLIB_SPARC_ULTRA3_PMU:
	case PFMLIB_SPARC_ULTRA3I_PMU:
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		e->event = 0;
		break;

	case PFMLIB_SPARC_NIAGARA1_PMU:
	case PFMLIB_SPARC_NIAGARA2_PMU:
	default:
		return PFMLIB_ERR_NOTSUPP;
	}

	return PFMLIB_SUCCESS;
}

static int
pfm_sparc_get_inst_retired(pfmlib_event_t *e)
{
	unsigned int i;

	switch (sparc_support.pmu_type) {
	case PFMLIB_SPARC_ULTRA12_PMU:
	case PFMLIB_SPARC_ULTRA3_PMU:
	case PFMLIB_SPARC_ULTRA3I_PMU:
	case PFMLIB_SPARC_ULTRA3PLUS_PMU:
	case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		e->event = 1;
		break;

	case PFMLIB_SPARC_NIAGARA1_PMU:
		e->event = 0;
		break;

	case PFMLIB_SPARC_NIAGARA2_PMU:
		e->event = 1;
		e->num_masks = EVENT_MASK_BITS;
		for (i = 0; i < e->num_masks; i++)
			e->unit_masks[i] = i;
		break;

	default:
		return PFMLIB_ERR_NOTSUPP;
	}
	return PFMLIB_SUCCESS;
}

/**
 * sparc_support
 **/
pfm_pmu_support_t sparc_support = {
	/* the next 3 fields are initialized in pfm_sparc_pmu_detect */
	.pmu_name		= NULL,
	.pmu_type		= PFMLIB_UNKNOWN_PMU,
	.pme_count		= 0,

	.pmd_count		= 2,
	.pmc_count		= 1,
	.num_cnt		= 2,

	.get_event_code		= pfm_sparc_get_event_code,
	.get_event_name		= pfm_sparc_get_event_name,
	.get_event_mask_name	= pfm_sparc_get_event_mask_name,
	.get_event_counters	= pfm_sparc_get_event_counters,
	.get_num_event_masks	= pfm_sparc_get_num_event_masks,
	.dispatch_events	= pfm_sparc_dispatch_events,
	.pmu_detect		= pfm_sparc_pmu_detect,
	.get_impl_pmcs		= pfm_sparc_get_impl_pmcs,
	.get_impl_pmds		= pfm_sparc_get_impl_pmds,
	.get_impl_counters	= pfm_sparc_get_impl_counters,
	.get_hw_counter_width	= pfm_sparc_get_hw_counter_width,
	.get_event_desc         = pfm_sparc_get_event_desc,
	.get_event_mask_desc	= pfm_sparc_get_event_mask_desc,
	.get_event_mask_code	= pfm_sparc_get_event_mask_code,
	.get_cycle_event	= pfm_sparc_get_cycle_event,
	.get_inst_retired_event = pfm_sparc_get_inst_retired
};
