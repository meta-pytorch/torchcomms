/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Copyright (c) 2006 IBM Corp.
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
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
 * pfmlib_pentium4.c
 *
 * Support for libpfm for the Pentium4/Xeon/EM64T processor family (family=15).
 */

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE /* for getline */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <perfmon/pfmlib_pentium4.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_pentium4_priv.h"
#include "pentium4_events.h"

typedef struct {
	unsigned long addr;
	char *name;
} p4_regmap_t;

#define P4_REGMAP(a, n) { .addr = a, .name = n }

static p4_regmap_t p4_pmc_regmap[]={
/* 0 */ P4_REGMAP(0x3b2, "BPU_ESCR0"),
/* 1 */ P4_REGMAP(0x3ba, "IS_ESCR0"),
/* 2 */ P4_REGMAP(0x3aa, "MOB_ESCR0"),
/* 3 */ P4_REGMAP(0x3b6, "ITLB_ESCR0"),
/* 4 */ P4_REGMAP(0x3ac, "PMH_ESCR0"),
/* 5 */ P4_REGMAP(0x3c8, "IX_ESCR0"),
/* 6 */ P4_REGMAP(0x3a2, "FSB_ESCR0"),
/* 7 */ P4_REGMAP(0x3a0, "BSU_ESCR0"),
/* 8 */ P4_REGMAP(0x3c0, "MS_ESCR0"),
/* 9 */ P4_REGMAP(0x3c4, "TC_ESCR0"),
/* 10 */ P4_REGMAP(0x3c2, "TBPU_ESCR0"),
/* 11 */ P4_REGMAP(0x3a6, "FLAME_ESCR0"),
/* 12 */ P4_REGMAP(0x3a4, "FIRM_ESCR0"),
/* 13 */ P4_REGMAP(0x3ae, "SAAT_ESCR0"),
/* 14 */ P4_REGMAP(0x3b0, "U2L_ESCR0"),
/* 15 */ P4_REGMAP(0x3a8, "DAC_ESCR0"),
/* 16 */ P4_REGMAP(0x3ba, "IQ_ESCR0"),
/* 17 */ P4_REGMAP(0x3ca, "ALF_ESCR0"),
/* 18 */ P4_REGMAP(0x3bc, "RAT_ESCR0"),
/* 19 */ P4_REGMAP(0x3be, "SSU_ESCR0"),
/* 20 */ P4_REGMAP(0x3b8, "CRU_ESCR0"),
/* 21 */ P4_REGMAP(0x3cc, "CRU_ESCR2"),
/* 22 */ P4_REGMAP(0x3e0, "CRU_ESCR4"),
/* 23 */ P4_REGMAP(0x360, "BPU_CCCR0"),
/* 24 */ P4_REGMAP(0x361, "BPU_CCCR1"),
/* 25 */ P4_REGMAP(0x364, "MS_CCCR0"),
/* 26 */ P4_REGMAP(0x365, "MS_CCCR1"),
/* 27 */ P4_REGMAP(0x368, "FLAME_CCCR0"),
/* 28 */ P4_REGMAP(0x369, "FLAME_CCCR1"),
/* 29 */ P4_REGMAP(0x36c, "IQ_CCCR0"),
/* 30 */ P4_REGMAP(0x36d, "IQ_CCCR1"),
/* 31 */ P4_REGMAP(0x370, "IQ_CCCR4"),
/* 32 */ P4_REGMAP(0x3b3, "BPU_ESCR1"),
/* 33 */ P4_REGMAP(0x3b5, "IS_ESCR1"),
/* 34 */ P4_REGMAP(0x3ab, "MOB_ESCR1"),
/* 35 */ P4_REGMAP(0x3b7, "ITLB_ESCR1"),
/* 36 */ P4_REGMAP(0x3ad, "PMH_ESCR1"),
/* 37 */ P4_REGMAP(0x3c9, "IX_ESCR1"),
/* 38 */ P4_REGMAP(0x3a3, "FSB_ESCR1"),
/* 39 */ P4_REGMAP(0x3a1, "BSU_ESCR1"),
/* 40 */ P4_REGMAP(0x3c1, "MS_ESCR1"),
/* 41 */ P4_REGMAP(0x3c5, "TC_ESCR1"),
/* 42 */ P4_REGMAP(0x3c3, "TBPU_ESCR1"),
/* 43 */ P4_REGMAP(0x3a7, "FLAME_ESCR1"),
/* 44 */ P4_REGMAP(0x3a5, "FIRM_ESCR1"),
/* 45 */ P4_REGMAP(0x3af, "SAAT_ESCR1"),
/* 46 */ P4_REGMAP(0x3b1, "U2L_ESCR1"),
/* 47 */ P4_REGMAP(0x3a9, "DAC_ESCR1"),
/* 48 */ P4_REGMAP(0x3bb, "IQ_ESCR1"),
/* 49 */ P4_REGMAP(0x3cb, "ALF_ESCR1"),
/* 50 */ P4_REGMAP(0x3bd, "RAT_ESCR1"),
/* 51 */ P4_REGMAP(0x3b9, "CRU_ESCR1"),
/* 52 */ P4_REGMAP(0x3cd, "CRU_ESCR3"),
/* 53 */ P4_REGMAP(0x3e1, "CRU_ESCR5"),
/* 54 */ P4_REGMAP(0x362, "BPU_CCCR2"),
/* 55 */ P4_REGMAP(0x363, "BPU_CCCR3"),
/* 56 */ P4_REGMAP(0x366, "MS_CCCR2"),
/* 57 */ P4_REGMAP(0x367, "MS_CCCR3"),
/* 58 */ P4_REGMAP(0x36a, "FLAME_CCCR2"),
/* 59 */ P4_REGMAP(0x36b, "FLAME_CCCR3"),
/* 60 */ P4_REGMAP(0x36e, "IQ_CCCR2"),
/* 61 */ P4_REGMAP(0x36f, "IQ_CCCR3"),
/* 62 */ P4_REGMAP(0x371, "IQ_CCCR5"),
/* 63 */ P4_REGMAP(0x3f2, "PEBS_MATRIX_VERT"),
/* 64 */ P4_REGMAP(0x3f1, "PEBS_ENABLE"),
};

#define PMC_PEBS_MATRIX_VERT 63
#define PMC_PEBS_ENABLE		 64

static p4_regmap_t p4_pmd_regmap[]={
/* 0 */ P4_REGMAP(0x300, "BPU_CTR0"),
/* 1 */ P4_REGMAP(0x301, "BPU_CTR1"),
/* 2 */ P4_REGMAP(0x304, "MS_CTR0"),
/* 3 */ P4_REGMAP(0x305, "MS_CTR1"),
/* 4 */ P4_REGMAP(0x308, "FLAME_CTR0"),
/* 5 */ P4_REGMAP(0x309, "FLAME_CTR1"),
/* 6 */ P4_REGMAP(0x30c, "IQ_CTR0"),
/* 7 */ P4_REGMAP(0x30d, "IQ_CTR1"),
/* 8 */ P4_REGMAP(0x310, "IQ_CTR4"),
/* 9 */ P4_REGMAP(0x302, "BPU_CTR2"),
/* 10 */ P4_REGMAP(0x303, "BPU_CTR3"),
/* 11 */ P4_REGMAP(0x306, "MS_CTR2"),
/* 12 */ P4_REGMAP(0x307, "MS_CTR3"),
/* 13 */ P4_REGMAP(0x30a, "FLAME_CTR2"),
/* 14 */ P4_REGMAP(0x30b, "FLAME_CTR3"),
/* 15 */ P4_REGMAP(0x30d, "IQ_CTR2"),
/* 16 */ P4_REGMAP(0x30f, "IQ_CTR3"),
/* 17 */ P4_REGMAP(0x311, "IQ_CTR5"),
};

/* This array provides values for the PEBS_ENABLE and PEBS_MATRIX_VERT
	registers to support a series of metric for replay_event.
	The first two entries are dummies; the remaining 9 correspond to 
	virtual bit masks in the replay_event definition and map onto Intel
	documentation.
*/

#define P4_REPLAY_REAL_MASK 0x00000003
#define P4_REPLAY_VIRT_MASK 0x00000FFC

static pentium4_replay_regs_t p4_replay_regs[]={
/* 0 */ {.enb		= 0,			/* dummy */
		 .mat_vert	= 0,
		},
/* 1 */ {.enb		= 0,			/* dummy */
		 .mat_vert	= 0,
		},
/* 2 */ {.enb		= 0x01000001,	/* 1stL_cache_load_miss_retired */
		 .mat_vert	= 0x00000001,
		},
/* 3 */ {.enb		= 0x01000002,	/* 2ndL_cache_load_miss_retired */
		 .mat_vert	= 0x00000001,
		},
/* 4 */ {.enb		= 0x01000004,	/* DTLB_load_miss_retired */
		 .mat_vert	= 0x00000001,
		},
/* 5 */ {.enb		= 0x01000004,	/* DTLB_store_miss_retired */
		 .mat_vert	= 0x00000002,
		},
/* 6 */ {.enb		= 0x01000004,	/* DTLB_all_miss_retired */
		 .mat_vert	= 0x00000003,
		},
/* 7 */ {.enb		= 0x01018001,	/* Tagged_mispred_branch */
		 .mat_vert	= 0x00000010,
		},
/* 8 */ {.enb		= 0x01000200,	/* MOB_load_replay_retired */
		 .mat_vert	= 0x00000001,
		},
/* 9 */ {.enb		= 0x01000400,	/* split_load_retired */
		 .mat_vert	= 0x00000001,
		},
/* 10 */ {.enb		= 0x01000400,	/* split_store_retired */
		 .mat_vert	= 0x00000002,
		},
};

static int p4_model;

/**
 * pentium4_get_event_code
 *
 * Return the event-select value for the specified event as
 * needed for the specified PMD counter.
 **/
static int pentium4_get_event_code(unsigned int event,
				   unsigned int pmd,
				   int *code)
{
	int i, j, escr, cccr;
	int rc = PFMLIB_ERR_INVAL;

	if (pmd >= PENTIUM4_NUM_PMDS && pmd != PFMLIB_CNT_FIRST) {
		goto out;
	}

	/* Check that the specified event is allowed for the specified PMD.
	 * Each event has a specific set of ESCRs it can use, which implies
	 * a specific set of CCCRs (and thus PMDs). A specified PMD of -1
	 * means assume any allowable PMD.
	 */
	if (pmd == PFMLIB_CNT_FIRST) {
		*code = pentium4_events[event].event_select;
		rc = PFMLIB_SUCCESS;
		goto out;
	}

	for (i = 0; i < MAX_ESCRS_PER_EVENT; i++) {
		escr = pentium4_events[event].allowed_escrs[i];
		if (escr < 0) {
			continue;
		}

		for (j = 0; j < MAX_CCCRS_PER_ESCR; j++) {
			cccr = pentium4_escrs[escr].allowed_cccrs[j];
			if (cccr < 0) {
				continue;
			}

			if (pmd == pentium4_cccrs[cccr].pmd) {
				*code = pentium4_events[event].event_select;
				rc = PFMLIB_SUCCESS;
				goto out;
			}
		}
	}

out:
	return rc;
}

/**
 * pentium4_get_event_name
 *
 * Return the name of the specified event.
 **/
static char *pentium4_get_event_name(unsigned int event)
{
	return pentium4_events[event].name;
}

/**
 * pentium4_get_event_mask_name
 *
 * Return the name of the specified event-mask.
 **/
static char *pentium4_get_event_mask_name(unsigned int event, unsigned int mask)
{
	if (mask >= EVENT_MASK_BITS || pentium4_events[event].event_masks[mask].name == NULL)
		return NULL;

	return pentium4_events[event].event_masks[mask].name;
}

/**
 * pentium4_get_event_counters
 *
 * Fill in the 'counters' bitmask with all possible PMDs that could be
 * used to count the specified event.
 **/
static void pentium4_get_event_counters(unsigned int event,
					pfmlib_regmask_t *counters)
{
	int i, j, escr, cccr;

	memset(counters, 0, sizeof(*counters));

	for (i = 0; i < MAX_ESCRS_PER_EVENT; i++) {
		escr = pentium4_events[event].allowed_escrs[i];
		if (escr < 0) {
			continue;
		}

		for (j = 0; j < MAX_CCCRS_PER_ESCR; j++) {
			cccr = pentium4_escrs[escr].allowed_cccrs[j];
			if (cccr < 0) {
				continue;
			}
			pfm_regmask_set(counters, pentium4_cccrs[cccr].pmd);
		}
	}
}

/**
 * pentium4_get_num_event_masks
 *
 * Count the number of available event-masks for the specified event. All
 * valid masks in pentium4_events[].event_masks are contiguous in the array
 * and have a non-NULL name.
 **/
static unsigned int pentium4_get_num_event_masks(unsigned int event)
{
	unsigned int i = 0;

	while (pentium4_events[event].event_masks[i].name) {
		i++;
	}
	return i;
}

/**
 * pentium4_dispatch_events
 *
 * Examine each desired event specified in "input" and find an appropriate
 * ESCR/CCCR pair that can be used to count them.
 **/
static int pentium4_dispatch_events(pfmlib_input_param_t *input,
				    void *model_input,
				    pfmlib_output_param_t *output,
				    void *model_output)
{
	unsigned int assigned_pmcs[PENTIUM4_NUM_PMCS] = {0};
	unsigned int event, event_mask, mask;
	unsigned int bit, tag_value, tag_enable;
	unsigned int plm;
	unsigned int i, j, k, m, n;
	int escr, escr_pmc;
	int cccr, cccr_pmc, cccr_pmd;
	int assigned;
	pentium4_escr_value_t escr_value;
	pentium4_cccr_value_t cccr_value;

	if (input->pfp_event_count > PENTIUM4_NUM_PMDS) {
		/* Can't specify more events than we have counters. */
		return PFMLIB_ERR_TOOMANY;
	}

	if (input->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		/* Can't specify privilege levels 1 or 2. */
		return PFMLIB_ERR_INVAL;
	}

	/* Examine each event specified in input->pfp_events. i counts
	 * through the input->pfp_events array, and j counts through the
	 * PMCs in output->pfp_pmcs as they are set up.
	 */
	for (i = 0, j = 0; i < input->pfp_event_count; i++) {

		if (input->pfp_events[i].plm & (PFM_PLM1|PFM_PLM2)) {
			/* Can't specify privilege levels 1 or 2. */
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * INSTR_COMPLETED event only exist for model 3, 4, 6 (Prescott)
		 */
		if (input->pfp_events[i].event == PME_INSTR_COMPLETED &&
		    p4_model != 3 && p4_model != 4 && p4_model != 6)
				return PFMLIB_ERR_EVTINCOMP;

		event = input->pfp_events[i].event;
		assigned = 0;

		/* Use the event-specific privilege mask if set.
		 * Otherwise use the default privilege mask.
		 */
		plm = input->pfp_events[i].plm ?
		      input->pfp_events[i].plm : input->pfp_dfl_plm;

		/* Examine each ESCR that this event could be assigned to. */
		for (k = 0; k < MAX_ESCRS_PER_EVENT && !assigned; k++) {
			escr = pentium4_events[event].allowed_escrs[k];
			if (escr < 0)
				continue;

			/* Make sure this ESCR isn't already assigned
			 * and isn't on the "unavailable" list.
			 */
			escr_pmc = pentium4_escrs[escr].pmc;
			if (assigned_pmcs[escr_pmc] ||
			    pfm_regmask_isset(&input->pfp_unavail_pmcs, escr_pmc)) {
				continue;
			}

			/* Examine each CCCR that can be used with this ESCR. */
			for (m = 0; m < MAX_CCCRS_PER_ESCR && !assigned; m++) {
				cccr = pentium4_escrs[escr].allowed_cccrs[m];
				if (cccr < 0) {
					continue;
				}

				/* Make sure this CCCR isn't already assigned
				 * and isn't on the "unavailable" list.
				 */
				cccr_pmc = pentium4_cccrs[cccr].pmc;
				cccr_pmd = pentium4_cccrs[cccr].pmd;
				if (assigned_pmcs[cccr_pmc] ||
				    pfm_regmask_isset(&input->pfp_unavail_pmcs, cccr_pmc)) {
					continue;
				}

				/* Found an available ESCR/CCCR pair. */
				assigned = 1;
				assigned_pmcs[escr_pmc] = 1;
				assigned_pmcs[cccr_pmc] = 1;

				/* Calculate the event-mask value. Invalid masks
				 * specified by the caller are ignored.
				 */
				event_mask = 0;
				tag_value = 0;
				tag_enable = 0;
				for (n = 0; n < input->pfp_events[i].num_masks; n++) {
					mask = input->pfp_events[i].unit_masks[n];
					bit = pentium4_events[event].event_masks[mask].bit;
					if (bit < EVENT_MASK_BITS &&
						pentium4_events[event].event_masks[mask].name) {
						event_mask |= (1 << bit);
					}
					if (bit >= EVENT_MASK_BITS &&
						pentium4_events[event].event_masks[mask].name) {
						tag_value |= (1 << (bit - EVENT_MASK_BITS));
						tag_enable = 1;
					}
				}

				/* Set up the ESCR and CCCR register values. */
				escr_value.val = 0;

				escr_value.bits.t1_usr       = 0; /* controlled by kernel */
				escr_value.bits.t1_os        = 0; /* controlled by kernel */
				escr_value.bits.t0_usr       = (plm & PFM_PLM3) ? 1 : 0;
				escr_value.bits.t0_os        = (plm & PFM_PLM0) ? 1 : 0;
				escr_value.bits.tag_enable   = tag_enable;
				escr_value.bits.tag_value    = tag_value;
				escr_value.bits.event_mask   = event_mask;
				escr_value.bits.event_select = pentium4_events[event].event_select;
				escr_value.bits.reserved     = 0;

				cccr_value.val = 0;

				cccr_value.bits.reserved1     = 0;
				cccr_value.bits.enable        = 1;
				cccr_value.bits.escr_select   = pentium4_events[event].escr_select;
				cccr_value.bits.active_thread = 3; /* FIXME: This is set to count when either logical
								    *        CPU is active. Need a way to distinguish
								    *        between logical CPUs when HT is enabled. */
				cccr_value.bits.compare       = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.complement    = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.threshold     = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.force_ovf     = 0; /* FIXME: Do we want to allow "forcing" overflow
								    *        interrupts on all counter increments? */
				cccr_value.bits.ovf_pmi_t0    = 1;
				cccr_value.bits.ovf_pmi_t1    = 0; /* PMI taken care of by kernel typically */
				cccr_value.bits.reserved2     = 0;
				cccr_value.bits.cascade       = 0; /* FIXME: How do we handle "cascading" counters? */
				cccr_value.bits.overflow      = 0;

				/* Special processing for the replay event:
					Remove virtual mask bits from actual mask;
					scan mask bit list and OR bit values for each virtual mask
					into the PEBS ENABLE and PEBS MATRIX VERT registers */
				if (event == PME_REPLAY_EVENT) {
					escr_value.bits.event_mask &= P4_REPLAY_REAL_MASK;	 /* remove virtual mask bits */
					if (event_mask & P4_REPLAY_VIRT_MASK) {				 /* find a valid virtual mask */
						output->pfp_pmcs[j].reg_value = 0;
						output->pfp_pmcs[j].reg_num = PMC_PEBS_ENABLE;
						output->pfp_pmcs[j].reg_addr = p4_pmc_regmap[PMC_PEBS_ENABLE].addr;
						output->pfp_pmcs[j+1].reg_value = 0;
						output->pfp_pmcs[j+1].reg_num = PMC_PEBS_MATRIX_VERT;
						output->pfp_pmcs[j+1].reg_addr = p4_pmc_regmap[PMC_PEBS_MATRIX_VERT].addr;
						for (n = 0; n < input->pfp_events[i].num_masks; n++) {
							mask = input->pfp_events[i].unit_masks[n];
							if (mask > 1 && mask < 11) { /* process each valid mask we find */
								output->pfp_pmcs[j].reg_value |= p4_replay_regs[mask].enb;
								output->pfp_pmcs[j+1].reg_value |= p4_replay_regs[mask].mat_vert;
							}
						}
						j += 2;
						output->pfp_pmc_count += 2;
					}
				}

				 /* Set up the PMCs in the
				 * output->pfp_pmcs array.
				 */
				output->pfp_pmcs[j].reg_num = escr_pmc;
				output->pfp_pmcs[j].reg_value = escr_value.val;
				output->pfp_pmcs[j].reg_addr = p4_pmc_regmap[escr_pmc].addr;
				j++;

				__pfm_vbprintf("[%s(pmc%u)=0x%lx os=%u usr=%u tag=%u tagval=0x%x mask=%u sel=0x%x] %s\n",
						p4_pmc_regmap[escr_pmc].name,
						escr_pmc,
						escr_value.val,
						escr_value.bits.t0_os,
						escr_value.bits.t0_usr,
						escr_value.bits.tag_enable,
						escr_value.bits.tag_value,
						escr_value.bits.event_mask,
						escr_value.bits.event_select,
						pentium4_events[event].name);

				output->pfp_pmcs[j].reg_num = cccr_pmc;
				output->pfp_pmcs[j].reg_value = cccr_value.val;
				output->pfp_pmcs[j].reg_addr = p4_pmc_regmap[cccr_pmc].addr;

				output->pfp_pmds[i].reg_num = cccr_pmd;
				output->pfp_pmds[i].reg_addr = p4_pmd_regmap[cccr_pmd].addr;

				__pfm_vbprintf("[%s(pmc%u)=0x%lx ena=1 sel=0x%x cmp=%u cmpl=%u thres=%u edg=%u cas=%u] %s\n",
						p4_pmc_regmap[cccr_pmc].name,
						cccr_pmc,
						cccr_value.val,
						cccr_value.bits.escr_select,
						cccr_value.bits.compare,
						cccr_value.bits.complement,
						cccr_value.bits.threshold,
						cccr_value.bits.edge,
						cccr_value.bits.cascade,
						pentium4_events[event].name);
				__pfm_vbprintf("[%s(pmd%u)]\n", p4_pmd_regmap[output->pfp_pmds[i].reg_num].name, output->pfp_pmds[i].reg_num);
				j++;

				output->pfp_pmc_count += 2;
			}
		}
		if (k == MAX_ESCRS_PER_EVENT && !assigned) {
			/* Couldn't find an available ESCR and/or CCCR. */
			return PFMLIB_ERR_NOASSIGN;
		}
	}
	output->pfp_pmd_count = input->pfp_event_count;

	return PFMLIB_SUCCESS;
}

/**
 * pentium4_pmu_detect
 *
 * Determine whether the system we're running on is a Pentium4
 * (or other CPU that uses the same PMU).
 **/
static int pentium4_pmu_detect(void)
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

	ret = __pfm_getcpuinfo_attr("model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * we use model to detect model 2 which has one more counter IQ_ESCR1
	 */
	p4_model = atoi(buffer);
	if (family != 15)
		return PFMLIB_ERR_NOTSUPP;
	/*
	 * IQ_ESCR0, IQ_ESCR1 only for model 1 and 2 
	 */
	if (p4_model >2)
		pentium4_support.pmc_count -= 2;

	return family == 15 ? PFMLIB_SUCCESS : PFMLIB_ERR_NOTSUPP;
}

/**
 * pentium4_get_impl_pmcs
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMC that's
 * available on Pentium4.
 *
 * FIXME: How can we detect when HyperThreading is enabled?
 **/
static void pentium4_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i;

	for(i = 0; i < PENTIUM4_NUM_PMCS; i++) {
		pfm_regmask_set(impl_pmcs, i);
	}
	/*
	 * IQ_ESCR0, IQ_ESCR1 only available on model 1 and 2
	 */
	if (p4_model > 2) {
		pfm_regmask_clr(impl_pmcs, 16);
		pfm_regmask_clr(impl_pmcs, 48);
	}
}

/**
 * pentium4_get_impl_pmds
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMD that's
 * available on Pentium4.
 *
 * FIXME: How can we detect when HyperThreading is enabled?
 **/
static void pentium4_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i;

	for(i = 0; i < PENTIUM4_NUM_PMDS; i++) {
		pfm_regmask_set(impl_pmds, i);
	}
}

/**
 * pentium4_get_impl_counters
 *
 * Set the appropriate bit in the impl_counters bitmask for each counter
 * that's available on Pentium4.
 *
 * For now, all PMDs are counters, so just call get_impl_pmds().
 **/
static void pentium4_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	pentium4_get_impl_pmds(impl_counters);
}

/**
 * pentium4_get_hw_counter_width
 *
 * Return the number of usable bits in the PMD counters.
 **/
static void pentium4_get_hw_counter_width(unsigned int *width)
{
	*width = PENTIUM4_COUNTER_WIDTH;
}

/**
 * pentium4_get_event_desc
 *
 * Return the description for the specified event (if it has one).
 *
 * FIXME: In this routine, we make a copy of the description string to
 *        return. But in get_event_name(), we just return the string
 *        directly. Why the difference?
 **/
static int pentium4_get_event_desc(unsigned int event, char **desc)
{
	if (pentium4_events[event].desc) {
		*desc = strdup(pentium4_events[event].desc);
	} else {
		*desc = NULL;
	}

	return PFMLIB_SUCCESS;
}

/**
 * pentium4_get_event_mask_desc
 *
 * Return the description for the specified event-mask (if it has one).
 **/
static int pentium4_get_event_mask_desc(unsigned int event,
					unsigned int mask, char **desc)
{
	if (mask >= EVENT_MASK_BITS || pentium4_events[event].event_masks[mask].desc == NULL)
		return PFMLIB_ERR_INVAL;

	*desc = strdup(pentium4_events[event].event_masks[mask].desc);

	return PFMLIB_SUCCESS;
}

static int pentium4_get_event_mask_code(unsigned int event,
				 	unsigned int mask, unsigned int *code)
{
	*code = 1U << pentium4_events[event].event_masks[mask].bit;
	return PFMLIB_SUCCESS;
}

static int
pentium4_get_cycle_event(pfmlib_event_t *e)
{
	e->event = PENTIUM4_CPU_CLK_UNHALTED;
	e->num_masks = 1;
	e->unit_masks[0] = 0;
	return PFMLIB_SUCCESS;

}

static int
pentium4_get_inst_retired(pfmlib_event_t *e)
{
	/*
	 * some models do not implement INSTR_COMPLETED
	 */
	if (p4_model != 3 && p4_model != 4 && p4_model != 6) {
		e->event = PENTIUM4_INST_RETIRED;
		e->num_masks = 2;
		e->unit_masks[0] = 0;
		e->unit_masks[1] = 1;
	} else {
		e->event = PME_INSTR_COMPLETED;
		e->num_masks = 1;
		e->unit_masks[0] = 0;
	}
	return PFMLIB_SUCCESS;
}

/**
 * pentium4_support
 **/
pfm_pmu_support_t pentium4_support = {
	.pmu_name		= "Pentium4/Xeon/EM64T",
	.pmu_type		= PFMLIB_PENTIUM4_PMU,
	.pme_count		= PENTIUM4_EVENT_COUNT,
	.pmd_count		= PENTIUM4_NUM_PMDS,
	.pmc_count		= PENTIUM4_NUM_PMCS,
	.num_cnt		= PENTIUM4_NUM_PMDS,
	.get_event_code		= pentium4_get_event_code,
	.get_event_name		= pentium4_get_event_name,
	.get_event_mask_name	= pentium4_get_event_mask_name,
	.get_event_counters	= pentium4_get_event_counters,
	.get_num_event_masks	= pentium4_get_num_event_masks,
	.dispatch_events	= pentium4_dispatch_events,
	.pmu_detect		= pentium4_pmu_detect,
	.get_impl_pmcs		= pentium4_get_impl_pmcs,
	.get_impl_pmds		= pentium4_get_impl_pmds,
	.get_impl_counters	= pentium4_get_impl_counters,
	.get_hw_counter_width	= pentium4_get_hw_counter_width,
	.get_event_desc         = pentium4_get_event_desc,
	.get_event_mask_desc	= pentium4_get_event_mask_desc,
	.get_event_mask_code	= pentium4_get_event_mask_code,
	.get_cycle_event	= pentium4_get_cycle_event,
	.get_inst_retired_event = pentium4_get_inst_retired
};

