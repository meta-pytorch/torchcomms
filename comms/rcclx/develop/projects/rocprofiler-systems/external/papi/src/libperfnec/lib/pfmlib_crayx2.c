/*
 * Copyright (c) 2007 Cray Inc.
 * Contributed by Steve Kaufmann <sbk@cray.com> based on code from
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

#include <string.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_crayx2.h>

#include "pfmlib_priv.h"
#include "pfmlib_crayx2_priv.h"
#include "crayx2_events.h"

#define CRAYX2_NO_REDUNDANT 0	/* if>0 an error if chip:ctr:ev repeated */

typedef enum {
	CTR_REDUNDANT = -2,	/* event on counter repeated */
	CTR_CONFLICT = -1,	/* event on counter not the same as previous */
	CTR_OK = 0		/* event on counter open */
} counter_use_t;

static int
pfm_crayx2_get_event_code (unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > crayx2_support.num_cnt) {
		DPRINT ("return: count %d exceeded #counters\n", cnt);
		return PFMLIB_ERR_INVAL;
	} else if (i >= crayx2_support.pme_count) {
		DPRINT ("return: event index %d exceeded #events\n", i);
		return PFMLIB_ERR_INVAL;
	}
	*code = crayx2_pe[i].pme_code;
	DPRINT ("return: event code is %#x\n", *code);

	return PFMLIB_SUCCESS;
}

static char *
pfm_crayx2_get_event_name (unsigned int i)
{
	if (i >= crayx2_support.pme_count) {
		DPRINT ("return: event index %d exceeded #events\n", i);
		return NULL;
	}
	DPRINT ("return: event name '%s'\n", crayx2_pe[i].pme_name);

	return (char *) crayx2_pe[i].pme_name;
}

static void
pfm_crayx2_get_event_counters (unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset (counters, 0, sizeof (*counters));

	DPRINT ("event counters for %d counters\n", PMU_CRAYX2_NUM_COUNTERS);
	for (i=0; i<PMU_CRAYX2_NUM_COUNTERS; i++) {
		pfm_regmask_set (counters, i);
	}
	return;
}

static int
pfm_crayx2_chip_use (uint32_t used[ ], unsigned int n)
{
	int i, u = 0;

	for (i=0; i<n; i++) {
		u += pfmlib_popcnt (used[i]);
	}
	DPRINT ("number of counters used on chip %d\n", u);

	return u;
}

static counter_use_t
pfm_crayx2_counter_use (unsigned int ctr, unsigned int event, uint32_t *used, uint64_t *evmsk)
{
	counter_use_t ret = CTR_OK;

	if (*used & (1 << ctr)) {
		if (event == PFM_EVENT_GET (*evmsk, ctr)) {
			ret = CTR_REDUNDANT;
		} else {
			ret = CTR_CONFLICT;
		}
	} else {
		*evmsk |= PFM_EVENT_SET (ctr, event);
		*used |= (1 << ctr);
	}
	return ret;
}

static int
pfm_crayx2_dispatch_events (pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	unsigned int i, npmcs = 0, npmds = 0, base_pmc = 0;
	uint32_t Pused[PME_CRAYX2_CPU_CHIPS];
	uint32_t Cused[PME_CRAYX2_CACHE_CHIPS];
	uint32_t Mused[PME_CRAYX2_MEMORY_CHIPS];
	uint64_t Pevents = 0, Cevents = 0, Mevents = 0;

	DPRINT ("dispatching event info to the PMCs and PMDs\n");

	/*	NOTES:
	 *	Multiplexing is not supported on X2.
	 *	The priviledge level is ignored for the C and M chips.
	 *	The priviledge level is ignored per event.
	 */

	if (PFMLIB_DEBUG ( )) {
		int j;
		DPRINT ("input: pfp_event_count %d pfp_dfl_plm %#x pfp_flags %#x\n", inp->pfp_event_count, inp->pfp_dfl_plm, inp->pfp_flags);
		for (i=0; i<inp->pfp_event_count; i++) {
			DPRINT (" %3d: event %3d plm %#3x flags %#8lx num_masks %d\n", i, inp->pfp_events[i].event, inp->pfp_events[i].plm, inp->pfp_events[i].flags, inp->pfp_events[i].num_masks);
			for (j=0; j<inp->pfp_events[i].num_masks; j++) {
				DPRINT (" unit-mask-%2d: %d\n", j, inp->pfp_events[i].unit_masks[j]);
			}
		}
	}

	/*	Better have at least one event specified and not exceed limit.
	 */
	if (inp->pfp_event_count == 0) {
		DPRINT ("return: event count is 0\n");
		return PFMLIB_ERR_INVAL;
	} else if (inp->pfp_event_count > PMU_CRAYX2_NUM_COUNTERS) {
		DPRINT ("return: event count exceeds max %d\n", PMU_CRAYX2_NUM_COUNTERS);
		return PFMLIB_ERR_TOOMANY;
	}

	memset (Pused, 0, sizeof(Pused));
	memset (Cused, 0, sizeof(Cused));
	memset (Mused, 0, sizeof(Mused));

	/*	Loop through the input parameters describing the events.
	 */
	for (i=0; i<inp->pfp_event_count; i++) {
		unsigned int code, chip, ctr, ev, chipno;
		counter_use_t ret;

		/*	Acquire details describing this event code:
		 *	o which substrate/chip it is on
		 *	o which counter on the chip
		 *	o which event on the counter
		 */
		code = inp->pfp_events[i].event;
		chip = crayx2_pe[code].pme_chip;
		ctr = crayx2_pe[code].pme_ctr;
		ev = crayx2_pe[code].pme_event;
		chipno = crayx2_pe[code].pme_chipno;

		DPRINT ("%3d: code %3d chip %1d ctr %2d ev %1d chipno %2d\n", code, i, chip, ctr, ev, chipno);

		/*	These priviledge levels are not recognized.
		 */
		if (inp->pfp_events[i].plm != 0) {
			DPRINT ("%3d: priviledge level %#x per event not allowed\n", i, inp->pfp_events[i].plm);
			return PFMLIB_ERR_INVAL;
		}

		/*	No masks exist.
		 */
		if (inp->pfp_events[i].num_masks > 0) {
			DPRINT ("too many masks for event\n");
			return PFMLIB_ERR_TOOMANY;
		}

		/*	The event code. Set-up the event selection mask for
		 *	the PMC of the respective chip. Check if more than
		 *	one event on the same counter is selected.
		 */
		if (chip == PME_CRAYX2_CHIP_CPU) {
			ret = pfm_crayx2_counter_use (ctr, ev, &Pused[chipno], &Pevents);
		} else if (chip == PME_CRAYX2_CHIP_CACHE) {
			ret = pfm_crayx2_counter_use (ctr, ev, &Cused[chipno], &Cevents);
		} else if (chip == PME_CRAYX2_CHIP_MEMORY) {
			ret = pfm_crayx2_counter_use (ctr, ev, &Mused[chipno], &Mevents);
		} else {
			DPRINT ("return: invalid chip\n");
			return PFMLIB_ERR_INVAL;
		}

		/*	Each chip's counter can only count one event.
		 */
		if (ret == CTR_CONFLICT) {
			DPRINT ("return: ctr conflict\n");
			return PFMLIB_ERR_EVTINCOMP;
		} else if (ret == CTR_REDUNDANT) {
#if (CRAYX2_NO_REDUNDANT != 0)
			DPRINT ("return: ctr redundant\n");
			return PFMLIB_ERR_EVTMANY;
#else
			DPRINT ("warning: ctr redundant\n");
#endif /* CRAYX2_NO_REDUNDANT */
		}

		/*	Set up the output PMDs.
		 */
		outp->pfp_pmds[npmds].reg_num = crayx2_pe[code].pme_base + ctr + chipno*crayx2_pe[code].pme_nctrs;
		outp->pfp_pmds[npmds].reg_addr = 0;
		outp->pfp_pmds[npmds].reg_alt_addr = 0;
		outp->pfp_pmds[npmds].reg_value = 0;
		npmds++;
	}
	outp->pfp_pmd_count = npmds;

	if (PFMLIB_DEBUG ( )) {
		DPRINT ("P event mask %#16lx\n", Pevents);
		DPRINT ("C event mask %#16lx\n", Cevents);
		DPRINT ("M event mask %#16lx\n", Mevents);
		DPRINT ("PMDs: pmd_count %d\n", outp->pfp_pmd_count);
		for (i=0; i<outp->pfp_pmd_count; i++) {
			DPRINT (" %3d: reg_value %3lld reg_num %3d reg_addr %#16llx\n", i, outp->pfp_pmds[i].reg_value, outp->pfp_pmds[i].reg_num, outp->pfp_pmds[i].reg_addr);
		}
	}

	/*	Set up the PMC basics for the chips that will be doing
	 *	some counting.
	 */
	if (pfm_crayx2_chip_use (Pused, PME_CRAYX2_CPU_CHIPS) > 0) {
		uint64_t Pctrl = PFM_CPU_START;
		uint64_t Pen = PFM_ENABLE_RW;

		if (inp->pfp_dfl_plm & (PFM_PLM0 | PFM_PLM1)) {
			Pen |= PFM_ENABLE_KERNEL;
		}
		if (inp->pfp_dfl_plm & PFM_PLM2) {
			Pen |= PFM_ENABLE_EXL;
		}
		if (inp->pfp_dfl_plm & PFM_PLM3) {
			Pen |= PFM_ENABLE_USER;
		}

		/*	First of three CPU PMC registers.
		 */
		base_pmc = PMU_CRAYX2_CPU_PMC_BASE;

		outp->pfp_pmcs[npmcs].reg_value = Pctrl;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_CONTROL;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Pevents;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_EVENTS;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Pen;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_ENABLE;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;
	}
	if (pfm_crayx2_chip_use (Cused, PME_CRAYX2_CACHE_CHIPS) > 0) {
		uint64_t Cctrl = PFM_CACHE_START;
		uint64_t Cen = PFM_ENABLE_RW;		/* domains N/A */

		/*	Second of three Cache PMC registers.
		 */
		base_pmc = PMU_CRAYX2_CACHE_PMC_BASE;

		outp->pfp_pmcs[npmcs].reg_value = Cctrl;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_CONTROL;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Cevents;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_EVENTS;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Cen;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_ENABLE;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;
	}
	if (pfm_crayx2_chip_use (Mused, PME_CRAYX2_MEMORY_CHIPS) > 0) {
		uint64_t Mctrl = PFM_MEM_START;
		uint64_t Men = PFM_ENABLE_RW;		/* domains N/A */

		/*	Third of three Memory PMC registers.
		 */
		base_pmc = PMU_CRAYX2_MEMORY_PMC_BASE;

		outp->pfp_pmcs[npmcs].reg_value = Mctrl;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_CONTROL;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Mevents;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_EVENTS;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;

		outp->pfp_pmcs[npmcs].reg_value = Men;
		outp->pfp_pmcs[npmcs].reg_num = base_pmc + PMC_ENABLE;
		outp->pfp_pmcs[npmcs].reg_addr = 0;
		outp->pfp_pmcs[npmcs].reg_alt_addr = 0;
		npmcs++;
	}
	outp->pfp_pmc_count = npmcs;

	if (PFMLIB_DEBUG ( )) {
		DPRINT ("PMCs: pmc_count %d\n", outp->pfp_pmc_count);
		for (i=0; i<outp->pfp_pmc_count; i++) {
			DPRINT (" %3d: reg_value %#16llx reg_num %3d reg_addr %#16llx\n", i, outp->pfp_pmcs[i].reg_value, outp->pfp_pmcs[i].reg_num, outp->pfp_pmcs[i].reg_addr);
		}
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_crayx2_pmu_detect (void)
{
	char buffer[128];
	int ret;

	DPRINT ("detect the PMU attributes\n");

	ret = __pfm_getcpuinfo_attr ("vendor_id", buffer, sizeof(buffer));

	if (ret != 0 || strcasecmp (buffer, "Cray") != 0) {
		DPRINT ("return: no 'Cray' vendor_id\n");
		return PFMLIB_ERR_NOTSUPP;
	}

	ret = __pfm_getcpuinfo_attr ("type", buffer, sizeof(buffer));

	if (ret != 0 || strcasecmp (buffer, "craynv2") != 0) {
		DPRINT ("return: no 'craynv2' type\n");
		return PFMLIB_ERR_NOTSUPP;
	}

	DPRINT ("Cray X2 nv2 found\n");

	return PFMLIB_SUCCESS;
}

static void
pfm_crayx2_get_impl_pmcs (pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i;

	DPRINT ("entered with PMC_COUNT %d\n", PMU_CRAYX2_PMC_COUNT);
	for (i=0; i<PMU_CRAYX2_PMC_COUNT; i++) {
		pfm_regmask_set (impl_pmcs, i);
	}

	return;
}

static void
pfm_crayx2_get_impl_pmds (pfmlib_regmask_t *impl_pmds)
{
	unsigned int i;

	DPRINT ("entered with PMD_COUNT %d\n", PMU_CRAYX2_PMD_COUNT);
	for (i=0; i<PMU_CRAYX2_PMD_COUNT; i++) {
		pfm_regmask_set (impl_pmds, i);
	}

	return;
}

static void
pfm_crayx2_get_impl_counters (pfmlib_regmask_t *impl_counters)
{
	unsigned int i;

	DPRINT ("entered with NUM_COUNTERS %d\n", PMU_CRAYX2_NUM_COUNTERS);
	for (i=0; i<PMU_CRAYX2_NUM_COUNTERS; i++) {
		pfm_regmask_set (impl_counters, i);
	}

	return;
}

static void
pfm_crayx2_get_hw_counter_width (unsigned int *width)
{
	*width = PMU_CRAYX2_COUNTER_WIDTH;
	DPRINT ("return: width set to %d\n", *width);

	return;
}

static int
pfm_crayx2_get_event_desc (unsigned int ev, char **str)
{
	const char *s = crayx2_pe[ev].pme_desc;

	*str = (s == NULL ? NULL : strdup (s));
	DPRINT ("return: event description is '%s'\n", (s == NULL ? "" : s));

	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_crayx2_get_num_event_masks (unsigned int ev)
{
	DPRINT ("return: #event masks is %d\n", crayx2_pe[ev].pme_numasks);
	return crayx2_pe[ev].pme_numasks;
}

static char *
pfm_crayx2_get_event_mask_name (unsigned int ev, unsigned int midx)
{
	DPRINT ("return: event mask name is '%s'\n", crayx2_pe[ev].pme_umasks[midx].pme_uname);
	return (char *) crayx2_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_crayx2_get_event_mask_code (unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code = crayx2_pe[ev].pme_umasks[midx].pme_ucode;
	DPRINT ("return: event mask code is %#x\n", *code);

	return PFMLIB_SUCCESS;
}

static int
pfm_crayx2_get_event_mask_desc (unsigned int ev, unsigned int midx, char **str)
{
	const char *s = crayx2_pe[ev].pme_umasks[midx].pme_udesc;

	*str = (s == NULL ? NULL : strdup (s));
	DPRINT ("return: event mask description is '%s'\n", (s == NULL ? "" : s));

	return PFMLIB_SUCCESS;
}

static int
pfm_crayx2_get_cycle_event (pfmlib_event_t *e)
{
	e->event = PME_CRAYX2_CYCLES;
	DPRINT ("return: event code for cycles %#x\n", e->event);

	return PFMLIB_SUCCESS;
}

static int
pfm_crayx2_get_inst_retired (pfmlib_event_t *e)
{
	e->event = PME_CRAYX2_INSTR_GRADUATED;
	DPRINT ("return: event code for retired instr %#x\n", e->event);

	return PFMLIB_SUCCESS;
}


/*	Register the constants and the access functions.
 */
pfm_pmu_support_t crayx2_support = {
	.pmu_name		= PMU_CRAYX2_NAME,
	.pmu_type		= PFMLIB_CRAYX2_PMU,
	.pme_count		= PME_CRAYX2_EVENT_COUNT,
	.pmc_count		= PMU_CRAYX2_PMC_COUNT,
	.pmd_count		= PMU_CRAYX2_PMD_COUNT,
	.num_cnt		= PMU_CRAYX2_NUM_COUNTERS,
	.get_event_code		= pfm_crayx2_get_event_code,
	.get_event_name		= pfm_crayx2_get_event_name,
	.get_event_counters	= pfm_crayx2_get_event_counters,
	.dispatch_events	= pfm_crayx2_dispatch_events,
	.pmu_detect		= pfm_crayx2_pmu_detect,
	.get_impl_pmcs		= pfm_crayx2_get_impl_pmcs,
	.get_impl_pmds		= pfm_crayx2_get_impl_pmds,
	.get_impl_counters	= pfm_crayx2_get_impl_counters,
	.get_hw_counter_width	= pfm_crayx2_get_hw_counter_width,
	.get_event_desc		= pfm_crayx2_get_event_desc,
	.get_num_event_masks	= pfm_crayx2_get_num_event_masks,
	.get_event_mask_name	= pfm_crayx2_get_event_mask_name,
	.get_event_mask_code	= pfm_crayx2_get_event_mask_code,
	.get_event_mask_desc	= pfm_crayx2_get_event_mask_desc,
	.get_cycle_event	= pfm_crayx2_get_cycle_event,
	.get_inst_retired_event = pfm_crayx2_get_inst_retired
};
