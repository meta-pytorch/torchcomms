/*
 * Cell PMU specific types and definitions
 *
 * Copyright (c) 2007 TOSHIBA CORPORATION based on code from
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
#ifndef __PFMLIB_CELL_H__
#define __PFMLIB_CELL_H__

#include <perfmon/pfmlib.h>

#define PMU_CELL_NUM_COUNTERS		8	/* total number of EvtSel/EvtCtr */
#define PMU_CELL_NUM_PERFSEL		8	/* total number of EvtSel */
#define PMU_CELL_NUM_PERFCTR		8	/* total number of EvtCtr */

typedef struct {
	unsigned int pmX_control_num;	/* for pmX_control X=1(pm0_control)...X=8(pm7_control) */
	unsigned int spe_subunit;
	unsigned int polarity;
	unsigned int input_control;
	unsigned int cnt_mask;		/* threshold (reserved) */
	unsigned int flags;		/* counter specific flag (reserved) */
} pfmlib_cell_counter_t;

/*
 * Cell specific parameters for the library
 */
typedef struct {
	unsigned int triggers;
	unsigned int interval;
	unsigned int control;
	pfmlib_cell_counter_t pfp_cell_counters[PMU_CELL_NUM_COUNTERS];	/* extended counter features */
	uint64_t              reserved[4];				/* for future use */
} pfmlib_cell_input_param_t;

typedef struct {
	uint64_t	reserved[8];	/* for future use */
} pfmlib_cell_output_param_t;

int pfm_cell_spe_event(unsigned int event_index);

#endif /* __PFMLIB_CELL_H__ */
