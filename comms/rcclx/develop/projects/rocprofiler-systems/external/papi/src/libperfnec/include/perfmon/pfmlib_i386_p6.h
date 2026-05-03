/*
 * Intel Pentium II/Pentium Pro/Pentium III/Pentium M PMU specific types and definitions
 *
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_I386_P6_H__
#define __PFMLIB_I386_P6_H__

#include <perfmon/pfmlib.h>

/*
 * privilege level mask usage for i386-p6:
 *
 * PFM_PLM0 = OS (kernel, hypervisor, ..)
 * PFM_PLM1 = unused (ignored)
 * PFM_PLM2 = unused (ignored)
 * PFM_PLM3 = USR (user level)
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_I386_P6_NUM_COUNTERS	2	/* total numbers of EvtSel/EvtCtr */
#define PMU_I386_P6_NUM_PERFSEL		2	/* total number of EvtSel defined */
#define PMU_I386_P6_NUM_PERFCTR		2	/* total number of EvtCtr defined */
#define PMU_I386_P6_COUNTER_WIDTH	32	/* hardware counter bit width   */

/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 */
typedef union {
	unsigned long	val;			/* complete register value */
	struct {
		unsigned long sel_event_mask:8;	/* event mask */
		unsigned long sel_unit_mask:8;		/* unit mask */
		unsigned long sel_usr:1;		/* user level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_edge:1;		/* edge detec */
		unsigned long sel_pc:1;			/* pin control */
		unsigned long sel_int:1;		/* enable APIC intr */
		unsigned long sel_res1:1;		/* reserved */
		unsigned long sel_en:1;			/* enable */
		unsigned long sel_inv:1;		/* invert counter mask */
		unsigned long sel_cnt_mask:8;		/* counter mask */
	} perfsel;
} pfm_i386_p6_sel_reg_t;

typedef union {
	uint64_t val;	/* counter value */
	/* counting perfctr register */
	struct {
		unsigned long ctr_count:32;	/* 32-bit hardware counter  */
		unsigned long ctr_res1:32;	/* reserved */
	} perfctr;
} pfm_i386_p6_ctr_reg_t;

typedef enum {
	PFM_I386_P6_CNT_MASK_0,
	PFM_I386_P6_CNT_MASK_1,
	PFM_I386_P6_CNT_MASK_2,
	PFM_I386_P6_CNT_MASK_3
} pfm_i386_p6_cnt_mask_t;

typedef struct {
	pfm_i386_p6_cnt_mask_t	cnt_mask;	/* threshold (cnt_mask)  */
	unsigned int		flags;		/* counter specific flag */
} pfmlib_i386_p6_counter_t;

#define PFM_I386_P6_SEL_INV	0x1	/* inverse */
#define PFM_I386_P6_SEL_EDGE	0x2	/* edge detect */

/*
 * P6-specific parameters for the library
 */
typedef struct {
	pfmlib_i386_p6_counter_t	pfp_i386_p6_counters[PMU_I386_P6_NUM_COUNTERS];	/* extended counter features */
	uint64_t			reserved[4];		/* for future use */
} pfmlib_i386_p6_input_param_t;

typedef struct {
	uint64_t	reserved[8];		/* for future use */
} pfmlib_i386_p6_output_param_t;

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_I386_P6_H__ */
