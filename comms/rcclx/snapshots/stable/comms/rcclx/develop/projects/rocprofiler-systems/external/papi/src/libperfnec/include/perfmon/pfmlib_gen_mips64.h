/*
 * Generic MIPS64 PMU specific types and definitions
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

#ifndef __PFMLIB_GEN_MIPS64_H__
#define __PFMLIB_GEN_MIPS64_H__

#include <endian.h> /* MIPS are bi-endian */

#include <perfmon/pfmlib.h>
/*
 * privilege level mask usage for MIPS:
 *
 * PFM_PLM0 = KERNEL
 * PFM_PLM1 = SUPERVISOR
 * PFM_PLM2 = INTERRUPT
 * PFM_PLM3 = USER
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_GEN_MIPS64_NUM_COUNTERS		4	/* total numbers of EvtSel/EvtCtr */
#define PMU_GEN_MIPS64_COUNTER_WIDTH		32	/* hardware counter bit width   */
/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 *
 * It needs to be adjusted based on endianess
 */
#if __BYTE_ORDER == __LITTLE_ENDIAN

typedef union {
	uint64_t	val;				/* complete register value */
	struct {
	        unsigned long sel_exl:1;		/* int level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_sup:1;		/* supervisor level */
		unsigned long sel_usr:1;		/* user level */
	        unsigned long sel_int:1;		/* enable intr */
	  	unsigned long sel_event_mask:5;		/* event mask */
		unsigned long sel_res1:22;		/* reserved */
		unsigned long sel_res2:32;		/* reserved */
	} perfsel;
} pfm_gen_mips64_sel_reg_t;

#elif __BYTE_ORDER == __BIG_ENDIAN

typedef union {
	uint64_t	val;				/* complete register value */
	struct {
		unsigned long sel_res2:32;		/* reserved */
		unsigned long sel_res1:22;		/* reserved */
	  	unsigned long sel_event_mask:5;		/* event mask */
	        unsigned long sel_int:1;		/* enable intr */
		unsigned long sel_usr:1;		/* user level */
		unsigned long sel_sup:1;		/* supervisor level */
		unsigned long sel_os:1;			/* system level */
	        unsigned long sel_exl:1;		/* int level */
	} perfsel;
} pfm_gen_mips64_sel_reg_t;

#else
#error "cannot determine endianess"
#endif

typedef union {
	uint64_t val;	/* counter value */
	/* counting perfctr register */
	struct {
		unsigned long ctr_count:32;	/* 32-bit hardware counter  */
	} perfctr;
} pfm_gen_mips64_ctr_reg_t;

typedef struct {
	unsigned int	cnt_mask;	/* threshold ([4-255] are reserved) */
	unsigned int	flags;		/* counter specific flag */
} pfmlib_gen_mips64_counter_t;

/*
 * MIPS64 specific parameters for the library
 */
typedef struct {
	pfmlib_gen_mips64_counter_t	pfp_gen_mips64_counters[PMU_GEN_MIPS64_NUM_COUNTERS];	/* extended counter features */
	uint64_t			reserved[4];		/* for future use */
} pfmlib_gen_mips64_input_param_t;

typedef struct {
	uint64_t	reserved[8];		/* for future use */
} pfmlib_gen_mips64_output_param_t;

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_GEN_MIPS64_H__ */
