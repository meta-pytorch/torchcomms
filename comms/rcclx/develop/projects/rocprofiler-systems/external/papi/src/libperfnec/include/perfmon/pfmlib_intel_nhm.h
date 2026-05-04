/*
 * Intel Nehalem PMU
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#ifndef __PFMLIB_NHM_H__
#define __PFMLIB_NHM_H__

#include <perfmon/pfmlib.h>
/*
 * privilege level mask usage for Intel Core
 *
 * PFM_PLM0 = OS (kernel, hypervisor, ..)
 * PFM_PLM1 = unused (ignored)
 * PFM_PLM2 = unused (ignored)
 * PFM_PLM3 = USR (user level)
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * total number of counters:
 * 	- 4 generic core
 * 	- 3 fixed core
 * 	- 1 uncore fixed
 * 	- 8 uncore generic
 */
#define PMU_NHM_NUM_COUNTERS 16

typedef union {
	unsigned long long val;			/* complete register value */
	struct {
		unsigned long sel_event:8;	/* event mask */
		unsigned long sel_umask:8;	/* unit mask */
		unsigned long sel_usr:1;	/* user level */
		unsigned long sel_os:1;		/* system level */
		unsigned long sel_edge:1;	/* edge detec */
		unsigned long sel_pc:1;		/* pin control */
		unsigned long sel_int:1;	/* enable APIC intr */
		unsigned long sel_anythr:1;	/* measure any thread */
		unsigned long sel_en:1;		/* enable */
		unsigned long sel_inv:1;	/* invert counter mask */
		unsigned long sel_cnt_mask:8;	/* counter mask */
		unsigned long sel_res2:32;
	} perfevtsel;
	struct {
		unsigned long usel_event:8;	/* event select */
		unsigned long usel_umask:8;	/* event unit mask */
		unsigned long usel_res1:1;	/* reserved */
		unsigned long usel_occ:1;	/* occupancy reset */
		unsigned long usel_edge:1;	/* edge detection */
		unsigned long usel_res2:1;	/* reserved */
		unsigned long usel_int:1;	/* PMI enable */
		unsigned long usel_res3:1;	/* reserved */
		unsigned long usel_en:1;		/* enable */
		unsigned long usel_inv:1;	/* invert */
		unsigned long usel_cnt_mask:8;	/* counter mask */
		unsigned long usel_res4:32;	/* reserved */
	} unc_perfevtsel;
	struct {
		unsigned long cpl_eq0:1;	/* filter out branches at pl0 */
		unsigned long cpl_neq0:1;	/* filter out branches at pl1-pl3 */
		unsigned long jcc:1;		/* filter out condition branches */
		unsigned long near_rel_call:1;	/* filter out near relative calls */
		unsigned long near_ind_call:1;	/* filter out near indirect calls */
		unsigned long near_ret:1;	/* filter out near returns */
		unsigned long near_ind_jmp:1;	/* filter out near unconditional jmp/calls */
		unsigned long near_rel_jmp:1;	/* filter out near uncoditional relative jmp */
		unsigned long far_branch:1;	/* filter out far branches */ 
		unsigned long reserved1:23;	/* reserved */
		unsigned long reserved2:32;	/* reserved */
	} lbr_select;
} pfm_nhm_sel_reg_t;

typedef struct {
	unsigned long		cnt_mask;	/* counter mask (occurences) */
	unsigned int		flags;		/* counter specific flag */
} pfmlib_nhm_counter_t;

/*
 * flags for pfmlib_nhm_counter_t
 */
#define PFM_NHM_SEL_INV		0x1	/* inverse */
#define PFM_NHM_SEL_EDGE	0x2	/* edge detect */
#define PFM_NHM_SEL_ANYTHR	0x4	/* any thread (core only) */
#define PFM_NHM_SEL_OCC_RST	0x8	/* reset occupancy (uncore only) */

typedef struct {
	unsigned int lbr_used;	/* set to 1 if LBR is used */
	unsigned int lbr_plm;	/* priv level PLM0 or PLM3 */
	unsigned int lbr_filter;/* filters */
} pfmlib_nhm_lbr_t;

/*
 * lbr_filter: filter out branches
 * refer to IA32 SDM vol3b section 18.6.2
 */
#define PFM_NHM_LBR_JCC			0x4  /* do not capture conditional branches */
#define PFM_NHM_LBR_NEAR_REL_CALL	0x8  /* do not capture near calls */
#define PFM_NHM_LBR_NEAR_IND_CALL	0x10 /* do not capture indirect calls */
#define PFM_NHM_LBR_NEAR_RET		0x20 /* do not capture near returns */
#define PFM_NHM_LBR_NEAR_IND_JMP	0x40 /* do not capture indirect jumps */
#define PFM_NHM_LBR_NEAR_REL_JMP	0x80 /* do not capture near relative jumps */
#define PFM_NHM_LBR_FAR_BRANCH		0x100/* do not capture far branches */
#define PFM_NHM_LBR_ALL			0x1fc /* filter out all branches */

/*
 * PEBS input parameters
 */
typedef struct {
	unsigned int pebs_used;		/* set to 1 if PEBS is used */
	unsigned int ld_lat_thres;	/* load latency threshold (cycles) */
} pfmlib_nhm_pebs_t;


/*
 * model-specific input parameter to pfm_dispatch_events()
 */
typedef struct {
	pfmlib_nhm_counter_t	pfp_nhm_counters[PMU_NHM_NUM_COUNTERS];
	pfmlib_nhm_pebs_t	pfp_nhm_pebs;	/* PEBS settings */
	pfmlib_nhm_lbr_t	pfp_nhm_lbr;	/* LBR settings */
	uint64_t		reserved[4];	/* for future use */
} pfmlib_nhm_input_param_t;

/*
 * no pfmlib_nhm_output_param_t defined
 */

/*
 * Model-specific interface
 * can be called directly
 */
extern int pfm_nhm_is_pebs(pfmlib_event_t *e);
extern int pfm_nhm_is_uncore(pfmlib_event_t *e);
extern int pfm_nhm_data_src_desc(unsigned int val, char **desc);

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_NHM_H__ */
