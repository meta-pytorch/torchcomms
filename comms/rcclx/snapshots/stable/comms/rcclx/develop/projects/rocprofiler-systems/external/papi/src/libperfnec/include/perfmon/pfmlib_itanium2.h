/*
 * Itanium 2 PMU specific types and definitions
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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

#ifndef __PFMLIB_ITANIUM2_H__
#define __PFMLIB_ITANIUM2_H__

#include <perfmon/pfmlib.h>
#include <endian.h>

#if BYTE_ORDER != LITTLE_ENDIAN
#error "this file only supports little endian environments"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define PMU_ITA2_FIRST_COUNTER	4	/* index of first PMC/PMD counter */
#define PMU_ITA2_NUM_COUNTERS	4	/* total numbers of PMC/PMD pairs used as counting monitors */
#define PMU_ITA2_NUM_PMCS	16	/* total number of PMCS defined */
#define PMU_ITA2_NUM_PMDS	18	/* total number of PMDS defined */
#define PMU_ITA2_NUM_BTB	8	/* total number of PMDS in BTB  */
#define PMU_ITA2_COUNTER_WIDTH	47	/* hardware counter bit width   */


/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 */
typedef union {
	unsigned long pmc_val;			/* complete register value */

	/* This is the Itanium2-specific PMC layout for counter config */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* reserved */
		unsigned long pmc_es:8;		/* event select */
		unsigned long pmc_umask:4;	/* unit mask */
		unsigned long pmc_thres:3;	/* threshold */
		unsigned long pmc_enable:1;	/* pmc4 only: power enable bit */
		unsigned long pmc_ism:2;	/* instruction set mask */
		unsigned long pmc_ig2:38;	/* reserved */
	} pmc_ita2_counter_reg;

	/* opcode matchers */
	struct {
		unsigned long opcm_ig_ad:1;	/* ignore instruction address range checking */
		unsigned long opcm_inv:1;	/* invert range check */
		unsigned long opcm_bit2:1;	/* must be 1 */
		unsigned long opcm_mask:27;	/* mask encoding bits {41:27}{12:0} */
		unsigned long opcm_ig1:3;	/* reserved */
		unsigned long opcm_match:27;	/* match encoding bits {41:27}{12:0} */
		unsigned long opcm_b:1;		/* B-syllable */
		unsigned long opcm_f:1;		/* F-syllable */
		unsigned long opcm_i:1;		/* I-syllable */
		unsigned long opcm_m:1;		/* M-syllable */
	} pmc8_9_ita2_reg;

	/*
	 * instruction event address register configuration
	 *
	 * The register has two layout depending on the value of the ct field.
	 * In cache mode(ct=1x):
	 * 	- ct is 1 bit, umask is 8 bits
	 * In TLB mode (ct=00):
	 * 	- ct is 2 bits, umask is 7 bits
	 * ct=11 <=> cache mode and use a latency with eighth bit set
	 * ct=01 => nothing monitored
	 *
	 * The ct=01 value is the only reason why we cannot fix the layout
	 * to ct 1 bit and umask 8 bits. Even though in TLB mode, only 6 bits
	 * are effectively used for the umask, if the user inadvertently use
	 * a umask with the most significant bit set, it would be equivalent
	 * to no monitoring.
	 */
	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_umask:8;	/* event unit mask: 7 bits in TLB mode, 8 bits in cache mode */
		unsigned long iear_ct:1;	/* cache tlb bit13: 0 for TLB mode, 1 for cache mode  */
		unsigned long iear_ism:2;	/* instruction set */
		unsigned long iear_ig4:48;	/* reserved */
	} pmc10_ita2_cache_reg;

	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_umask:7;	/* event unit mask: 7 bits in TLB mode, 8 bits in cache mode */
		unsigned long iear_ct:2;	/* cache tlb bit13: 0 for TLB mode, 1 for cache mode  */
		unsigned long iear_ism:2;	/* instruction set */
		unsigned long iear_ig4:48;	/* reserved */
	} pmc10_ita2_tlb_reg;

	/* data event address register configuration */
	struct {
		unsigned long dear_plm:4;	/* privilege level mask */
		unsigned long dear_ig1:2;	/* reserved */
		unsigned long dear_pm:1;	/* privileged monitor */
		unsigned long dear_mode:2;	/* mode */
		unsigned long dear_ig2:7;	/* reserved */
		unsigned long dear_umask:4;	/* unit mask */
		unsigned long dear_ig3:4;	/* reserved */
		unsigned long dear_ism:2;	/* instruction set */
		unsigned long dear_ig4:38;	/* reserved */
	} pmc11_ita2_reg;

	/* branch trace buffer configuration register */
	struct {
		unsigned long btbc_plm:4;	/* privilege level */
		unsigned long btbc_ig1:2;
		unsigned long btbc_pm:1;	/* privileged monitor */
		unsigned long btbc_ds:1;	/* data selector */
		unsigned long btbc_tm:2;	/* taken mask */
		unsigned long btbc_ptm:2;	/* predicted taken address mask */
		unsigned long btbc_ppm:2;	/* predicted predicate mask */
		unsigned long btbc_brt:2;	/* branch type mask */
		unsigned long btbc_ig2:48;
	} pmc12_ita2_reg;

	/* data address range configuration register */
	struct {
		unsigned long darc_ig1:3;
		unsigned long darc_cfg_dbrp0:2;	/* constraint on dbr0 */
		unsigned long darc_ig2:6;
		unsigned long darc_cfg_dbrp1:2;	/* constraint on dbr1 */
		unsigned long darc_ig3:6;
		unsigned long darc_cfg_dbrp2:2;	/* constraint on dbr2 */
		unsigned long darc_ig4:6;
		unsigned long darc_cfg_dbrp3:2;	/* constraint on dbr3 */
		unsigned long darc_ig5:16;
		unsigned long darc_ena_dbrp0:1;	/* enable constraint dbr0 */
		unsigned long darc_ena_dbrp1:1;	/* enable constraint dbr1 */
		unsigned long darc_ena_dbrp2:1;	/* enable constraint dbr2 */
		unsigned long darc_ena_dbrp3:1; 	/* enable constraint dbr3 */
		unsigned long darc_ig6:15;
	} pmc13_ita2_reg;

	/* instruction address range configuration register */
	struct {
		unsigned long iarc_ig1:1;
		unsigned long iarc_ibrp0:1;	/* constrained by ibr0 */
		unsigned long iarc_ig2:2;
		unsigned long iarc_ibrp1:1;	/* constrained by ibr1 */
		unsigned long iarc_ig3:2;
		unsigned long iarc_ibrp2:1;	/* constrained by ibr2 */
		unsigned long iarc_ig4:2;
		unsigned long iarc_ibrp3:1;	/* constrained by ibr3 */
		unsigned long iarc_ig5:2;
		unsigned long iarc_fine:1;	/* fine mode */
		unsigned long iarc_ig6:50;
	} pmc14_ita2_reg;

	/* opcode matcher configuration register */
	struct {
		unsigned long	opcmc_ibrp0_pmc8:1;
		unsigned long	opcmc_ibrp1_pmc9:1;
		unsigned long	opcmc_ibrp2_pmc8:1;
		unsigned long	opcmc_ibrp3_pmc9:1;
		unsigned long 	opcmc_ig1:60;
	} pmc15_ita2_reg;
} pfm_ita2_pmc_reg_t;

typedef union {
	unsigned long	pmd_val;	/* counter value */

	/* counting pmd register */
	struct {
		unsigned long pmd_count:47;	/* 47-bit hardware counter  */
		unsigned long pmd_sxt47:17;	/* sign extension of bit 46 */
	} pmd_ita2_counter_reg;

	/* instruction event address register: data address register */
	struct {
		unsigned long iear_stat:2;	/* status bit */
		unsigned long iear_ig1:3;
		unsigned long iear_iaddr:59;	/* instruction cache line address {60:51} sxt {50}*/
	} pmd0_ita2_reg;

	/* instruction event address register: data address register */
	struct {
		unsigned long iear_latency:12;	/* latency */
		unsigned long iear_overflow:1;	/* latency overflow */
		unsigned long iear_ig1:51;	/* reserved */
	} pmd1_ita2_reg;

	/* data event address register: data address register */
	struct {
		unsigned long dear_daddr;	/* data address */
	} pmd2_ita2_reg;

	/* data event address register: data address register */
	struct {
		unsigned long dear_latency:13;	/* latency  */
		unsigned long dear_overflow:1;	/* overflow */
		unsigned long dear_stat:2;	/* status   */
		unsigned long dear_ig1:48;	/* ignored  */
	} pmd3_ita2_reg;

	/* branch trace buffer data register when pmc12.ds == 0 */
	struct {
		unsigned long btb_b:1;		/* branch bit */
		unsigned long btb_mp:1;		/* mispredict bit */
		unsigned long btb_slot:2;	/* which slot, 3=not taken branch */
		unsigned long btb_addr:60;	/* bundle address(b=1), target address(b=0) */
	} pmd8_15_ita2_reg;

	/* branch trace buffer data register when pmc12.ds == 1 */
	struct {
		unsigned long btb_b:1;		/* branch bit */
		unsigned long btb_mp:1;		/* mispredict bit */
		unsigned long btb_slot:2;	/* which slot, 3=not taken branch */
		unsigned long btb_loaddr:37;	/* b=1, bundle address, b=0 target address */
		unsigned long btb_pred:20;	/* low 20bits of L1IBR */
		unsigned long btb_hiaddr:3;	/* hi 3bits of bundle address(b=1) or target address (b=0)*/
	} pmd8_15_ds_ita2_reg;

	/* branch trace buffer index register */
	struct {
		unsigned long btbi_bbi:3;		/* next entry index  */
		unsigned long btbi_full:1;		/* full bit (sticky) */
		unsigned long btbi_pmd8ext_b1:1;	/* pmd8 ext  */
		unsigned long btbi_pmd8ext_bruflush:1;	/* pmd8 ext  */
		unsigned long btbi_pmd8ext_ig:2;	/* pmd8 ext  */
		unsigned long btbi_pmd9ext_b1:1;	/* pmd9 ext  */
		unsigned long btbi_pmd9ext_bruflush:1;	/* pmd9 ext  */
		unsigned long btbi_pmd9ext_ig:2;	/* pmd9 ext  */
		unsigned long btbi_pmd10ext_b1:1;	/* pmd10 ext */
		unsigned long btbi_pmd10ext_bruflush:1;	/* pmd10 ext */
		unsigned long btbi_pmd10ext_ig:2;	/* pmd10 ext */
		unsigned long btbi_pmd11ext_b1:1;	/* pmd11 ext */
		unsigned long btbi_pmd11ext_bruflush:1;	/* pmd11 ext */
		unsigned long btbi_pmd11ext_ig:2;	/* pmd11 ext */
		unsigned long btbi_pmd12ext_b1:1;	/* pmd12 ext */
		unsigned long btbi_pmd12ext_bruflush:1;	/* pmd12 ext */
		unsigned long btbi_pmd12ext_ig:2;	/* pmd12 ext */
		unsigned long btbi_pmd13ext_b1:1;	/* pmd13 ext */
		unsigned long btbi_pmd13ext_bruflush:1;	/* pmd13 ext */
		unsigned long btbi_pmd13ext_ig:2;	/* pmd13 ext */
		unsigned long btbi_pmd14ext_b1:1;	/* pmd14 ext */
		unsigned long btbi_pmd14ext_bruflush:1;	/* pmd14 ext */
		unsigned long btbi_pmd14ext_ig:2;	/* pmd14 ext */
		unsigned long btbi_pmd15ext_b1:1;	/* pmd15 ext */
		unsigned long btbi_pmd15ext_bruflush:1;	/* pmd15 ext */
		unsigned long btbi_pmd15ext_ig:2;	/* pmd15 ext */
		unsigned long btbi_ignored:28;
	} pmd16_ita2_reg;

	/* data event address register: data address register */
	struct {
		unsigned long dear_slot:2;	/* slot   */
		unsigned long dear_bn:1;	/* bundle bit (if 1 add 16 to address) */
		unsigned long dear_vl:1;	/* valid  */
		unsigned long dear_iaddr:60;	/* instruction address (2-bundle window)*/
	} pmd17_ita2_reg;
} pfm_ita2_pmd_reg_t;

/*
 * type definition for Itanium 2 instruction set support
 */
typedef enum {
	PFMLIB_ITA2_ISM_BOTH=0, 	/* IA-32 and IA-64 (default) */
	PFMLIB_ITA2_ISM_IA32=1, 	/* IA-32 only */
	PFMLIB_ITA2_ISM_IA64=2 		/* IA-64 only */
} pfmlib_ita2_ism_t;

typedef struct {
	unsigned int	  flags;	/* counter specific flags */
	unsigned int 	  thres;	/* per event threshold */
	pfmlib_ita2_ism_t  ism;		/* per event instruction set */
} pfmlib_ita2_counter_t;

/*
 * counter specific flags
 */
#define PFMLIB_ITA2_FL_EVT_NO_QUALCHECK	0x1 /* don't check qualifier constraints */

typedef struct {
	unsigned char	 opcm_used;	/* set to 1 if this opcode matcher is used */
	unsigned long	 pmc_val;	/* full opcode mask (41bits) */
} pfmlib_ita2_opcm_t;

/*
 *
 * The BTB can be configured via 4 different methods:
 *
 * 	- BRANCH_EVENT is in the event list, pfp_ita2_btb.btb_used == 0:
 * 		The BTB will be configured (PMC12) to record all branches AND a counting
 * 		monitor will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is in the event list, pfp_ita2_btb.btb_used == 1:
 * 		The BTB will be configured (PMC12) according to information in pfp_ita2_btb AND
 * 		a counter will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_ita2_btb.btb_used == 0:
 * 	   	Nothing is programmed
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_ita2_btb.btb_used == 1:
 * 		The BTB will be configured (PMC12) according to information in pfp_ita2_btb.
 * 		This is the free running BTB mode.
 */
typedef struct {
	unsigned char	 btb_used;	/* set to 1 if the BTB is used */

	unsigned char	 btb_ds;	/* data selector */
	unsigned char	 btb_tm;	/* taken mask */
	unsigned char	 btb_ptm;	/* predicted target mask */
	unsigned char	 btb_ppm;	/* predicted predicate mask */
	unsigned char	 btb_brt;	/* branch type mask */
	unsigned int	 btb_plm;	/* BTB privilege level mask */
} pfmlib_ita2_btb_t;

/*
 * There are four ways to configure EAR:
 *
 * 	- an EAR event is in the event list AND pfp_ita2_?ear.ear_used = 0:
 * 		The EAR will be programmed (PMC10 or PMC11) based on the information encoded in the
 * 		event (umask, cache, tlb,alat). A counting monitor will be programmed to
 * 		count DATA_EAR_EVENTS or L1I_EAR_EVENTS depending on the type of EAR.
 *
 * 	- an EAR event is in the event list AND pfp_ita2_?ear.ear_used = 1:
 * 		The EAR will be programmed (PMC10 or PMC11) according to the information in the
 * 		pfp_ita2_?ear structure	because it contains more detailed information
 * 		(such as priv level and instruction set). A counting monitor will be programmed
 * 		to count DATA_EAR_EVENTS or L1I_EAR_EVENTS depending on the type of EAR.
 *
 * 	- no EAR event is in the event list AND pfp_ita2_?ear.ear_used = 0:
 * 	 	Nothing is programmed.
 *
 * 	- no EAR event is in the event list AND pfp_ita2_?ear.ear_used = 1:
 * 		The EAR will be programmed (PMC10 or PMC11) according to the information in the
 * 		pfp_ita2_?ear structure. This is the free running mode for EAR
 */

typedef enum {
	PFMLIB_ITA2_EAR_CACHE_MODE= 0,	/* Cache mode : I-EAR and D-EAR */
	PFMLIB_ITA2_EAR_TLB_MODE  = 1, 	/* TLB mode   : I-EAR and D-EAR */
	PFMLIB_ITA2_EAR_ALAT_MODE = 2	/* ALAT mode  : D-EAR only      */
} pfmlib_ita2_ear_mode_t;

typedef struct {
	unsigned char		ear_used;	/* when set will force definition of PMC[10] */

	pfmlib_ita2_ear_mode_t	ear_mode;	/* EAR mode */
	pfmlib_ita2_ism_t	ear_ism;	/* instruction set */
	unsigned int		ear_plm;	/* IEAR privilege level mask */
	unsigned long		ear_umask;	/* umask value for PMC10 */
} pfmlib_ita2_ear_t;

/*
 * describes one range. rr_plm is ignored for data ranges
 * a range is interpreted as unused (not defined) when rr_start = rr_end = 0.
 * if rr_plm is not set it will use the default settings set in the generic
 * library param structure.
 */
typedef struct {
	unsigned int		rr_plm;		/* privilege level (ignored for data ranges) */
	unsigned long		rr_start;	/* start address */
	unsigned long		rr_end;		/* end address (not included) */
} pfmlib_ita2_input_rr_desc_t;

typedef struct {
	unsigned long		rr_soff;	/* start offset from actual start */
	unsigned long		rr_eoff;	/* end offset from actual end */
} pfmlib_ita2_output_rr_desc_t;


/*
 * rr_used must be set to true for the library to configure the debug registers.
 * rr_inv only applies when the rr_limits table contains ONLY 1 range.
 *
 * If using less than 4 intervals, must mark the end with entry: rr_start = rr_end = 0
 */
typedef struct {
	unsigned int			rr_flags;	/* set of flags for all ranges              */
	pfmlib_ita2_input_rr_desc_t	rr_limits[4];	/* at most 4 distinct intervals             */
	unsigned char	 		rr_used;	/* set if address range restriction is used */
} pfmlib_ita2_input_rr_t;

typedef struct {
	unsigned int			rr_nbr_used;	/* how many registers were used */
	pfmlib_ita2_output_rr_desc_t	rr_infos[4];	/* at most 4 distinct intervals */
	pfmlib_reg_t			rr_br[8];	/* debug reg to configure       */
} pfmlib_ita2_output_rr_t;


#define PFMLIB_ITA2_RR_INV		0x1 /* inverse instruction ranges (iranges only) */
#define PFMLIB_ITA2_RR_NO_FINE_MODE	0x2 /* force non fine mode for instruction ranges */

/*
 * Itanium 2 specific parameters for the library
 */
typedef struct {
	pfmlib_ita2_counter_t	pfp_ita2_counters[PMU_ITA2_NUM_COUNTERS];	/* extended counter features */

	unsigned long		pfp_ita2_flags;		/* Itanium2 specific flags */

	pfmlib_ita2_opcm_t	pfp_ita2_pmc8;		/* PMC8 (opcode matcher) configuration */
	pfmlib_ita2_opcm_t	pfp_ita2_pmc9;		/* PMC9 (opcode matcher) configuration */
	pfmlib_ita2_ear_t	pfp_ita2_iear;		/* IEAR configuration */
	pfmlib_ita2_ear_t	pfp_ita2_dear;		/* DEAR configuration */
	pfmlib_ita2_btb_t	pfp_ita2_btb;		/* BTB configuration */
	pfmlib_ita2_input_rr_t	pfp_ita2_drange;	/* data range restrictions */
	pfmlib_ita2_input_rr_t	pfp_ita2_irange;	/* code range restrictions */
	unsigned long		reserved[1];		/* for future use */
} pfmlib_ita2_input_param_t;

typedef struct {
	pfmlib_ita2_output_rr_t	pfp_ita2_drange;	/* data range restrictions */
	pfmlib_ita2_output_rr_t	pfp_ita2_irange;	/* code range restrictions */
	unsigned long		reserved[6];		/* for future use */
} pfmlib_ita2_output_param_t;


extern int pfm_ita2_is_ear(unsigned int i);
extern int pfm_ita2_is_dear(unsigned int i);
extern int pfm_ita2_is_dear_tlb(unsigned int i);
extern int pfm_ita2_is_dear_cache(unsigned int i);
extern int pfm_ita2_is_dear_alat(unsigned int i);
extern int pfm_ita2_is_iear(unsigned int i);
extern int pfm_ita2_is_iear_tlb(unsigned int i);
extern int pfm_ita2_is_iear_cache(unsigned int i);
extern int pfm_ita2_is_btb(unsigned int i);
extern int pfm_ita2_support_opcm(unsigned int i);
extern int pfm_ita2_support_iarr(unsigned int i);
extern int pfm_ita2_support_darr(unsigned int i);
extern int pfm_ita2_get_ear_mode(unsigned int i, pfmlib_ita2_ear_mode_t *m);
extern int pfm_ita2_irange_is_fine(pfmlib_output_param_t *outp, pfmlib_ita2_output_param_t *mod_out);

extern int pfm_ita2_get_event_maxincr(unsigned int i, unsigned int *maxincr);
extern int pfm_ita2_get_event_umask(unsigned int i, unsigned long *umask);
extern int pfm_ita2_get_event_group(unsigned int i, int *grp);
extern int pfm_ita2_get_event_set(unsigned int i, int *set);

/*
 * values of group (grp) returned by pfm_ita2_get_event_group()
 */
#define PFMLIB_ITA2_EVT_NO_GRP		 0 /* event does not belong to a group */
#define PFMLIB_ITA2_EVT_L1_CACHE_GRP	 1 /* event belongs to L1 Cache group */
#define PFMLIB_ITA2_EVT_L2_CACHE_GRP	 2 /* event belongs to L2 Cache group */

/*
 * possible values returned in set by pfm_ita2_get_event_set()
 */
#define PFMLIB_ITA2_EVT_NO_SET		-1 /* event does not belong to a set */

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_ITANIUM2_H__ */
