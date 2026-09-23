/*
 * Itanium PMU specific types and definitions
 *
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
#ifndef __PFMLIB_ITANIUM_H__
#define __PFMLIB_ITANIUM_H__

#include <perfmon/pfmlib.h>
#include <endian.h>

#if BYTE_ORDER != LITTLE_ENDIAN
#error "this file only supports little endian environments"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_ITA_FIRST_COUNTER	4	/* index of first PMC/PMD counter */
#define PMU_ITA_NUM_COUNTERS	4	/* total numbers of PMC/PMD pairs used as counting monitors */
#define PMU_ITA_NUM_PMCS	14	/* total number of PMCS defined */
#define PMU_ITA_NUM_PMDS	18	/* total number of PMDS defined */
#define PMU_ITA_NUM_BTB		8	/* total number of PMDS in BTB  */
#define PMU_ITA_COUNTER_WIDTH	32	/* hardware counter bit width   */


/*
 * This structure provides a detailed way to setup a PMC register.
 */
typedef union {
	unsigned long pmc_val;			/* complete register value */

	/* This is the Itanium-specific PMC layout for counter config */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* reserved */
		unsigned long pmc_es:7;		/* event select */
		unsigned long pmc_ig2:1;	/* reserved */
		unsigned long pmc_umask:4;	/* unit mask */
		unsigned long pmc_thres:3;	/* threshold */
		unsigned long pmc_ig3:1;	/* reserved (missing from table on p6-17) */
		unsigned long pmc_ism:2;	/* instruction set mask */
		unsigned long pmc_ig4:38;	/* reserved */
	} pmc_ita_count_reg;

	/* Opcode matcher */
	struct {
		unsigned long ignored1:3;
		unsigned long mask:27;		/* mask encoding bits {40:27}{12:0} */
		unsigned long ignored2:3;	
		unsigned long match:27;		/* match encoding bits {40:27}{12:0} */
		unsigned long b:1;		/* B-syllable */
		unsigned long f:1;		/* F-syllable */
		unsigned long i:1;		/* I-syllable */
		unsigned long m:1;		/* M-syllable */
	} pmc8_9_ita_reg;

	/* Instruction Event Address Registers */
	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_ig1:2;	/* reserved */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_tlb:1;	/* cache/tlb mode */
		unsigned long iear_ig2:8;	/* reserved */
		unsigned long iear_umask:4;	/* unit mask */
		unsigned long iear_ig3:4;	/* reserved */
		unsigned long iear_ism:2;	/* instruction set */
		unsigned long iear_ig4:38;	/* reserved */
	} pmc10_ita_reg;

	/* Data Event Address Registers */
	struct {
		unsigned long dear_plm:4;	/* privilege level mask */
		unsigned long dear_ig1:2;	/* reserved */
		unsigned long dear_pm:1;	/* privileged monitor */
		unsigned long dear_tlb:1;	/* cache/tlb mode */
		unsigned long dear_ig2:8;	/* reserved */
		unsigned long dear_umask:4;	/* unit mask */
		unsigned long dear_ig3:4;	/* reserved */
		unsigned long dear_ism:2;	/* instruction set */
		unsigned long dear_ig4:2;	/* reserved */
		unsigned long dear_pt:1;	/* pass tags */
		unsigned long dear_ig5:35;	/* reserved */
	} pmc11_ita_reg;

	/* Branch Trace Buffer registers */
	struct {
		unsigned long btbc_plm:4;	/* privilege level */
		unsigned long btbc_ig1:2;
		unsigned long btbc_pm:1;	/* privileged monitor */
		unsigned long btbc_tar:1;	/* target address register */
		unsigned long btbc_tm:2;	/* taken mask */
		unsigned long btbc_ptm:2;	/* predicted taken address mask */
		unsigned long btbc_ppm:2;	/* predicted predicate mask */
		unsigned long btbc_bpt:1;	/* branch prediction table */
		unsigned long btbc_bac:1;	/* branch address calculator */
		unsigned long btbc_ig2:48;
	} pmc12_ita_reg;

	struct {
		unsigned long irange_ta:1;	/* tag all bit */
		unsigned long irange_ig:63;
	} pmc13_ita_reg;
} pfm_ita_pmc_reg_t;

typedef union {
	unsigned long	pmd_val;	/* counter value */

	/* counting pmd register */
	struct {
		unsigned long pmd_count:32;	/* 32-bit hardware counter  */
		unsigned long pmd_sxt32:32;	/* sign extension of bit 32 */
	} pmd_ita_counter_reg;

	struct {
		unsigned long iear_v:1;		/* valid bit */
		unsigned long iear_tlb:1;	/* tlb miss bit */
		unsigned long iear_ig1:3;	/* reserved */
		unsigned long iear_icla:59;	/* instruction cache line address {60:51} sxt {50}*/
	} pmd0_ita_reg;

	struct {
		unsigned long iear_lat:12;	/* latency */
		unsigned long iear_ig1:52;	/* reserved */
	} pmd1_ita_reg;

	struct {
		unsigned long dear_daddr;	/* data address */
	} pmd2_ita_reg;

	struct {
		unsigned long dear_latency:12;	/* latency */
		unsigned long dear_ig1:50;	/* reserved */
		unsigned long dear_level:2;	/* level */
	} pmd3_ita_reg;

	struct {
		unsigned long btb_b:1;		/* branch bit */
		unsigned long btb_mp:1;		/* mispredict bit */
		unsigned long btb_slot:2;	/* which slot, 3=not taken branch */
		unsigned long btb_addr:60;	/* b=1, bundle address, b=0 target address */
	} pmd8_15_ita_reg;

	struct {
		unsigned long btbi_bbi:3;	/* branch buffer index */
		unsigned long btbi_full:1;	/* full bit (sticky) */
		unsigned long btbi_ignored:60;
	} pmd16_ita_reg;

	struct {
		unsigned long dear_vl:1;	/* valid bit */
		unsigned long dear_ig1:1;	/* reserved */
		unsigned long dear_slot:2;	/* slot number */
		unsigned long dear_iaddr:60;	/* instruction address */
	} pmd17_ita_reg;
} pfm_ita_pmd_reg_t;

/*
 * type definition for Itanium instruction set support
 */
typedef enum {
	PFMLIB_ITA_ISM_BOTH=0, 	/* IA-32 and IA-64 (default) */
	PFMLIB_ITA_ISM_IA32=1, 	/* IA-32 only */
	PFMLIB_ITA_ISM_IA64=2 	/* IA-64 only */
} pfmlib_ita_ism_t;

typedef struct {
	unsigned int 	 flags; /* counter specific flags */
	unsigned int	 thres;	/* per event threshold */
	pfmlib_ita_ism_t ism;	/* per event instruction set */
} pfmlib_ita_counter_t;

/*
 * counter specific flags
 */
#define PFMLIB_ITA_FL_EVT_NO_QUALCHECK	0x1 /* don't check qualifier constraints */

typedef struct {
	unsigned char	 opcm_used;	/* set to 1 if this opcode matcher is used */
	unsigned long	 pmc_val;	/* value of opcode matcher for PMC8 */
} pfmlib_ita_opcm_t;

/*
 *
 * The BTB can be configured via 4 different methods:
 *
 * 	- BRANCH_EVENT is in the event list, pfp_ita_btb.btb_used == 0:
 * 		The BTB will be configured (PMC12) to record all branches AND a counting
 * 		monitor will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is in the event list, pfp_ita_btb.btb_used == 1:
 * 		The BTB will be configured (PMC12) according to information in pfp_ita_btb AND
 * 		a counter will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_ita_btb.btb_used == 0:
 * 	   	Nothing is programmed
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_ita_btb.btb_used == 1:
 * 		The BTB will be configured (PMC12) according to information in pfp_ita_btb.
 * 		This is the free running BTB mode.
 */
typedef struct {
	unsigned char	 btb_used;	/* set to 1 if the BTB is used */

	unsigned char	 btb_tar;
	unsigned char	 btb_tac;
	unsigned char	 btb_bac;
	unsigned char	 btb_tm;
	unsigned char	 btb_ptm;
	unsigned char	 btb_ppm;
	unsigned int	 btb_plm;	/* BTB privilege level mask */
} pfmlib_ita_btb_t;

/*
 * There are four ways to configure EAR:
 *
 * 	- an EAR event is in the event list AND pfp_ita_ear.ear_used = 0:
 * 		The EAR will be programmed (PMC10 or PMC11) based on the information encoded in the
 * 		event (umask, cache, tlb). A counting monitor will be programmed to
 * 		count DATA_EAR_EVENTS or INSTRUCTION_EAR_EVENTS depending on the type of EAR.
 *
 * 	- an EAR event is in the event list AND pfp_ita_ear.ear_used = 1:
 * 		The EAR will be programmed (PMC10 or PMC11) according to the information in the
 * 		pfp_ita_ear structure	because it contains more detailed information
 * 		(such as priv level and instruction set). A counting monitor will be programmed
 * 		to count DATA_EAR_EVENTS or INSTRUCTION_EAR_EVENTS depending on the type of EAR.
 *
 * 	- no EAR event is in the event list AND pfp_ita_ear.ear_used = 0:
 * 	 	Nothing is programmed.
 *
 * 	- no EAR event is in the event list AND pfp_ita_ear.ear_used = 1:
 * 		The EAR will be programmed (PMC10 or PMC11) according to the information in the
 * 		pfp_ita_ear structure. This is the free running mode for EAR
 */
typedef enum {
	PFMLIB_ITA_EAR_CACHE_MODE=0,	/* Cache mode : I-EAR and D-EAR */
	PFMLIB_ITA_EAR_TLB_MODE  =1, 	/* TLB mode   : I-EAR and D-EAR */
} pfmlib_ita_ear_mode_t;


typedef struct {
	unsigned char	 	ear_used;	/* when set will force definition of PMC[10] */

	pfmlib_ita_ear_mode_t	ear_mode;	/* EAR mode */
	pfmlib_ita_ism_t 	ear_ism;	/* instruction set */
	unsigned int	 	ear_plm;	/* IEAR privilege level mask */
	unsigned long	 	ear_umask;	/* umask value for PMC10 */
} pfmlib_ita_ear_t;

/*
 * describes one range. rr_plm is ignored for data ranges
 * a range is interpreted as unused (not defined) when rr_start = rr_end = 0.
 * if rr_plm is not set it will use the default settings set in the generic
 * library param structure.
 */
typedef struct {
	unsigned int            rr_flags;	/* currently unused */
	unsigned int		rr_plm;		/* privilege level (ignored for data ranges) */
	unsigned long		rr_start;	/* start address */
	unsigned long		rr_end;		/* end address (not included) */
} pfmlib_ita_input_rr_desc_t;

typedef struct {
	unsigned long		rr_soff;	/* output: start offset from actual start */
	unsigned long		rr_eoff;	/* output: end offset from actual end */
} pfmlib_ita_output_rr_desc_t;


/*
 * rr_used must be set to true for the library to configure the debug registers.
 * If using less than 4 intervals, must mark the end with entry: rr_limits[x].rr_start = rr_limits[x].rr_end = 0
 */
typedef struct {
	unsigned char	 		rr_used;	/* set if address range restriction is used */
	unsigned int			rr_flags;	/* set of flags for all ranges              */
	unsigned int			rr_nbr_used;	/* how many registers were used (output)    */
	pfmlib_ita_input_rr_desc_t	rr_limits[4];	/* at most 4 distinct intervals             */
} pfmlib_ita_input_rr_t;

typedef struct {
	unsigned int			rr_nbr_used;	/* how many registers were used (output)    */
	pfmlib_ita_output_rr_desc_t	rr_infos[4];	/* at most 4 distinct intervals             */
	pfmlib_reg_t			rr_br[8];	/* array of debug reg requests to configure */
} pfmlib_ita_output_rr_t;


/*
 * Itanium specific parameters for the library
 */
typedef struct {
	pfmlib_ita_counter_t	pfp_ita_counters[PMU_ITA_NUM_COUNTERS];	/* extended counter features */

	unsigned long		pfp_ita_flags;		/* Itanium specific flags */

	pfmlib_ita_opcm_t	pfp_ita_pmc8;		/* PMC8 (opcode matcher) configuration */
	pfmlib_ita_opcm_t	pfp_ita_pmc9;		/* PMC9 (opcode matcher) configuration */
	pfmlib_ita_ear_t	pfp_ita_iear;		/* IEAR configuration */
	pfmlib_ita_ear_t	pfp_ita_dear;		/* DEAR configuration */
	pfmlib_ita_btb_t	pfp_ita_btb;		/* BTB configuration */

	pfmlib_ita_input_rr_t	pfp_ita_drange;		/* data range restrictions */
	pfmlib_ita_input_rr_t	pfp_ita_irange;		/* code range restrictions */
	unsigned long		reserved[1];		/* for future use */
} pfmlib_ita_input_param_t;

typedef struct {
	pfmlib_ita_output_rr_t	pfp_ita_drange;		/* data range restrictions */
	pfmlib_ita_output_rr_t	pfp_ita_irange;		/* code range restrictions */
	unsigned long		reserved[6];		/* for future use */
} pfmlib_ita_output_param_t;

extern int pfm_ita_is_ear(unsigned int i);
extern int pfm_ita_is_dear(unsigned int i);
extern int pfm_ita_is_dear_tlb(unsigned int i);
extern int pfm_ita_is_dear_cache(unsigned int i);
extern int pfm_ita_is_iear(unsigned int i);
extern int pfm_ita_is_iear_tlb(unsigned int i);
extern int pfm_ita_is_iear_cache(unsigned int i);
extern int pfm_ita_is_btb(unsigned int i);
extern int pfm_ita_support_opcm(unsigned int i);
extern int pfm_ita_support_iarr(unsigned int i);
extern int pfm_ita_support_darr(unsigned int i);
extern int pfm_ita_get_ear_mode(unsigned int i, pfmlib_ita_ear_mode_t *m);

extern int pfm_ita_get_event_maxincr(unsigned int i, unsigned int *maxincr);
extern int pfm_ita_get_event_umask(unsigned int i, unsigned long *umask);

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_ITANIUM_H__ */
