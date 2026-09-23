/*
 * Dual-Core Itanium 2 PMU specific types and definitions
 *
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
#ifndef __PFMLIB_MONTECITO_H__
#define __PFMLIB_MONTECITO_H__

#include <perfmon/pfmlib.h>
#include <endian.h>

#if BYTE_ORDER != LITTLE_ENDIAN
#error "this file only supports little endian environments"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define PMU_MONT_FIRST_COUNTER	4	/* index of first PMC/PMD counter */
#define PMU_MONT_NUM_COUNTERS	12	/* total numbers of PMC/PMD pairs used as counting monitors */
#define PMU_MONT_NUM_PMCS	27	/* total number of PMCS defined */
#define PMU_MONT_NUM_PMDS	36	/* total number of PMDS defined */
#define PMU_MONT_NUM_ETB	16	/* total number of PMDS in ETB  */
#define PMU_MONT_COUNTER_WIDTH	47	/* hardware counter bit width   */

/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 */
typedef union {
	unsigned long pmc_val;			/* complete register value */

	/* This is the Montecito-specific PMC layout for counters PMC4-PMC15 */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* ignored */
		unsigned long pmc_es:8;		/* event select */
		unsigned long pmc_umask:4;	/* unit mask */
		unsigned long pmc_thres:3;	/* threshold */
		unsigned long pmc_ig2:1;	/* ignored */
		unsigned long pmc_ism:2;	/* instruction set: must be 2  */
		unsigned long pmc_all:1;	/* 0=only self, 1=both threads */
		unsigned long pmc_i:1;		/* Invalidate */
		unsigned long pmc_s:1;		/* Shared */
		unsigned long pmc_e:1;		/* Exclusive */
		unsigned long pmc_m:1;		/* Modified */
		unsigned long pmc_res3:33;	/* reserved */
	} pmc_mont_counter_reg;

	/* opcode matchers mask registers */
	struct {
		unsigned long opcm_mask:41;	/* opcode mask */
		unsigned long opcm_ig1:7;	/* ignored */
		unsigned long opcm_b:1;		/* B-syllable  */
		unsigned long opcm_f:1;		/* F-syllable  */
		unsigned long opcm_i:1;		/* I-syllable  */
		unsigned long opcm_m:1;		/* M-syllable  */
		unsigned long opcm_ig2:4;	/* ignored */
		unsigned long opcm_inv:1;	/* inverse range for ibrp0 */
		unsigned long opcm_ig_ad:1;	/* ignore address range restrictions */
		unsigned long opcm_ig3:6;	/* ignored */
	} pmc32_34_mont_reg;

	/* opcode matchers match registers */
	struct {
		unsigned long opcm_match:41;	/* opcode match */
		unsigned long opcm_ig1:23;	/* ignored */
	} pmc33_35_mont_reg;

	/* opcode matcher config register */
	struct {
		unsigned long opcm_ch0_ig_opcm:1;	/* chan0 opcode constraint */
		unsigned long opcm_ch1_ig_opcm:1;	/* chan1 opcode constraint */
		unsigned long opcm_ch2_ig_opcm:1;	/* chan2 opcode constraint */
		unsigned long opcm_ch3_ig_opcm:1;	/* chan3 opcode constraint */
		unsigned long opcm_res:28;		/* reserved */
		unsigned long opcm_ig:32;		/* ignored */
	} pmc36_mont_reg;

	/*
	 * instruction event address register configuration (I-EAR)
	 *
	 * The register has two layouts depending on the value of the ct field.
	 * In cache mode(ct=1x):
	 * 	- ct is 1 bit, umask is 8 bits
	 * In TLB mode (ct=0x):
	 * 	- ct is 2 bits, umask is 7 bits
	 * ct=11 => cache mode using a latency filter with eighth bit set
	 * ct=01 => nothing monitored
	 *
	 * The ct=01 value is the only reason why we cannot fix the layout
	 * to ct 1 bit and umask 8 bits. Even though in TLB mode, only 6 bits
	 * are effectively used for the umask, if the user inadvertently sets
	 * a umask with the most significant bit set, it would be equivalent
	 * to no monitoring.
	 */
	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_umask:8;	/* event unit mask */
		unsigned long iear_ct:1;	/* =1 for i-cache */
		unsigned long iear_res:2;	/* reserved */
		unsigned long iear_ig:48;	/* ignored */
	} pmc37_mont_cache_reg;

	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_umask:7;	/* event unit mask */
		unsigned long iear_ct:2;	/* 00=i-tlb, 01=nothing 1x=illegal */
		unsigned long iear_res:50;	/* reserved */
	} pmc37_mont_tlb_reg;

	/* data event address register configuration (D-EAR) */
	struct {
		unsigned long dear_plm:4;	/* privilege level mask */
		unsigned long dear_ig1:2;	/* ignored */
		unsigned long dear_pm:1;	/* privileged monitor */
		unsigned long dear_mode:2;	/* mode */
		unsigned long dear_ig2:7;	/* ignored */
		unsigned long dear_umask:4;	/* unit mask */
		unsigned long dear_ig3:4;	/* ignored */
		unsigned long dear_ism:2;	/* instruction set: must be 2 */
		unsigned long dear_ig4:38;	/* ignored */
	} pmc40_mont_reg;

	/* IP event address register (IP-EAR) */
	struct {
		unsigned long ipear_plm:4;	/* privilege level mask */
		unsigned long ipear_ig1:2;	/* ignored */
		unsigned long ipear_pm:1;	/* privileged monitor */
		unsigned long ipear_ig2:1;	/* ignored */
		unsigned long ipear_mode:3;	/* mode */
		unsigned long ipear_delay:8;	/* delay */
		unsigned long ipear_ig3:45;	/* reserved */
	} pmc42_mont_reg;
			
	/* execution trace buffer configuration register (ETB) */
	struct {
		unsigned long etbc_plm:4;	/* privilege level */
		unsigned long etbc_res1:2;	/* reserved */
		unsigned long etbc_pm:1;	/* privileged monitor */
		unsigned long etbc_ds:1;	/* data selector */
		unsigned long etbc_tm:2;	/* taken mask */
		unsigned long etbc_ptm:2;	/* predicted taken address mask */
		unsigned long etbc_ppm:2;	/* predicted predicate mask */
		unsigned long etbc_brt:2;	/* branch type mask */
		unsigned long etbc_ig:48;	/* ignored */
	} pmc39_mont_reg;

	/* data address range configuration register */
	struct {
		unsigned long darc_res1:3;	/* reserved */
		unsigned long darc_cfg_dtag0:2;	/* constraints on dbrp0 */
		unsigned long darc_res2:6;	/* reserved */
		unsigned long darc_cfg_dtag1:2;	/* constraints on dbrp1 */
		unsigned long darc_res3:6;	/* reserved */
		unsigned long darc_cfg_dtag2:2;	/* constraints on dbrp2 */
		unsigned long darc_res4:6;	/* reserved */
		unsigned long darc_cfg_dtag3:2;	/* constraints on dbrp3 */
		unsigned long darc_res5:16;	/* reserved */
		unsigned long darc_ena_dbrp0:1;	/* enable constraints dbrp0 */
		unsigned long darc_ena_dbrp1:1;	/* enable constraints dbrp1 */
		unsigned long darc_ena_dbrp2:1;	/* enable constraints dbrp2 */
		unsigned long darc_ena_dbrp3:1; /* enable constraint dbr3 */
		unsigned long darc_res6:15;
	} pmc41_mont_reg;

	/* instruction address range configuration register */
	struct {
		unsigned long iarc_res1:1;	/* reserved */
		unsigned long iarc_ig_ibrp0:1;	/* constrained by ibrp0 */
		unsigned long iarc_res2:2;	/* reserved */
		unsigned long iarc_ig_ibrp1:1;	/* constrained by ibrp1 */
		unsigned long iarc_res3:2;	/* reserved */
		unsigned long iarc_ig_ibrp2:1;	/* constrained by ibrp2 */
		unsigned long iarc_res4:2;	/* reserved */
		unsigned long iarc_ig_ibrp3:1;	/* constrained by ibrp3 */
		unsigned long iarc_res5:2;	/* reserved */
		unsigned long iarc_fine:1;	/* fine mode */
		unsigned long iarc_ig6:50;	/* reserved */
	} pmc38_mont_reg;

} pfm_mont_pmc_reg_t;

typedef union {
	unsigned long	pmd_val;	/* counter value */

	/* counting pmd register */
	struct {
		unsigned long pmd_count:47;	/* 47-bit hardware counter  */
		unsigned long pmd_sxt47:17;	/* sign extension of bit 46 */
	} pmd_mont_counter_reg;

	/* data event address register */
	struct {
		unsigned long dear_daddr;	/* data address */
	} pmd32_mont_reg;

	/* data event address register (D-EAR) */
	struct {
		unsigned long dear_latency:13;	/* latency  */
		unsigned long dear_ov:1;	/* latency overflow */
		unsigned long dear_stat:2;	/* status   */
		unsigned long dear_ig:48;	/* ignored */
	} pmd33_mont_reg;

	/* instruction event address register (I-EAR) */
	struct {
		unsigned long iear_stat:2;	/* status bit */
		unsigned long iear_ig:3;	/* ignored */
		unsigned long iear_iaddr:59;	/* instruction cache line address {60:51} sxt {50}*/
	} pmd34_mont_reg;

	/* instruction event address register (I-EAR) */
	struct {
		unsigned long iear_latency:12;	/* latency */
		unsigned long iear_ov:1;	/* latency overflow */
		unsigned long iear_ig:51;	/* ignored */
	} pmd35_mont_reg;

	/* data event address register (D-EAR) */
	struct {
		unsigned long dear_slot:2;	/* slot */
		unsigned long dear_bn:1;	/* bundle bit (if 1 add 16 to iaddr) */
		unsigned long dear_vl:1;	/* valid */
		unsigned long dear_iaddr:60;	/* instruction address (2-bundle window)*/
	} pmd36_mont_reg;

	/* execution trace buffer index register (ETB) */
	struct {
		unsigned long etbi_ebi:4;	/* next entry index  */
		unsigned long etbi_ig1:1;	/* ignored */
		unsigned long etbi_full:1;	/* ETB overflowed at least once */
		unsigned long etbi_ig2:58;	/* ignored */
	} pmd38_mont_reg;

	/* execution trace buffer extension register (ETB) */
	struct {
		unsigned long etb_pmd48ext_b1:1;	/* pmd48 ext  */
		unsigned long etb_pmd48ext_bruflush:1;	/* pmd48 ext  */
		unsigned long etb_pmd48ext_res:2;	/* reserved   */

		unsigned long etb_pmd56ext_b1:1;	/* pmd56 ext */
		unsigned long etb_pmd56ext_bruflush:1;	/* pmd56 ext */
		unsigned long etb_pmd56ext_res:2;	/* reserved  */ 

		unsigned long etb_pmd49ext_b1:1;	/* pmd49 ext  */
		unsigned long etb_pmd49ext_bruflush:1;	/* pmd49 ext  */
		unsigned long etb_pmd49ext_res:2;	/* reserved   */

		unsigned long etb_pmd57ext_b1:1;	/* pmd57 ext */
		unsigned long etb_pmd57ext_bruflush:1;	/* pmd57 ext */
		unsigned long etb_pmd57ext_res:2;	/* reserved  */

		unsigned long etb_pmd50ext_b1:1;	/* pmd50 ext */
		unsigned long etb_pmd50ext_bruflush:1;	/* pmd50 ext */
		unsigned long etb_pmd50ext_res:2;	/* reserved  */

		unsigned long etb_pmd58ext_b1:1;	/* pmd58 ext */
		unsigned long etb_pmd58ext_bruflush:1;	/* pmd58 ext */
		unsigned long etb_pmd58ext_res:2;	/* reserved  */

		unsigned long etb_pmd51ext_b1:1;	/* pmd51 ext */
		unsigned long etb_pmd51ext_bruflush:1;	/* pmd51 ext */
		unsigned long etb_pmd51ext_res:2;	/* reserved  */

		unsigned long etb_pmd59ext_b1:1;	/* pmd59 ext */
		unsigned long etb_pmd59ext_bruflush:1;	/* pmd59 ext */
		unsigned long etb_pmd59ext_res:2;	/* reserved  */

		unsigned long etb_pmd52ext_b1:1;	/* pmd52 ext */
		unsigned long etb_pmd52ext_bruflush:1;	/* pmd52 ext */
		unsigned long etb_pmd52ext_res:2;	/* reserved  */

		unsigned long etb_pmd60ext_b1:1;	/* pmd60 ext */
		unsigned long etb_pmd60ext_bruflush:1;	/* pmd60 ext */
		unsigned long etb_pmd60ext_res:2;	/* reserved  */

		unsigned long etb_pmd53ext_b1:1;	/* pmd53 ext */
		unsigned long etb_pmd53ext_bruflush:1;	/* pmd53 ext */
		unsigned long etb_pmd53ext_res:2;	/* reserved  */

		unsigned long etb_pmd61ext_b1:1;	/* pmd61 ext */
		unsigned long etb_pmd61ext_bruflush:1;	/* pmd61 ext */
		unsigned long etb_pmd61ext_res:2;	/* reserved  */

		unsigned long etb_pmd54ext_b1:1;	/* pmd54 ext */
		unsigned long etb_pmd54ext_bruflush:1;	/* pmd54 ext */
		unsigned long etb_pmd54ext_res:2;	/* reserved  */

		unsigned long etb_pmd62ext_b1:1;	/* pmd62 ext */
		unsigned long etb_pmd62ext_bruflush:1;	/* pmd62 ext */
		unsigned long etb_pmd62ext_res:2;	/* reserved  */

		unsigned long etb_pmd55ext_b1:1;	/* pmd55 ext */
		unsigned long etb_pmd55ext_bruflush:1;	/* pmd55 ext */
		unsigned long etb_pmd55ext_res:2;	/* reserved  */

		unsigned long etb_pmd63ext_b1:1;	/* pmd63 ext */
		unsigned long etb_pmd63ext_bruflush:1;	/* pmd63 ext */
		unsigned long etb_pmd63ext_res:2;	/* reserved  */
	} pmd39_mont_reg;

	/*
	 * execution trace buffer extension register when used with IP-EAR
	 *
	 * to be used in conjunction with pmd48_63_ipear_reg (see  below)
	 */
	struct {
		unsigned long ipear_pmd48ext_cycles:2;	/* pmd48 upper 2 bits of cycles */
		unsigned long ipear_pmd48ext_f:1;	/* pmd48 flush bit    */
		unsigned long ipear_pmd48ext_ef:1;	/* pmd48 early freeze */

		unsigned long ipear_pmd56ext_cycles:2;	/* pmd56 upper 2 bits of cycles */
		unsigned long ipear_pmd56ext_f:1;	/* pmd56 flush bit    */
		unsigned long ipear_pmd56ext_ef:1;	/* pmd56 early freeze */

		unsigned long ipear_pmd49ext_cycles:2;	/* pmd49 upper 2 bits of cycles */
		unsigned long ipear_pmd49ext_f:1;	/* pmd49 flush bit    */
		unsigned long ipear_pmd49ext_ef:1;	/* pmd49 early freeze */

		unsigned long ipear_pmd57ext_cycles:2;	/* pmd57 upper 2 bits of cycles */
		unsigned long ipear_pmd57ext_f:1;	/* pmd57 flush bit    */
		unsigned long ipear_pmd57ext_ef:1;	/* pmd57 early freeze */

		unsigned long ipear_pmd50ext_cycles:2;	/* pmd50 upper 2 bits of cycles */
		unsigned long ipear_pmd50ext_f:1;	/* pmd50 flush bit    */
		unsigned long ipear_pmd50ext_ef:1;	/* pmd50 early freeze */

		unsigned long ipear_pmd58ext_cycles:2;	/* pmd58 upper 2 bits of cycles */
		unsigned long ipear_pmd58ext_f:1;	/* pmd58 flush bit    */
		unsigned long ipear_pmd58ext_ef:1;	/* pmd58 early freeze */

		unsigned long ipear_pmd51ext_cycles:2;	/* pmd51 upper 2 bits of cycles */
		unsigned long ipear_pmd51ext_f:1;	/* pmd51 flush bit    */
		unsigned long ipear_pmd51ext_ef:1;	/* pmd51 early freeze */

		unsigned long ipear_pmd59ext_cycles:2;	/* pmd59 upper 2 bits of cycles */
		unsigned long ipear_pmd59ext_f:1;	/* pmd59 flush bit    */
		unsigned long ipear_pmd59ext_ef:1;	/* pmd59 early freeze */

		unsigned long ipear_pmd52ext_cycles:2;	/* pmd52 upper 2 bits of cycles */
		unsigned long ipear_pmd52ext_f:1;	/* pmd52 flush bit    */
		unsigned long ipear_pmd52ext_ef:1;	/* pmd52 early freeze */

		unsigned long ipear_pmd60ext_cycles:2;	/* pmd60 upper 2 bits of cycles */
		unsigned long ipear_pmd60ext_f:1;	/* pmd60 flush bit    */
		unsigned long ipear_pmd60ext_ef:1;	/* pmd60  early freeze */

		unsigned long ipear_pmd53ext_cycles:2;	/* pmd53 upper 2 bits of cycles */
		unsigned long ipear_pmd53ext_f:1;	/* pmd53 flush bit    */
		unsigned long ipear_pmd53ext_ef:1;	/* pmd53 early freeze */

		unsigned long ipear_pmd61ext_cycles:2;	/* pmd61 upper 2 bits of cycles */
		unsigned long ipear_pmd61ext_f:1;	/* pmd61 flush bit    */
		unsigned long ipear_pmd61ext_ef:1;	/* pmd61 early freeze */

		unsigned long ipear_pmd54ext_cycles:2;	/* pmd54 upper 2 bits of cycles */
		unsigned long ipear_pmd54ext_f:1;	/* pmd54 flush bit    */
		unsigned long ipear_pmd54ext_ef:1;	/* pmd54 early freeze */

		unsigned long ipear_pmd62ext_cycles:2;	/* pmd62 upper 2 bits of cycles */
		unsigned long ipear_pmd62ext_f:1;	/* pmd62 flush bit    */
		unsigned long ipear_pmd62ext_ef:1;	/* pmd62 early freeze */

		unsigned long ipear_pmd55ext_cycles:2;	/* pmd55 upper 2 bits of cycles */
		unsigned long ipear_pmd55ext_f:1;	/* pmd55 flush bit    */
		unsigned long ipear_pmd55ext_ef:1;	/* pmd55 early freeze */

		unsigned long ipear_pmd63ext_cycles:2;	/* pmd63 upper 2 bits of cycles */
		unsigned long ipear_pmd63ext_f:1;	/* pmd63 flush bit    */
		unsigned long ipear_pmd63ext_ef:1;	/* pmd63 early freeze */
	} pmd39_ipear_mont_reg;

	/* 
	 * execution trace buffer data register (ETB)
	 *
	 * when pmc39.ds == 0: pmd48-63 contains branch targets
	 * when pmc39.ds == 1: pmd48-63 content is undefined
	 */
	struct {
		unsigned long etb_s:1;		/* source bit */
		unsigned long etb_mp:1;		/* mispredict bit */
		unsigned long etb_slot:2;	/* which slot, 3=not taken branch */
		unsigned long etb_addr:60;	/* bundle address(s=1), target address(s=0) */
	} pmd48_63_etb_mont_reg;

	/* 
	 * execution trace buffer when used with IP-EAR with PMD48-63.ef=0
	 *
	 * The cycles field straddles pmdXX and corresponding extension in
	 * pmd39 (pmd39_ipear_mont_reg). For instance, cycles for pmd48: 
	 *
	 * cycles= pmd39_ipear_mont_reg.etb_pmd48ext_cycles << 4
	 *       | pmd48_63_etb_ipear_mont_reg.etb_cycles
	 */
	struct {
		unsigned long	ipear_addr:60;	/* retired IP[63:4]      */
		unsigned long	ipear_cycles:4;	/* lower 4 bit of cycles */
	} pmd48_63_ipear_mont_reg;

	/* 
	 * execution trace buffer when used with IP-EAR with PMD48-63.ef=1
	 *
	 * The cycles field straddles pmdXX and corresponding extension in
	 * pmd39 (pmd39_ipear_mont_reg). For instance, cycles for pmd48: 
	 *
	 * cycles= pmd39_ipear_mont_reg.etb_pmd48ext_cycles << 4
	 *       | pmd48_63_etb_ipear_ef_mont_reg.etb_cycles
	 */

	struct {
		unsigned long	ipear_delay:8;	/* delay count           */
		unsigned long	ipear_addr:52;	/* retired IP[61:12]     */
		unsigned long	ipear_cycles:4;	/* lower 5 bit of cycles */
	} pmd48_63_ipear_ef_mont_reg;

} pfm_mont_pmd_reg_t;

typedef struct {
	unsigned int	  flags;	/* counter specific flags */
	unsigned int 	  thres;	/* per event threshold */
} pfmlib_mont_counter_t;

/*
 * counter specific flags
 */
#define PFMLIB_MONT_FL_EVT_NO_QUALCHECK	0x1 /* don't check qualifier constraints */
#define PFMLIB_MONT_FL_EVT_ALL_THRD	0x2 /* event measured for both threads */
#define PFMLIB_MONT_FL_EVT_ACTIVE_ONLY	0x4 /* measure the event only when the thread is active */
#define PFMLIB_MONT_FL_EVT_ALWAYS	0x8 /* measure the event at all times (active or inactive) */

/*
 *
 * The ETB can be configured via 4 different methods:
 *
 * 	- BRANCH_EVENT is in the event list, pfp_mont_etb.etb_used == 0:
 * 		The ETB will be configured (PMC12) to record all branches AND a counting
 * 		monitor will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is in the event list, pfp_mont_etb.etb_used == 1:
 * 		The ETB will be configured (PMC12) according to information in pfp_mont_etb AND
 * 		a counter will be setup to count BRANCH_EVENT.
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_mont_etb.etb_used == 0:
 * 	   	Nothing is programmed
 *
 * 	-  BRANCH_EVENT is NOT in the event list, pfp_mont_etb.etb_used == 1:
 * 		The ETB will be configured (PMC12) according to information in pfp_mont_etb.
 * 		This is the free running ETB mode.
 */
typedef struct {
	unsigned char	 etb_used;	/* set to 1 if the ETB is used */
	unsigned int	 etb_plm;	/* ETB privilege level mask */
	unsigned char	 etb_tm;	/* taken mask */
	unsigned char	 etb_ptm;	/* predicted target mask */
	unsigned char	 etb_ppm;	/* predicted predicate mask */
	unsigned char	 etb_brt;	/* branch type mask */
} pfmlib_mont_etb_t;

/*
 * There are four ways to configure EAR:
 *
 * 	- an EAR event is in the event list AND pfp_mont_?ear.ear_used = 0:
 * 		The EAR will be programmed (PMC37 or PMC40) based on the information encoded in the
 * 		event (umask, cache, tlb,alat). A counting monitor will be programmed to
 * 		count DATA_EAR_EVENTS or L1I_EAR_EVENTS depending on the type of EAR.
 *
 * 	- an EAR event is in the event list AND pfp_mont_?ear.ear_used = 1:
 * 		The EAR will be programmed (PMC37 or PMC40) according to the information in the
 * 		pfp_mont_?ear structure	because it contains more detailed information
 * 		(such as priv level and instruction set). A counting monitor will be programmed
 * 		to count DATA_EAR_EVENTS or L1I_EAR_EVENTS depending on the type of EAR.
 *
 * 	- no EAR event is in the event list AND pfp_mont_?ear.ear_used = 0:
 * 	 	Nothing is programmed.
 *
 * 	- no EAR event is in the event list AND pfp_mont_?ear.ear_used = 1:
 * 		The EAR will be programmed (PMC37 or PMC40) according to the information in the
 * 		pfp_mont_?ear structure. This is the free running mode for EAR
 */

typedef enum {
	PFMLIB_MONT_EAR_CACHE_MODE= 0,	/* Cache mode : I-EAR and D-EAR */
	PFMLIB_MONT_EAR_TLB_MODE  = 1, 	/* TLB mode   : I-EAR and D-EAR */
	PFMLIB_MONT_EAR_ALAT_MODE = 2	/* ALAT mode  : D-EAR only      */
} pfmlib_mont_ear_mode_t;

typedef struct {
	unsigned char		ear_used;	/* when set will force definition of PMC[10] */

	pfmlib_mont_ear_mode_t	ear_mode;	/* EAR mode */
	unsigned int		ear_plm;	/* IEAR privilege level mask */
	unsigned long		ear_umask;	/* umask value for PMC10 */
} pfmlib_mont_ear_t;

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
} pfmlib_mont_input_rr_desc_t;
typedef struct {
	unsigned long		rr_soff;	/* start offset from actual start */
	unsigned long		rr_eoff;	/* end offset from actual end */
} pfmlib_mont_output_rr_desc_t;

/*
 * rr_used must be set to true for the library to configure the debug registers.
 * rr_inv only applies when the rr_limits table contains ONLY 1 range.
 *
 * If using less than 4 intervals, must mark the end with entry: rr_start = rr_end = 0
 */
typedef struct {
	unsigned int			rr_flags;	/* set of flags for all ranges              */
	pfmlib_mont_input_rr_desc_t	rr_limits[4];	/* at most 4 distinct intervals             */
	unsigned char	 		rr_used;	/* set if address range restriction is used */
} pfmlib_mont_input_rr_t;
/*
 * rr_flags values:
 * 	PFMLIB_MONT_IRR_DEMAND_FETCH, PFMLIB_MONT_IRR_PREFETCH_MATCH to be used
 * 	ONLY in conunction with any of the following (dual) events:
 *
 * 	- ISB_BUNPAIRS_IN, L1I_FETCH_RAB_HIT, L1I_FETCH_ISB_HIT, L1I_FILLS
 * 
 *	PFMLIB_MONT_IRR_DEMAND_FETCH: declared interest in demand fetched cache
 *	line (force use of IBRP0)
 *
 *	PFMLIB_MONT_IRR_PREFETCH_MATCH: declared interest in regular prefetched cache
 *	line (force use of IBRP1)
 */
#define PFMLIB_MONT_RR_INV		0x1 /* inverse instruction ranges (iranges only) */
#define PFMLIB_MONT_RR_NO_FINE_MODE	0x2 /* force non fine mode for instruction ranges */
#define PFMLIB_MONT_IRR_DEMAND_FETCH	0x4 /* demand fetch only for dual events */
#define PFMLIB_MONT_IRR_PREFETCH_MATCH	0x8 /* regular prefetches for dual events */


typedef struct {
	unsigned int			rr_nbr_used;	/* how many registers were used */
	pfmlib_mont_output_rr_desc_t	rr_infos[4];	/* at most 4 distinct intervals */
	pfmlib_reg_t			rr_br[8];	/* debug reg to configure       */
} pfmlib_mont_output_rr_t;

typedef struct {
	unsigned char		opcm_used;	/* set when opcm is used */
	unsigned char		opcm_m;		/* M slot */
	unsigned char		opcm_i;		/* I slot */
	unsigned char		opcm_f;		/* F slot */
	unsigned char		opcm_b;		/* B slot */
	unsigned long		opcm_match;	/* match field */
	unsigned long		opcm_mask;	/* mask field */
} pfmlib_mont_opcm_t;

typedef struct {
	unsigned char			ipear_used;	/* set when ipear is used */
	unsigned int			ipear_plm;	/* IP-EAR privilege level mask */
	unsigned short			ipear_delay;	/* delay in cycles */
} pfmlib_mont_ipear_t;

/*
 * Montecito specific parameters for the library
 */
typedef struct {
	pfmlib_mont_counter_t	pfp_mont_counters[PMU_MONT_NUM_COUNTERS];	/* extended counter features */

	unsigned long		pfp_mont_flags;		/* Montecito specific flags */

	pfmlib_mont_opcm_t	pfp_mont_opcm1;		/* pmc32/pmc33 (opcode matcher) configuration */
	pfmlib_mont_opcm_t	pfp_mont_opcm2;		/* pmc34/pmc35 (opcode matcher) configuration */
	pfmlib_mont_ear_t	pfp_mont_iear;		/* IEAR configuration */
	pfmlib_mont_ear_t	pfp_mont_dear;		/* DEAR configuration */
	pfmlib_mont_etb_t	pfp_mont_etb;		/* ETB configuration */
	pfmlib_mont_ipear_t	pfp_mont_ipear;		/* IP-EAR configuration */
	pfmlib_mont_input_rr_t	pfp_mont_drange;	/* data range restrictions */
	pfmlib_mont_input_rr_t	pfp_mont_irange;	/* code range restrictions */
	unsigned long		reserved[1];		/* for future use */
} pfmlib_mont_input_param_t;

typedef struct {
	pfmlib_mont_output_rr_t	pfp_mont_drange;	/* data range restrictions */
	pfmlib_mont_output_rr_t	pfp_mont_irange;	/* code range restrictions */
	unsigned long		reserved[6];		/* for future use */
} pfmlib_mont_output_param_t;

extern int pfm_mont_is_ear(unsigned int i);
extern int pfm_mont_is_dear(unsigned int i);
extern int pfm_mont_is_dear_tlb(unsigned int i);
extern int pfm_mont_is_dear_cache(unsigned int i);
extern int pfm_mont_is_dear_alat(unsigned int i);
extern int pfm_mont_is_iear(unsigned int i);
extern int pfm_mont_is_iear_tlb(unsigned int i);
extern int pfm_mont_is_iear_cache(unsigned int i);
extern int pfm_mont_is_etb(unsigned int i);
extern int pfm_mont_support_opcm(unsigned int i);
extern int pfm_mont_support_iarr(unsigned int i);
extern int pfm_mont_support_darr(unsigned int i);
extern int pfm_mont_support_all(unsigned int i);
extern int pfm_mont_get_ear_mode(unsigned int i, pfmlib_mont_ear_mode_t *m);
extern int pfm_mont_irange_is_fine(pfmlib_output_param_t *outp, pfmlib_mont_output_param_t *mod_out);

extern int pfm_mont_get_event_maxincr(unsigned int i, unsigned int *maxincr);
extern int pfm_mont_get_event_umask(unsigned int i, unsigned long *umask);
extern int pfm_mont_get_event_group(unsigned int i, int *grp);
extern int pfm_mont_get_event_set(unsigned int i, int *set);
extern int pfm_mont_get_event_type(unsigned int i, int *type);

/*
 * values of group (grp) returned by pfm_mont_get_event_group()
 */
#define PFMLIB_MONT_EVT_NO_GRP		 0 /* event does not belong to a group */
#define PFMLIB_MONT_EVT_L1D_CACHE_GRP	 1 /* event belongs to L1D Cache group */
#define PFMLIB_MONT_EVT_L2D_CACHE_GRP	 2 /* event belongs to L2D Cache group */

/*
 * possible values returned in set by pfm_mont_get_event_set()
 */
#define PFMLIB_MONT_EVT_NO_SET		-1 /* event does not belong to a set */

/*
 * values of type returned by pfm_mont_get_event_type()
 */
#define PFMLIB_MONT_EVT_ACTIVE		 0 /* event measures only when thread is active */
#define PFMLIB_MONT_EVT_FLOATING	 1
#define PFMLIB_MONT_EVT_CAUSAL		 2
#define PFMLIB_MONT_EVT_SELF_FLOATING	 3 /* floating with .self, causal otherwise */

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_MONTECITO_H__ */
