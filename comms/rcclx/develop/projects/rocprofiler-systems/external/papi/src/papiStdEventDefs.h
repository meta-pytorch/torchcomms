
/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful
in tuning application performance. These events have identical
assignments in the header files on different platforms however they
may differ in their actual semantics. In addition, all of these events
are not guaranteed to be present on all platforms.  Please check your
platform's documentation carefully.

*/
#ifndef _PAPISTDEVENTDEFS
#define _PAPISTDEVENTDEFS

/*
   Masks to indicate the event is a preset- the presets will have 
   the high bit set to one, as the vendors probably won't use the 
   higher numbers for the native events 
   This causes a problem for signed ints on 64 bit systems, since the
   'high bit' is no longer the high bit. An alternative is to AND
   with PAPI_PRESET_AND_MASK) instead of XOR with PAPI_PRESET_MASK to isolate
   the event bits.
   Native events for a specific platform can be defined by setting
   the next-highest bit. This gives PAPI a standardized way of 
   differentiating native events from preset events for query
   functions, etc.
*/

#define PAPI_PRESET_MASK     ((int)0x80000000)
#define PAPI_NATIVE_MASK     ((int)0x40000000)
#define PAPI_UE_MASK		 ((int)0xC0000000)
#define PAPI_PRESET_AND_MASK 0x7FFFFFFF
#define PAPI_NATIVE_AND_MASK 0xBFFFFFFF	/* this masks just the native bit */
#define PAPI_UE_AND_MASK     0x3FFFFFFF

#define PAPI_MAX_PRESET_EVENTS 128		/*The maxmimum number of preset events */
#define PAPI_MAX_USER_EVENTS 50			/*The maxmimum number of user defined events */
#define USER_EVENT_OPERATION_LEN 512	/*The maximum length of the operation string for user defined events */

/*
   NOTE: The table below defines each entry in terms of a mask and an integer.
   The integers MUST be in consecutive order with no gaps.
   If an event is removed or added, all following events MUST be renumbered.
   One way to fix this would be to recast each #define in terms of the preceeding
   one instead of an absolute number. e.g.:
     #define PAPI_L1_ICM  (PAPI_L1_DCM + 1)
   That way inserting or deleting events would only affect the definition of one
   other event.
*/

enum
{
	PAPI_L1_DCM_idx = 0,			   /*Level 1 data cache misses */
	PAPI_L1_ICM_idx,		 /*Level 1 instruction cache misses */
	PAPI_L2_DCM_idx,		 /*Level 2 data cache misses */
	PAPI_L2_ICM_idx,		 /*Level 2 instruction cache misses */
	PAPI_L3_DCM_idx,		 /*Level 3 data cache misses */
	PAPI_L3_ICM_idx,		 /*Level 3 instruction cache misses */
	PAPI_L1_TCM_idx,		 /*Level 1 total cache misses */
	PAPI_L2_TCM_idx,		 /*Level 2 total cache misses */
	PAPI_L3_TCM_idx,		 /*Level 3 total cache misses */
	PAPI_CA_SNP_idx,		 /*Snoops */
	PAPI_CA_SHR_idx,		 /*Request for shared cache line (SMP) */
	PAPI_CA_CLN_idx,		 /*Request for clean cache line (SMP) */
	PAPI_CA_INV_idx,		 /*Request for cache line Invalidation (SMP) */
	PAPI_CA_ITV_idx,		 /*Request for cache line Intervention (SMP) */
	PAPI_L3_LDM_idx,		 /*Level 3 load misses */
	PAPI_L3_STM_idx,		 /*Level 3 store misses */
/* 0x10 */
	PAPI_BRU_IDL_idx,		 /*Cycles branch units are idle */
	PAPI_FXU_IDL_idx,		 /*Cycles integer units are idle */
	PAPI_FPU_IDL_idx,		 /*Cycles floating point units are idle */
	PAPI_LSU_IDL_idx,		 /*Cycles load/store units are idle */
	PAPI_TLB_DM_idx,		 /*Data translation lookaside buffer misses */
	PAPI_TLB_IM_idx,		 /*Instr translation lookaside buffer misses */
	PAPI_TLB_TL_idx,		 /*Total translation lookaside buffer misses */
	PAPI_L1_LDM_idx,		 /*Level 1 load misses */
	PAPI_L1_STM_idx,		 /*Level 1 store misses */
	PAPI_L2_LDM_idx,		 /*Level 2 load misses */
	PAPI_L2_STM_idx,		 /*Level 2 store misses */
	PAPI_BTAC_M_idx,		 /*BTAC miss */
	PAPI_PRF_DM_idx,		 /*Prefetch data instruction caused a miss */
	PAPI_L3_DCH_idx,		 /*Level 3 Data Cache Hit */
	PAPI_TLB_SD_idx,		 /*Xlation lookaside buffer shootdowns (SMP) */
	PAPI_CSR_FAL_idx,		 /*Failed store conditional instructions */
/* 0x20 */
	PAPI_CSR_SUC_idx,		 /*Successful store conditional instructions */
	PAPI_CSR_TOT_idx,		 /*Total store conditional instructions */
	PAPI_MEM_SCY_idx,		 /*Cycles Stalled Waiting for Memory Access */
	PAPI_MEM_RCY_idx,		 /*Cycles Stalled Waiting for Memory Read */
	PAPI_MEM_WCY_idx,		 /*Cycles Stalled Waiting for Memory Write */
	PAPI_STL_ICY_idx,		 /*Cycles with No Instruction Issue */
	PAPI_FUL_ICY_idx,		 /*Cycles with Maximum Instruction Issue */
	PAPI_STL_CCY_idx,		 /*Cycles with No Instruction Completion */
	PAPI_FUL_CCY_idx,		 /*Cycles with Maximum Instruction Completion */
	PAPI_HW_INT_idx,		 /*Hardware interrupts */
	PAPI_BR_UCN_idx,		 /*Unconditional branch instructions executed */
	PAPI_BR_CN_idx,			 /*Conditional branch instructions executed */
	PAPI_BR_TKN_idx,		 /*Conditional branch instructions taken */
	PAPI_BR_NTK_idx,		 /*Conditional branch instructions not taken */
	PAPI_BR_MSP_idx,		 /*Conditional branch instructions mispred */
	PAPI_BR_PRC_idx,		 /*Conditional branch instructions corr. pred */
/* 0x30 */
	PAPI_FMA_INS_idx,		 /*FMA instructions completed */
	PAPI_TOT_IIS_idx,		 /*Total instructions issued */
	PAPI_TOT_INS_idx,		 /*Total instructions executed */
	PAPI_INT_INS_idx,		 /*Integer instructions executed */
	PAPI_FP_INS_idx,		 /*Floating point instructions executed */
	PAPI_LD_INS_idx,		 /*Load instructions executed */
	PAPI_SR_INS_idx,		 /*Store instructions executed */
	PAPI_BR_INS_idx,		 /*Total branch instructions executed */
	PAPI_VEC_INS_idx,		 /*Vector/SIMD instructions executed (could include integer) */
	PAPI_RES_STL_idx,		 /*Cycles processor is stalled on resource */
	PAPI_FP_STAL_idx,		 /*Cycles any FP units are stalled */
	PAPI_TOT_CYC_idx,		 /*Total cycles executed */
	PAPI_LST_INS_idx,		 /*Total load/store inst. executed */
	PAPI_SYC_INS_idx,		 /*Sync. inst. executed */
	PAPI_L1_DCH_idx,		 /*L1 D Cache Hit */
	PAPI_L2_DCH_idx,		 /*L2 D Cache Hit */
	/* 0x40 */
	PAPI_L1_DCA_idx,		 /*L1 D Cache Access */
	PAPI_L2_DCA_idx,		 /*L2 D Cache Access */
	PAPI_L3_DCA_idx,		 /*L3 D Cache Access */
	PAPI_L1_DCR_idx,		 /*L1 D Cache Read */
	PAPI_L2_DCR_idx,		 /*L2 D Cache Read */
	PAPI_L3_DCR_idx,		 /*L3 D Cache Read */
	PAPI_L1_DCW_idx,		 /*L1 D Cache Write */
	PAPI_L2_DCW_idx,		 /*L2 D Cache Write */
	PAPI_L3_DCW_idx,		 /*L3 D Cache Write */
	PAPI_L1_ICH_idx,		 /*L1 instruction cache hits */
	PAPI_L2_ICH_idx,		 /*L2 instruction cache hits */
	PAPI_L3_ICH_idx,		 /*L3 instruction cache hits */
	PAPI_L1_ICA_idx,		 /*L1 instruction cache accesses */
	PAPI_L2_ICA_idx,		 /*L2 instruction cache accesses */
	PAPI_L3_ICA_idx,		 /*L3 instruction cache accesses */
	PAPI_L1_ICR_idx,		 /*L1 instruction cache reads */
	/* 0x50 */
	PAPI_L2_ICR_idx,		 /*L2 instruction cache reads */
	PAPI_L3_ICR_idx,		 /*L3 instruction cache reads */
	PAPI_L1_ICW_idx,		 /*L1 instruction cache writes */
	PAPI_L2_ICW_idx,		 /*L2 instruction cache writes */
	PAPI_L3_ICW_idx,		 /*L3 instruction cache writes */
	PAPI_L1_TCH_idx,		 /*L1 total cache hits */
	PAPI_L2_TCH_idx,		 /*L2 total cache hits */
	PAPI_L3_TCH_idx,		 /*L3 total cache hits */
	PAPI_L1_TCA_idx,		 /*L1 total cache accesses */
	PAPI_L2_TCA_idx,		 /*L2 total cache accesses */
	PAPI_L3_TCA_idx,		 /*L3 total cache accesses */
	PAPI_L1_TCR_idx,		 /*L1 total cache reads */
	PAPI_L2_TCR_idx,		 /*L2 total cache reads */
	PAPI_L3_TCR_idx,		 /*L3 total cache reads */
	PAPI_L1_TCW_idx,		 /*L1 total cache writes */
	PAPI_L2_TCW_idx,		 /*L2 total cache writes */
	/* 0x60 */
	PAPI_L3_TCW_idx,		 /*L3 total cache writes */
	PAPI_FML_INS_idx,		 /*FM ins */
	PAPI_FAD_INS_idx,		 /*FA ins */
	PAPI_FDV_INS_idx,		 /*FD ins */
	PAPI_FSQ_INS_idx,		 /*FSq ins */
	PAPI_FNV_INS_idx,		 /*Finv ins */
	PAPI_FP_OPS_idx,		 /*Floating point operations executed */
	PAPI_SP_OPS_idx,		 /* Floating point operations executed; optimized to count scaled single precision vector operations */
	PAPI_DP_OPS_idx,		 /* Floating point operations executed; optimized to count scaled double precision vector operations */
	PAPI_VEC_SP_idx,		 /* Single precision vector/SIMD instructions */
	PAPI_VEC_DP_idx,		 /* Double precision vector/SIMD instructions */
	PAPI_REF_CYC_idx,		 /* Reference clock cycles */
	PAPI_END_idx			 /*This should always be last! */
};

#define PAPI_L1_DCM  (PAPI_L1_DCM_idx  | PAPI_PRESET_MASK)	/*Level 1 data cache misses */
#define PAPI_L1_ICM  (PAPI_L1_ICM_idx  | PAPI_PRESET_MASK)	/*Level 1 instruction cache misses */
#define PAPI_L2_DCM  (PAPI_L2_DCM_idx  | PAPI_PRESET_MASK)	/*Level 2 data cache misses */
#define PAPI_L2_ICM  (PAPI_L2_ICM_idx  | PAPI_PRESET_MASK)	/*Level 2 instruction cache misses */
#define PAPI_L3_DCM  (PAPI_L3_DCM_idx  | PAPI_PRESET_MASK)	/*Level 3 data cache misses */
#define PAPI_L3_ICM  (PAPI_L3_ICM_idx  | PAPI_PRESET_MASK)	/*Level 3 instruction cache misses */
#define PAPI_L1_TCM  (PAPI_L1_TCM_idx  | PAPI_PRESET_MASK)	/*Level 1 total cache misses */
#define PAPI_L2_TCM  (PAPI_L2_TCM_idx  | PAPI_PRESET_MASK)	/*Level 2 total cache misses */
#define PAPI_L3_TCM  (PAPI_L3_TCM_idx  | PAPI_PRESET_MASK)	/*Level 3 total cache misses */
#define PAPI_CA_SNP  (PAPI_CA_SNP_idx  | PAPI_PRESET_MASK)	/*Snoops */
#define PAPI_CA_SHR  (PAPI_CA_SHR_idx  | PAPI_PRESET_MASK)	/*Request for shared cache line (SMP) */
#define PAPI_CA_CLN  (PAPI_CA_CLN_idx  | PAPI_PRESET_MASK)	/*Request for clean cache line (SMP) */
#define PAPI_CA_INV  (PAPI_CA_INV_idx  | PAPI_PRESET_MASK)	/*Request for cache line Invalidation (SMP) */
#define PAPI_CA_ITV  (PAPI_CA_ITV_idx  | PAPI_PRESET_MASK)	/*Request for cache line Intervention (SMP) */
#define PAPI_L3_LDM  (PAPI_L3_LDM_idx  | PAPI_PRESET_MASK)	/*Level 3 load misses */
#define PAPI_L3_STM  (PAPI_L3_STM_idx  | PAPI_PRESET_MASK)	/*Level 3 store misses */
#define PAPI_BRU_IDL (PAPI_BRU_IDL_idx | PAPI_PRESET_MASK)	/*Cycles branch units are idle */
#define PAPI_FXU_IDL (PAPI_FXU_IDL_idx | PAPI_PRESET_MASK)	/*Cycles integer units are idle */
#define PAPI_FPU_IDL (PAPI_FPU_IDL_idx | PAPI_PRESET_MASK)	/*Cycles floating point units are idle */
#define PAPI_LSU_IDL (PAPI_LSU_IDL_idx | PAPI_PRESET_MASK)	/*Cycles load/store units are idle */
#define PAPI_TLB_DM  (PAPI_TLB_DM_idx  | PAPI_PRESET_MASK)	/*Data translation lookaside buffer misses */
#define PAPI_TLB_IM  (PAPI_TLB_IM_idx  | PAPI_PRESET_MASK)	/*Instr translation lookaside buffer misses */
#define PAPI_TLB_TL  (PAPI_TLB_TL_idx  | PAPI_PRESET_MASK)	/*Total translation lookaside buffer misses */
#define PAPI_L1_LDM  (PAPI_L1_LDM_idx  | PAPI_PRESET_MASK)	/*Level 1 load misses */
#define PAPI_L1_STM  (PAPI_L1_STM_idx  | PAPI_PRESET_MASK)	/*Level 1 store misses */
#define PAPI_L2_LDM  (PAPI_L2_LDM_idx  | PAPI_PRESET_MASK)	/*Level 2 load misses */
#define PAPI_L2_STM  (PAPI_L2_STM_idx  | PAPI_PRESET_MASK)	/*Level 2 store misses */
#define PAPI_BTAC_M  (PAPI_BTAC_M_idx  | PAPI_PRESET_MASK)	/*BTAC miss */
#define PAPI_PRF_DM  (PAPI_PRF_DM_idx  | PAPI_PRESET_MASK)	/*Prefetch data instruction caused a miss */
#define PAPI_L3_DCH  (PAPI_L3_DCH_idx  | PAPI_PRESET_MASK)	/*Level 3 Data Cache Hit */
#define PAPI_TLB_SD  (PAPI_TLB_SD_idx  | PAPI_PRESET_MASK)	/*Xlation lookaside buffer shootdowns (SMP) */
#define PAPI_CSR_FAL (PAPI_CSR_FAL_idx | PAPI_PRESET_MASK)	/*Failed store conditional instructions */
#define PAPI_CSR_SUC (PAPI_CSR_SUC_idx | PAPI_PRESET_MASK)	/*Successful store conditional instructions */
#define PAPI_CSR_TOT (PAPI_CSR_TOT_idx | PAPI_PRESET_MASK)	/*Total store conditional instructions */
#define PAPI_MEM_SCY (PAPI_MEM_SCY_idx | PAPI_PRESET_MASK)	/*Cycles Stalled Waiting for Memory Access */
#define PAPI_MEM_RCY (PAPI_MEM_RCY_idx | PAPI_PRESET_MASK)	/*Cycles Stalled Waiting for Memory Read */
#define PAPI_MEM_WCY (PAPI_MEM_WCY_idx | PAPI_PRESET_MASK)	/*Cycles Stalled Waiting for Memory Write */
#define PAPI_STL_ICY (PAPI_STL_ICY_idx | PAPI_PRESET_MASK)	/*Cycles with No Instruction Issue */
#define PAPI_FUL_ICY (PAPI_FUL_ICY_idx | PAPI_PRESET_MASK)	/*Cycles with Maximum Instruction Issue */
#define PAPI_STL_CCY (PAPI_STL_CCY_idx | PAPI_PRESET_MASK)	/*Cycles with No Instruction Completion */
#define PAPI_FUL_CCY (PAPI_FUL_CCY_idx | PAPI_PRESET_MASK)	/*Cycles with Maximum Instruction Completion */
#define PAPI_HW_INT  (PAPI_HW_INT_idx  | PAPI_PRESET_MASK)	/*Hardware interrupts */
#define PAPI_BR_UCN  (PAPI_BR_UCN_idx  | PAPI_PRESET_MASK)	/*Unconditional branch instructions executed */
#define PAPI_BR_CN   (PAPI_BR_CN_idx   | PAPI_PRESET_MASK)	/*Conditional branch instructions executed */
#define PAPI_BR_TKN  (PAPI_BR_TKN_idx  | PAPI_PRESET_MASK)	/*Conditional branch instructions taken */
#define PAPI_BR_NTK  (PAPI_BR_NTK_idx  | PAPI_PRESET_MASK)	/*Conditional branch instructions not taken */
#define PAPI_BR_MSP  (PAPI_BR_MSP_idx  | PAPI_PRESET_MASK)	/*Conditional branch instructions mispred */
#define PAPI_BR_PRC  (PAPI_BR_PRC_idx  | PAPI_PRESET_MASK)	/*Conditional branch instructions corr. pred */
#define PAPI_FMA_INS (PAPI_FMA_INS_idx | PAPI_PRESET_MASK)	/*FMA instructions completed */
#define PAPI_TOT_IIS (PAPI_TOT_IIS_idx | PAPI_PRESET_MASK)	/*Total instructions issued */
#define PAPI_TOT_INS (PAPI_TOT_INS_idx | PAPI_PRESET_MASK)	/*Total instructions executed */
#define PAPI_INT_INS (PAPI_INT_INS_idx | PAPI_PRESET_MASK)	/*Integer instructions executed */
#define PAPI_FP_INS  (PAPI_FP_INS_idx  | PAPI_PRESET_MASK)	/*Floating point instructions executed */
#define PAPI_LD_INS  (PAPI_LD_INS_idx  | PAPI_PRESET_MASK)	/*Load instructions executed */
#define PAPI_SR_INS  (PAPI_SR_INS_idx  | PAPI_PRESET_MASK)	/*Store instructions executed */
#define PAPI_BR_INS  (PAPI_BR_INS_idx  | PAPI_PRESET_MASK)	/*Total branch instructions executed */
#define PAPI_VEC_INS (PAPI_VEC_INS_idx | PAPI_PRESET_MASK)	/*Vector/SIMD instructions executed (could include integer) */
#define PAPI_RES_STL (PAPI_RES_STL_idx | PAPI_PRESET_MASK)	/*Cycles processor is stalled on resource */
#define PAPI_FP_STAL (PAPI_FP_STAL_idx | PAPI_PRESET_MASK)	/*Cycles any FP units are stalled */
#define PAPI_TOT_CYC (PAPI_TOT_CYC_idx | PAPI_PRESET_MASK)	/*Total cycles executed */
#define PAPI_LST_INS (PAPI_LST_INS_idx | PAPI_PRESET_MASK)	/*Total load/store inst. executed */
#define PAPI_SYC_INS (PAPI_SYC_INS_idx | PAPI_PRESET_MASK)	/*Sync. inst. executed */
#define PAPI_L1_DCH  (PAPI_L1_DCH_idx  | PAPI_PRESET_MASK)	/*L1 D Cache Hit */
#define PAPI_L2_DCH  (PAPI_L2_DCH_idx  | PAPI_PRESET_MASK)	/*L2 D Cache Hit */
#define PAPI_L1_DCA  (PAPI_L1_DCA_idx  | PAPI_PRESET_MASK)	/*L1 D Cache Access */
#define PAPI_L2_DCA  (PAPI_L2_DCA_idx  | PAPI_PRESET_MASK)	/*L2 D Cache Access */
#define PAPI_L3_DCA  (PAPI_L3_DCA_idx  | PAPI_PRESET_MASK)	/*L3 D Cache Access */
#define PAPI_L1_DCR  (PAPI_L1_DCR_idx  | PAPI_PRESET_MASK)	/*L1 D Cache Read */
#define PAPI_L2_DCR  (PAPI_L2_DCR_idx  | PAPI_PRESET_MASK)	/*L2 D Cache Read */
#define PAPI_L3_DCR  (PAPI_L3_DCR_idx  | PAPI_PRESET_MASK)	/*L3 D Cache Read */
#define PAPI_L1_DCW  (PAPI_L1_DCW_idx  | PAPI_PRESET_MASK)	/*L1 D Cache Write */
#define PAPI_L2_DCW  (PAPI_L2_DCW_idx  | PAPI_PRESET_MASK)	/*L2 D Cache Write */
#define PAPI_L3_DCW  (PAPI_L3_DCW_idx  | PAPI_PRESET_MASK)	/*L3 D Cache Write */
#define PAPI_L1_ICH  (PAPI_L1_ICH_idx  | PAPI_PRESET_MASK)	/*L1 instruction cache hits */
#define PAPI_L2_ICH  (PAPI_L2_ICH_idx  | PAPI_PRESET_MASK)	/*L2 instruction cache hits */
#define PAPI_L3_ICH  (PAPI_L3_ICH_idx  | PAPI_PRESET_MASK)	/*L3 instruction cache hits */
#define PAPI_L1_ICA  (PAPI_L1_ICA_idx  | PAPI_PRESET_MASK)	/*L1 instruction cache accesses */
#define PAPI_L2_ICA  (PAPI_L2_ICA_idx  | PAPI_PRESET_MASK)	/*L2 instruction cache accesses */
#define PAPI_L3_ICA  (PAPI_L3_ICA_idx  | PAPI_PRESET_MASK)	/*L3 instruction cache accesses */
#define PAPI_L1_ICR  (PAPI_L1_ICR_idx  | PAPI_PRESET_MASK)	/*L1 instruction cache reads */
#define PAPI_L2_ICR  (PAPI_L2_ICR_idx  | PAPI_PRESET_MASK)	/*L2 instruction cache reads */
#define PAPI_L3_ICR  (PAPI_L3_ICR_idx  | PAPI_PRESET_MASK)	/*L3 instruction cache reads */
#define PAPI_L1_ICW  (PAPI_L1_ICW_idx  | PAPI_PRESET_MASK)	/*L1 instruction cache writes */
#define PAPI_L2_ICW  (PAPI_L2_ICW_idx  | PAPI_PRESET_MASK)	/*L2 instruction cache writes */
#define PAPI_L3_ICW  (PAPI_L3_ICW_idx  | PAPI_PRESET_MASK)	/*L3 instruction cache writes */
#define PAPI_L1_TCH  (PAPI_L1_TCH_idx  | PAPI_PRESET_MASK)	/*L1 total cache hits */
#define PAPI_L2_TCH  (PAPI_L2_TCH_idx  | PAPI_PRESET_MASK)	/*L2 total cache hits */
#define PAPI_L3_TCH  (PAPI_L3_TCH_idx  | PAPI_PRESET_MASK)	/*L3 total cache hits */
#define PAPI_L1_TCA  (PAPI_L1_TCA_idx  | PAPI_PRESET_MASK)	/*L1 total cache accesses */
#define PAPI_L2_TCA  (PAPI_L2_TCA_idx  | PAPI_PRESET_MASK)	/*L2 total cache accesses */
#define PAPI_L3_TCA  (PAPI_L3_TCA_idx  | PAPI_PRESET_MASK)	/*L3 total cache accesses */
#define PAPI_L1_TCR  (PAPI_L1_TCR_idx  | PAPI_PRESET_MASK)	/*L1 total cache reads */
#define PAPI_L2_TCR  (PAPI_L2_TCR_idx  | PAPI_PRESET_MASK)	/*L2 total cache reads */
#define PAPI_L3_TCR  (PAPI_L3_TCR_idx  | PAPI_PRESET_MASK)	/*L3 total cache reads */
#define PAPI_L1_TCW  (PAPI_L1_TCW_idx  | PAPI_PRESET_MASK)	/*L1 total cache writes */
#define PAPI_L2_TCW  (PAPI_L2_TCW_idx  | PAPI_PRESET_MASK)	/*L2 total cache writes */
#define PAPI_L3_TCW  (PAPI_L3_TCW_idx  | PAPI_PRESET_MASK)	/*L3 total cache writes */
#define PAPI_FML_INS (PAPI_FML_INS_idx | PAPI_PRESET_MASK)	/*FM ins */
#define PAPI_FAD_INS (PAPI_FAD_INS_idx | PAPI_PRESET_MASK)	/*FA ins */
#define PAPI_FDV_INS (PAPI_FDV_INS_idx | PAPI_PRESET_MASK)	/*FD ins */
#define PAPI_FSQ_INS (PAPI_FSQ_INS_idx | PAPI_PRESET_MASK)	/*FSq ins */
#define PAPI_FNV_INS (PAPI_FNV_INS_idx | PAPI_PRESET_MASK)	/*Finv ins */
#define PAPI_FP_OPS  (PAPI_FP_OPS_idx  | PAPI_PRESET_MASK)	/*Floating point operations executed */
#define PAPI_SP_OPS  (PAPI_SP_OPS_idx  | PAPI_PRESET_MASK)	/* Floating point operations executed; optimized to count scaled single precision vector operations */
#define PAPI_DP_OPS  (PAPI_DP_OPS_idx  | PAPI_PRESET_MASK)	/* Floating point operations executed; optimized to count scaled double precision vector operations */
#define PAPI_VEC_SP  (PAPI_VEC_SP_idx  | PAPI_PRESET_MASK)	/* Single precision vector/SIMD instructions */
#define PAPI_VEC_DP  (PAPI_VEC_DP_idx  | PAPI_PRESET_MASK)	/* Double precision vector/SIMD instructions */
#define PAPI_REF_CYC (PAPI_REF_CYC_idx  | PAPI_PRESET_MASK)	/* Reference clock cycles */

#define PAPI_END     (PAPI_END_idx  | PAPI_PRESET_MASK)	/*This should always be last! */

#endif

