static pme_gen_mips64_entry_t gen_mips64_20K_pe[] = {
	{.pme_name="INSN_REQ_FROM_IFU_TO_BIU",
	 .pme_code = 0x00000009,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction requests from the IFU to the BIU"
	},
	{.pme_name="BRANCHES_MISSPREDICTED",
	 .pme_code = 0x00000005,
	 .pme_counters = 0x1,
	 .pme_desc = "Branches that mispredicted before completing execution"
	},
	{.pme_name="REPLAYS",
	 .pme_code = 0x0000000b,
	 .pme_counters = 0x1,
	 .pme_desc = "Total number of LSU requested replays, Load-dependent speculative dispatch or FPU exception prediction replays."
	},
	{.pme_name="JR_INSNS_COMPLETED",
	 .pme_code = 0x0000000d,
	 .pme_counters = 0x1,
	 .pme_desc = "JR instruction that completed execution"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x1,
	 .pme_desc = "CPU cycles"
	},
	{.pme_name="REPLAY_DUE_TO_LOAD_DEPENDENT_SPEC_DISPATCH",
	 .pme_code = 0x00000008,
	 .pme_counters = 0x1,
	 .pme_desc = "Replays due to load-dependent speculative dispatch"
	},
	{.pme_name="LSU_REPLAYS",
	 .pme_code = 0x0000000e,
	 .pme_counters = 0x1,
	 .pme_desc = "LSU requested replays"
	},
	{.pme_name="FP_INSNS_COMPLETED",
	 .pme_code = 0x00000003,
	 .pme_counters = 0x1,
	 .pme_desc = "Instructions completed in FPU datapath (computational event"
	},
	{.pme_name="FPU_EXCEPTIONS_TAKEN",
	 .pme_code = 0x0000000a,
	 .pme_counters = 0x1,
	 .pme_desc = "Taken FPU exceptions"
	},
	{.pme_name="TLB_REFILLS_TAKEN",
	 .pme_code = 0x00000004,
	 .pme_counters = 0x1,
	 .pme_desc = "Taken TLB refill exceptions"
	},
	{.pme_name="RPS_MISSPREDICTS",
	 .pme_code = 0x0000000c,
	 .pme_counters = 0x1,
	 .pme_desc = "JR instructions that mispredicted using the Return Prediction Stack (RPS)"
	},
	{.pme_name="INSN_ISSUED",
	 .pme_code = 0x00000001,
	 .pme_counters = 0x1,
	 .pme_desc = "Dispatched/issued instructions"
	},
	{.pme_name="INSNS_COMPLETED",
	 .pme_code = 0x0000000f,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction that completed execution (with or without exception)"
	},
	{.pme_name="BRANCHES_COMPLETED",
	 .pme_code = 0x00000006,
	 .pme_counters = 0x1,
	 .pme_desc = "Branches that completed execution"
	},
	{.pme_name="JTLB_EXCEPTIONS",
	 .pme_code = 0x00000007,
	 .pme_counters = 0x1,
	 .pme_desc = "Taken Joint-TLB exceptions"
	},
	{.pme_name="FETCH_GROUPS",
	 .pme_code = 0x00000002,
	 .pme_counters = 0x1,
	 .pme_desc = "Fetch groups entering CPU execution pipes"
	},
};

static pme_gen_mips64_entry_t gen_mips64_24K_pe[] = {
	{.pme_name="DCACHE_MISS",
	 .pme_code = 0x00000b0b,
	 .pme_counters = 0x3,
	 .pme_desc = "Data cache misses"
	},
	{.pme_name="REPLAY_TRAPS_NOT_UTLB",
	 .pme_code = 0x00001200,
	 .pme_counters = 0x2,
	 .pme_desc = "``replay traps'' (other than micro-TLB related)"
	},
	{.pme_name="ITLB_ACCESSES",
	 .pme_code = 0x00000005,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction micro-TLB accesses"
	},
	{.pme_name="INSTRUCTIONS",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "Instructions completed"
	},
	{.pme_name="LOADS_COMPLETED",
	 .pme_code = 0x0000000f,
	 .pme_counters = 0x1,
	 .pme_desc = "Loads completed (including FP)"
	},
	{.pme_name="SC_COMPLETE_BUT_FAILED",
	 .pme_code = 0x00001300,
	 .pme_counters = 0x2,
	 .pme_desc = "sc instructions completed, but store failed (because the link bit had been cleared)."
	},
	{.pme_name="JTLB_DATA_MISSES",
	 .pme_code = 0x00000800,
	 .pme_counters = 0x2,
	 .pme_desc = "Joint TLB data (non-instruction) misses"
	},
	{.pme_name="L2_MISSES",
	 .pme_code = 0x00001616,
	 .pme_counters = 0x3,
	 .pme_desc = "L2 cache misses"
	},
	{.pme_name="SC_COMPLETED",
	 .pme_code = 0x00000013,
	 .pme_counters = 0x1,
	 .pme_desc = "sc instructions completed"
	},
	{.pme_name="SUPERFLUOUS_INSTRUCTIONS",
	 .pme_code = 0x00001400,
	 .pme_counters = 0x2,
	 .pme_desc = "``superfluous'' prefetch instructions (data was already in cache)."
	},
	{.pme_name="DCACHE_WRITEBACKS",
	 .pme_code = 0x00000a00,
	 .pme_counters = 0x2,
	 .pme_desc = "Data cache writebacks"
	},
	{.pme_name="JR_31_MISSPREDICTS",
	 .pme_code = 0x00000300,
	 .pme_counters = 0x2,
	 .pme_desc = "jr r31 (return) mispredictions"
	},
	{.pme_name="JTLB_DATA_ACCESSES",
	 .pme_code = 0x00000007,
	 .pme_counters = 0x1,
	 .pme_desc = "Joint TLB instruction accesses"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00000900,
	 .pme_counters = 0x2,
	 .pme_desc = "Instruction cache misses"
	},
	{.pme_name="STALLS",
	 .pme_code = 0x00000012,
	 .pme_counters = 0x1,
	 .pme_desc = "Stalls"
	},
	{.pme_name="INTEGER_INSNS_COMPLETED",
	 .pme_code = 0x0000000e,
	 .pme_counters = 0x1,
	 .pme_desc = "Integer instructions completed"
	},
	{.pme_name="INTEGER_MUL_DIV_COMPLETED",
	 .pme_code = 0x00001100,
	 .pme_counters = 0x2,
	 .pme_desc = "integer multiply/divide unit instructions completed"
	},
	{.pme_name="STORES_COMPLETED",
	 .pme_code = 0x00000f00,
	 .pme_counters = 0x2,
	 .pme_desc = "Stores completed (including FP)"
	},
	{.pme_name="MIPS16_INSTRUCTIONS_COMPLETED",
	 .pme_code = 0x00001000,
	 .pme_counters = 0x2,
	 .pme_desc = "MIPS16 instructions completed"
	},
	{.pme_name="BRANCHES_LAUNCHED",
	 .pme_code = 0x00000002,
	 .pme_counters = 0x1,
	 .pme_desc = "Branch instructions launched (whether completed or mispredicted)"
	},
	{.pme_name="SCACHE_ACCESSES",
	 .pme_code = 0x00001500,
	 .pme_counters = 0x2,
	 .pme_desc = "L2 cache accesses"
	},
	{.pme_name="JR_31_LAUNCHED",
	 .pme_code = 0x00000003,
	 .pme_counters = 0x1,
	 .pme_desc = "jr r31 (return) instructions launched (whether completed or mispredicted)"
	},
	{.pme_name="PREFETCH_COMPLETED",
	 .pme_code = 0x00000014,
	 .pme_counters = 0x1,
	 .pme_desc = "Prefetch instructions completed"
	},
	{.pme_name="EXCEPTIONS_TAKEN",
	 .pme_code = 0x00000017,
	 .pme_counters = 0x1,
	 .pme_desc = "Exceptions taken"
	},
	{.pme_name="JR_NON_31_LAUNCHED",
	 .pme_code = 0x00000004,
	 .pme_counters = 0x1,
	 .pme_desc = "jr (not r31) issues, which cost the same as a mispredict."
	},
	{.pme_name="DTLB_ACCESSES",
	 .pme_code = 0x00000006,
	 .pme_counters = 0x1,
	 .pme_desc = "Data micro-TLB accesses"
	},
	{.pme_name="JTLB_INSTRUCTION_ACCESSES",
	 .pme_code = 0x00000008,
	 .pme_counters = 0x1,
	 .pme_desc = "Joint TLB data (non-instruction) accesses"
	},
	{.pme_name="CACHE_FIXUPS",
	 .pme_code = 0x00000018,
	 .pme_counters = 0x1,
	 .pme_desc = "``cache fixup'' events (specific to the 24K family microarchitecture)."
	},
	{.pme_name="INSTRUCTION_CACHE_ACCESSES",
	 .pme_code = 0x00000009,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction cache accesses"
	},
	{.pme_name="DTLB_MISSES",
	 .pme_code = 0x00000600,
	 .pme_counters = 0x2,
	 .pme_desc = "Data micro-TLB misses"
	},
	{.pme_name="J_JAL_INSNS_COMPLETED",
	 .pme_code = 0x00000010,
	 .pme_counters = 0x1,
	 .pme_desc = "j/jal instructions completed"
	},
	{.pme_name="DCACHE_ACCESSES",
	 .pme_code = 0x0000000a,
	 .pme_counters = 0x1,
	 .pme_desc = "Data cache accesses"
	},
	{.pme_name="BRANCH_MISSPREDICTS",
	 .pme_code = 0x00000200,
	 .pme_counters = 0x2,
	 .pme_desc = "Branch mispredictions"
	},
	{.pme_name="SCACHE_WRITEBACKS",
	 .pme_code = 0x00000015,
	 .pme_counters = 0x1,
	 .pme_desc = "L2 cache writebacks"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Cycles"
	},
	{.pme_name="JTLB_INSN_MISSES",
	 .pme_code = 0x00000700,
	 .pme_counters = 0x2,
	 .pme_desc = "Joint TLB instruction misses"
	},
	{.pme_name="FPU_INSNS_NON_LOAD_STORE_COMPLETED",
	 .pme_code = 0x00000e00,
	 .pme_counters = 0x2,
	 .pme_desc = "FPU instructions completed (not including loads/stores)"
	},
	{.pme_name="NOPS_COMPLETED",
	 .pme_code = 0x00000011,
	 .pme_counters = 0x1,
	 .pme_desc = "no-ops completed, ie instructions writing $0"
	},
	{.pme_name="ITLB_MISSES",
	 .pme_code = 0x00000500,
	 .pme_counters = 0x2,
	 .pme_desc = "Instruction micro-TLB misses"
	},
};

static pme_gen_mips64_entry_t gen_mips64_25K_pe[] = {
	{.pme_name="INSNS_FETCHED_FROM_ICACHE",
	 .pme_code = 0x00001818,
	 .pme_counters = 0x3,
	 .pme_desc = "Total number of instructions fetched from the I-Cache"
	},
	{.pme_name="FP_EXCEPTIONS_TAKEN",
	 .pme_code = 0x00000b0b,
	 .pme_counters = 0x3,
	 .pme_desc = "Taken FPU exceptions"
	},
	{.pme_name="INSN_ISSUED",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "Dispatched/issued instructions"
	},
	{.pme_name="STORE_INSNS_ISSUED",
	 .pme_code = 0x00000505,
	 .pme_counters = 0x3,
	 .pme_desc = "Store instructions issued"
	},
	{.pme_name="L2_MISSES",
	 .pme_code = 0x00001e1e,
	 .pme_counters = 0x3,
	 .pme_desc = "L2 Cache miss"
	},
	{.pme_name="REPLAYS_LOAD_DEP_DISPATCH",
	 .pme_code = 0x00002323,
	 .pme_counters = 0x3,
	 .pme_desc = "replays due to load-dependent speculative dispatch"
	},
	{.pme_name="BRANCHES_JUMPS_ISSUED",
	 .pme_code = 0x00000606,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch/Jump instructions issued"
	},
	{.pme_name="REPLAYS_LSU_LOAD_DEP_FPU",
	 .pme_code = 0x00002121,
	 .pme_counters = 0x3,
	 .pme_desc = "LSU requested replays, load-dependent speculative dispatch, FPU exception prediction"
	},
	{.pme_name="INSNS_COMPLETE",
	 .pme_code = 0x00000808,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction that completed execution (with or without exception)"
	},
	{.pme_name="JTLB_MISSES_LOADS_STORES",
	 .pme_code = 0x00001313,
	 .pme_counters = 0x3,
	 .pme_desc = "Raw count of Joint-TLB misses for loads/stores"
	},
	{.pme_name="CACHEABLE_DCACHE_REQUEST",
	 .pme_code = 0x00001d1d,
	 .pme_counters = 0x3,
	 .pme_desc = "number of cacheable requests to D-Cache"
	},
	{.pme_name="DCACHE_WRITEBACKS",
	 .pme_code = 0x00001c1c,
	 .pme_counters = 0x3,
	 .pme_desc = "D-Cache number of write-backs"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00001a1a,
	 .pme_counters = 0x3,
	 .pme_desc = "I-Cache miss"
	},
	{.pme_name="ICACHE_PSEUDO_HITS",
	 .pme_code = 0x00002626,
	 .pme_counters = 0x3,
	 .pme_desc = "I-Cache pseudo-hits"
	},
	{.pme_name="FP_EXCEPTION_PREDICTED",
	 .pme_code = 0x00000c0c,
	 .pme_counters = 0x3,
	 .pme_desc = "Predicted FPU exceptions"
	},
	{.pme_name="LOAD_STORE_ISSUED",
	 .pme_code = 0x00002727,
	 .pme_counters = 0x3,
	 .pme_desc = "Load/store instructions issued"
	},
	{.pme_name="REPLAYS_WBB_FULL",
	 .pme_code = 0x00002424,
	 .pme_counters = 0x3,
	 .pme_desc = "replays due to WBB full"
	},
	{.pme_name="L2_WBACKS",
	 .pme_code = 0x00001f1f,
	 .pme_counters = 0x3,
	 .pme_desc = "L2 Cache number of write-backs"
	},
	{.pme_name="JR_COMPLETED",
	 .pme_code = 0x00001010,
	 .pme_counters = 0x3,
	 .pme_desc = "JR instruction that completed execution"
	},
	{.pme_name="JR_RPD_MISSPREDICTED",
	 .pme_code = 0x00000f0f,
	 .pme_counters = 0x3,
	 .pme_desc = "JR instructions that mispredicted using the Return Prediction Stack"
	},
	{.pme_name="JTLB_IFETCH_REFILL_EXCEPTIONS",
	 .pme_code = 0x00001515,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint-TLB refill exceptions due to instruction fetch"
	},
	{.pme_name="DUAL_ISSUED_PAIRS",
	 .pme_code = 0x00000707,
	 .pme_counters = 0x3,
	 .pme_desc = "Dual-issued pairs"
	},
	{.pme_name="FSB_FULL_REPLAYS",
	 .pme_code = 0x00002525,
	 .pme_counters = 0x3,
	 .pme_desc = "replays due to FSB full"
	},
	{.pme_name="JTLB_REFILL_EXCEPTIONS",
	 .pme_code = 0x00001717,
	 .pme_counters = 0x3,
	 .pme_desc = "total Joint-TLB Instruction exceptions (refill)"
	},
	{.pme_name="INT_INSNS_ISSUED",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Integer instructions issued"
	},
	{.pme_name="FP_INSNS_ISSUED",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "FPU instructions issued"
	},
	{.pme_name="BRANCHES_MISSPREDICTED",
	 .pme_code = 0x00000d0d,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches that mispredicted before completing execution"
	},
	{.pme_name="FETCH_GROUPS_IN_PIPE",
	 .pme_code = 0x00000909,
	 .pme_counters = 0x3,
	 .pme_desc = "Fetch groups entering CPU execution pipes"
	},
	{.pme_name="CACHEABLE_L2_REQS",
	 .pme_code = 0x00002020,
	 .pme_counters = 0x3,
	 .pme_desc = "Number of cacheable requests to L2"
	},
	{.pme_name="JTLB_DATA_ACCESS_REFILL_EXCEPTIONS",
	 .pme_code = 0x00001616,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint-TLB refill exceptions due to data access"
	},
	{.pme_name="UTLB_MISSES",
	 .pme_code = 0x00001111,
	 .pme_counters = 0x3,
	 .pme_desc = "U-TLB misses"
	},
	{.pme_name="LOAD_INSNS_ISSUED",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Load instructions issued"
	},
	{.pme_name="JTLB_MISSES_IFETCH",
	 .pme_code = 0x00001212,
	 .pme_counters = 0x3,
	 .pme_desc = "Raw count of Joint-TLB misses for instruction fetch"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "CPU cycles"
	},
	{.pme_name="LSU_REQ_REPLAYS",
	 .pme_code = 0x00002222,
	 .pme_counters = 0x3,
	 .pme_desc = "LSU requested replays"
	},
	{.pme_name="INSN_REQ_FROM_IFU_BIU",
	 .pme_code = 0x00001919,
	 .pme_counters = 0x3,
	 .pme_desc = "instruction requests from the IFU to the BIU"
	},
	{.pme_name="JTLB_EXCEPTIONS",
	 .pme_code = 0x00001414,
	 .pme_counters = 0x3,
	 .pme_desc = "Refill, Invalid and Modified TLB exceptions"
	},
	{.pme_name="BRANCHES_COMPLETED",
	 .pme_code = 0x00000e0e,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches that completed execution"
	},
	{.pme_name="INSN_FP_DATAPATH_COMPLETED",
	 .pme_code = 0x00000a0a,
	 .pme_counters = 0x3,
	 .pme_desc = "Instructions completed in FPU datapath (computational instructions only)"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x00001b1b,
	 .pme_counters = 0x3,
	 .pme_desc = "D-Cache miss"
	},
};

static pme_gen_mips64_entry_t gen_mips64_34K_pe[] = {
	{.pme_name="YIELD_INSNS",
	 .pme_code = 0x00220022,
	 .pme_counters = 0x5,
	 .pme_desc = "yield instructions."
	},
	{.pme_name="BRANCH_MISPREDICT_STALLS",
	 .pme_code = 0x002e002e,
	 .pme_counters = 0x5,
	 .pme_desc = "Branch mispredict stalls"
	},
	{.pme_name="SC_FAILED_INSNS",
	 .pme_code = 0x00130013,
	 .pme_counters = 0x5,
	 .pme_desc = "sc instructions completed, but store failed (because the link bit had been cleared)."
	},
	{.pme_name="ITC_LOAD_STORE_STALLS",
	 .pme_code = 0x00280028,
	 .pme_counters = 0x5,
	 .pme_desc = "ITC load/store stalls"
	},
	{.pme_name="ITC_LOADS",
	 .pme_code = 0x00200020,
	 .pme_counters = 0x5,
	 .pme_desc = "ITC Loads"
	},
	{.pme_name="LOADS_COMPLETED",
	 .pme_code = 0x000f000f,
	 .pme_counters = 0x5,
	 .pme_desc = "Loads completed (including FP)"
	},
	{.pme_name="BRANCH_INSNS_LAUNCHED",
	 .pme_code = 0x00020002,
	 .pme_counters = 0x5,
	 .pme_desc = "Branch instructions launched (whether completed or mispredicted)"
	},
	{.pme_name="DATA_SIDE_SCRATCHPAD_ACCESS_STALLS",
	 .pme_code = 0x002b002b,
	 .pme_counters = 0x5,
	 .pme_desc = "Data-side scratchpad access stalls"
	},
	{.pme_name="FB_ENTRY_ALLOCATED",
	 .pme_code = 0x00300030,
	 .pme_counters = 0x5,
	 .pme_desc = "FB entry allocated"
	},
	{.pme_name="CP2_STALLS",
	 .pme_code = 0x002a002a,
	 .pme_counters = 0x5,
	 .pme_desc = "CP2 stalls"
	},
	{.pme_name="FSB_25_50_FULL",
	 .pme_code = 0x00320032,
	 .pme_counters = 0x5,
	 .pme_desc = "FSB 25-50% full"
	},
	{.pme_name="CACHE_FIXUP_EVENTS",
	 .pme_code = 0x00180018,
	 .pme_counters = 0x5,
	 .pme_desc = "cache fixup events (specific to the 34K family microarchitecture)"
	},
	{.pme_name="IFU_FB_FULL_REFETCHES",
	 .pme_code = 0x00300030,
	 .pme_counters = 0x5,
	 .pme_desc = "IFU FB full re-fetches"
	},
	{.pme_name="L1_DCACHE_MISS_STALLS",
	 .pme_code = 0x00250025,
	 .pme_counters = 0x5,
	 .pme_desc = "L1 D-cache miss stalls"
	},
	{.pme_name="INT_MUL_DIV_UNIT_INSNS_COMPLETED",
	 .pme_code = 0x00110011,
	 .pme_counters = 0x5,
	 .pme_desc = "integer multiply/divide unit instructions completed"
	},
	{.pme_name="JTLB_INSN_ACCESSES",
	 .pme_code = 0x00070007,
	 .pme_counters = 0x5,
	 .pme_desc = "Joint TLB instruction accesses"
	},
	{.pme_name="ALU_STALLS",
	 .pme_code = 0x00190019,
	 .pme_counters = 0x5,
	 .pme_desc = "ALU stalls"
	},
	{.pme_name="FPU_STALLS",
	 .pme_code = 0x00290029,
	 .pme_counters = 0x5,
	 .pme_desc = "FPU stalls"
	},
	{.pme_name="JTLB_DATA_ACCESSES",
	 .pme_code = 0x00080008,
	 .pme_counters = 0x5,
	 .pme_desc = "Joint TLB data (non-instruction) accesses"
	},
	{.pme_name="INTEGER_INSNS_COMPLETED",
	 .pme_code = 0x000e000e,
	 .pme_counters = 0x5,
	 .pme_desc = "Integer instructions completed"
	},
	{.pme_name="MFC2_MTC2_INSNS",
	 .pme_code = 0x00230023,
	 .pme_counters = 0x5,
	 .pme_desc = "CP2 move to/from instructions."
	},
	{.pme_name="STORES_COMPLETED",
	 .pme_code = 0x000f000f,
	 .pme_counters = 0x5,
	 .pme_desc = "Stores completed (including FP)"
	},
	{.pme_name="JR_NON_31_INSN_EXECED",
	 .pme_code = 0x00040004,
	 .pme_counters = 0x5,
	 .pme_desc = "jr $xx (not $31), which cost the same as a mispredict."
	},
	{.pme_name="EXCEPTIONS_TAKEN",
	 .pme_code = 0x00170017,
	 .pme_counters = 0x5,
	 .pme_desc = "Exceptions taken"
	},
	{.pme_name="L2_MISS_PENDING_CYCLES",
	 .pme_code = 0x00270027,
	 .pme_counters = 0x5,
	 .pme_desc = "Cycles where L2 miss is pending"
	},
	{.pme_name="LDQ_FULL_PIPE_STALLS",
	 .pme_code = 0x00350035,
	 .pme_counters = 0x5,
	 .pme_desc = "LDQ full pipeline stalls"
	},
	{.pme_name="DTLB_ACCESSES",
	 .pme_code = 0x00060006,
	 .pme_counters = 0x5,
	 .pme_desc = "Data micro-TLB accesses"
	},
	{.pme_name="SUPERFLUOUS_PREFETCHES",
	 .pme_code = 0x00140014,
	 .pme_counters = 0x5,
	 .pme_desc = "``superfluous'' prefetch instructions (data was already in cache)."
	},
	{.pme_name="LDQ_LESS_25_FULL",
	 .pme_code = 0x00340034,
	 .pme_counters = 0x5,
	 .pme_desc = "LDQ < 25% full"
	},
	{.pme_name="FORK_INSTRUCTIONS",
	 .pme_code = 0x00220022,
	 .pme_counters = 0x5,
	 .pme_desc = "fork instructions"
	},
	{.pme_name="UNCACHED_LOAD_STALLS",
	 .pme_code = 0x00280028,
	 .pme_counters = 0x5,
	 .pme_desc = "Uncached load stalls"
	},
	{.pme_name="FSB_FULL_PIPE_STALLS",
	 .pme_code = 0x00330033,
	 .pme_counters = 0x5,
	 .pme_desc = "FSB full pipeline stalls"
	},
	{.pme_name="MDU_STALLS",
	 .pme_code = 0x00290029,
	 .pme_counters = 0x5,
	 .pme_desc = "MDU stalls"
	},
	{.pme_name="FSB_LESS_25_FULL",
	 .pme_code = 0x00320032,
	 .pme_counters = 0x5,
	 .pme_desc = "FSB < 25% full"
	},
	{.pme_name="UNCACHED_LOADS",
	 .pme_code = 0x00210021,
	 .pme_counters = 0x5,
	 .pme_desc = "Uncached Loads"
	},
	{.pme_name="NO_OPS_COMPLETED",
	 .pme_code = 0x00110011,
	 .pme_counters = 0x5,
	 .pme_desc = "no-ops completed, ie instructions writing $0"
	},
	{.pme_name="DATA_SIDE_SCRATCHPAD_RAM_LOGIC",
	 .pme_code = 0x001d001d,
	 .pme_counters = 0x5,
	 .pme_desc = "Data-side scratchpad RAM logic"
	},
	{.pme_name="CYCLES_INSN_NOT_IN_SKID_BUFFER",
	 .pme_code = 0x00180018,
	 .pme_counters = 0x5,
	 .pme_desc = "Cycles lost when an unblocked thread's instruction isn't in the skid buffer, and must be re-fetched from I-cache."
	},
	{.pme_name="ITC_LOGIC",
	 .pme_code = 0x001f001f,
	 .pme_counters = 0x5,
	 .pme_desc = "ITC logic"
	},
	{.pme_name="L2_IMISS_STALLS",
	 .pme_code = 0x00260026,
	 .pme_counters = 0x5,
	 .pme_desc = "L2 I-miss stalls"
	},
	{.pme_name="DSP_RESULT_SATURATED",
	 .pme_code = 0x00240024,
	 .pme_counters = 0x5,
	 .pme_desc = "DSP result saturated"
	},
	{.pme_name="INSTRUCTIONS",
	 .pme_code = 0x01010101,
	 .pme_counters = 0xf,
	 .pme_desc = "Instructions completed"
	},
	{.pme_name="ITLB_ACCESSES",
	 .pme_code = 0x00050005,
	 .pme_counters = 0x5,
	 .pme_desc = "Instruction micro-TLB accesses"
	},
	{.pme_name="CP2_REG_TO_REG_INSNS",
	 .pme_code = 0x00230023,
	 .pme_counters = 0x5,
	 .pme_desc = "CP2 register-to-register instructions"
	},
	{.pme_name="SC_INSNS_COMPLETED",
	 .pme_code = 0x00130013,
	 .pme_counters = 0x5,
	 .pme_desc = "sc instructions completed"
	},
	{.pme_name="COREEXTEND_STALLS",
	 .pme_code = 0x002a002a,
	 .pme_counters = 0x5,
	 .pme_desc = "CorExtend stalls"
	},
	{.pme_name="LOAD_USE_STALLS",
	 .pme_code = 0x002d002d,
	 .pme_counters = 0x5,
	 .pme_desc = "Load to Use stalls"
	},
	{.pme_name="JR_31_INSN_EXECED",
	 .pme_code = 0x00030003,
	 .pme_counters = 0x5,
	 .pme_desc = "jr $31 (return) instructions executed."
	},
	{.pme_name="JR_31_MISPREDICTS",
	 .pme_code = 0x00030003,
	 .pme_counters = 0x5,
	 .pme_desc = "jr $31 mispredictions."
	},
	{.pme_name="REPLAY_CYCLES",
	 .pme_code = 0x00120012,
	 .pme_counters = 0x5,
	 .pme_desc = "Cycles lost due to ``replays'' - when a thread blocks, its instructions in the pipeline are discarded to allow other threads to advance."
	},
	{.pme_name="L2_MISSES",
	 .pme_code = 0x16161616,
	 .pme_counters = 0xf,
	 .pme_desc = "L2 cache misses"
	},
	{.pme_name="JTLB_DATA_MISSES",
	 .pme_code = 0x00080008,
	 .pme_counters = 0x5,
	 .pme_desc = "Joint TLB data (non-instruction) misses"
	},
	{.pme_name="SYSTEM_INTERFACE",
	 .pme_code = 0x001e001e,
	 .pme_counters = 0x5,
	 .pme_desc = "System interface"
	},
	{.pme_name="BRANCH_MISPREDICTS",
	 .pme_code = 0x00020002,
	 .pme_counters = 0x5,
	 .pme_desc = "Branch mispredictions"
	},
	{.pme_name="ITC_STORES",
	 .pme_code = 0x00200020,
	 .pme_counters = 0x5,
	 .pme_desc = "ITC Stores"
	},
	{.pme_name="LDQ_OVER_50_FULL",
	 .pme_code = 0x00350035,
	 .pme_counters = 0x5,
	 .pme_desc = "LDQ > 50% full"
	},
	{.pme_name="FSB_OVER_50_FULL",
	 .pme_code = 0x00330033,
	 .pme_counters = 0x5,
	 .pme_desc = "FSB > 50% full"
	},
	{.pme_name="STALLS_NO_ROOM_PENDING_WRITE",
	 .pme_code = 0x002c002c,
	 .pme_counters = 0x5,
	 .pme_desc = "Stalls when no more room to store pending write."
	},
	{.pme_name="JR_31_NOT_PREDICTED",
	 .pme_code = 0x00040004,
	 .pme_counters = 0x5,
	 .pme_desc = "jr $31 not predicted (stack mismatch)."
	},
	{.pme_name="EXTERNAL_YIELD_MANAGER_LOGIC",
	 .pme_code = 0x001f001f,
	 .pme_counters = 0x5,
	 .pme_desc = "External Yield Manager logic"
	},
	{.pme_name="DCACHE_WRITEBACKS",
	 .pme_code = 0x000a000a,
	 .pme_counters = 0x5,
	 .pme_desc = "Data cache writebacks"
	},
	{.pme_name="RELAX_BUBBLES",
	 .pme_code = 0x002f002f,
	 .pme_counters = 0x5,
	 .pme_desc = "``Relax bubbles'' - when thread scheduler chooses to schedule nothing to reduce power consumption."
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00090009,
	 .pme_counters = 0x5,
	 .pme_desc = "Instruction cache misses"
	},
	{.pme_name="MIPS16_INSNS_COMPLETED",
	 .pme_code = 0x00100010,
	 .pme_counters = 0x5,
	 .pme_desc = "MIPS16 instructions completed"
	},
	{.pme_name="OTHER_INTERLOCK_STALLS",
	 .pme_code = 0x002e002e,
	 .pme_counters = 0x5,
	 .pme_desc = "Other interlock stalls"
	},
	{.pme_name="L2_CACHE_WRITEBACKS",
	 .pme_code = 0x00150015,
	 .pme_counters = 0x5,
	 .pme_desc = "L2 cache writebacks"
	},
	{.pme_name="WBB_LESS_25_FULL",
	 .pme_code = 0x00360036,
	 .pme_counters = 0x5,
	 .pme_desc = "WBB < 25% full"
	},
	{.pme_name="L2_DCACHE_MISS_STALLS",
	 .pme_code = 0x00260026,
	 .pme_counters = 0x5,
	 .pme_desc = "L2 D-miss stalls"
	},
	{.pme_name="CACHE_INSTRUCTION_STALLS",
	 .pme_code = 0x002c002c,
	 .pme_counters = 0x5,
	 .pme_desc = "Stalls due to cache instructions"
	},
	{.pme_name="L1_DCACHE_MISS_PENDING_CYCLES",
	 .pme_code = 0x00270027,
	 .pme_counters = 0x5,
	 .pme_desc = "Cycles where L1 D-cache miss pending"
	},
	{.pme_name="ALU_TO_AGEN_STALLS",
	 .pme_code = 0x002d002d,
	 .pme_counters = 0x5,
	 .pme_desc = "ALU to AGEN stalls"
	},
	{.pme_name="L2_ACCESSES",
	 .pme_code = 0x00150015,
	 .pme_counters = 0x5,
	 .pme_desc = "L2 cache accesses"
	},
	{.pme_name="J_JAL_INSN_COMPLETED",
	 .pme_code = 0x00100010,
	 .pme_counters = 0x5,
	 .pme_desc = "j/jal instructions completed"
	},
	{.pme_name="ALL_STALLS",
	 .pme_code = 0x00120012,
	 .pme_counters = 0x5,
	 .pme_desc = "All stalls (no action in RF pipe stage)"
	},
	{.pme_name="DSP_INSTRUCTIONS",
	 .pme_code = 0x00240024,
	 .pme_counters = 0x5,
	 .pme_desc = "DSP instructions"
	},
	{.pme_name="UNCACHED_STORES",
	 .pme_code = 0x00210021,
	 .pme_counters = 0x5,
	 .pme_desc = "Uncached Stores"
	},
	{.pme_name="WBB_FULL_PIPE_STALLS",
	 .pme_code = 0x00370037,
	 .pme_counters = 0x5,
	 .pme_desc = "WBB full pipeline stalls"
	},
	{.pme_name="INSN_CACHE_ACCESSES",
	 .pme_code = 0x00090009,
	 .pme_counters = 0x5,
	 .pme_desc = "Instruction cache accesses"
	},
	{.pme_name="EXT_POLICY_MANAGER",
	 .pme_code = 0x001c001c,
	 .pme_counters = 0x5,
	 .pme_desc = "External policy manager"
	},
	{.pme_name="WBB_OVER_50_FULL",
	 .pme_code = 0x00370037,
	 .pme_counters = 0x5,
	 .pme_desc = "WBB > 50% full"
	},
	{.pme_name="DTLB_MISSES",
	 .pme_code = 0x00060006,
	 .pme_counters = 0x5,
	 .pme_desc = "Data micro-TLB misses"
	},
	{.pme_name="DCACHE_ACCESSES",
	 .pme_code = 0x000a000a,
	 .pme_counters = 0x5,
	 .pme_desc = "Data cache accesses"
	},
	{.pme_name="COREEXTEND_LOGIC",
	 .pme_code = 0x001e001e,
	 .pme_counters = 0x5,
	 .pme_desc = "CorExtend logic"
	},
	{.pme_name="LDQ_25_50_FULL",
	 .pme_code = 0x00340034,
	 .pme_counters = 0x5,
	 .pme_desc = "LDQ 25-50% full"
	},
	{.pme_name="PREFETCH_INSNS_COMPLETED",
	 .pme_code = 0x00140014,
	 .pme_counters = 0x5,
	 .pme_desc = "Prefetch instructions completed"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0xf,
	 .pme_desc = "Cycles"
	},
	{.pme_name="L1_ICACHE_MISS_STALLS",
	 .pme_code = 0x00250025,
	 .pme_counters = 0x5,
	 .pme_desc = "L1 I-cache miss stalls"
	},
	{.pme_name="JTLB_INSN_MISSES",
	 .pme_code = 0x00070007,
	 .pme_counters = 0x5,
	 .pme_desc = "Joint TLB instruction misses"
	},
	{.pme_name="COP2",
	 .pme_code = 0x001c001c,
	 .pme_counters = 0x5,
	 .pme_desc = "Co-Processor 2"
	},
	{.pme_name="FPU_INSNS_COMPLETED",
	 .pme_code = 0x000e000e,
	 .pme_counters = 0x5,
	 .pme_desc = "FPU instructions completed (not including loads/stores)"
	},
	{.pme_name="ITLB_MISSES",
	 .pme_code = 0x00050005,
	 .pme_counters = 0x5,
	 .pme_desc = "Instruction micro-TLB misses"
	},
	{.pme_name="IFU_STALLS",
	 .pme_code = 0x00190019,
	 .pme_counters = 0x5,
	 .pme_desc = "IFU stalls (when no instruction offered) ALU stalls"
	},
	{.pme_name="WBB_25_50_FULL",
	 .pme_code = 0x00360036,
	 .pme_counters = 0x5,
	 .pme_desc = "WBB 25-50% full"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x0b0b0b0b,
	 .pme_counters = 0xf,
	 .pme_desc = "Data cache misses"
	},
};

static pme_gen_mips64_entry_t gen_mips64_5K_pe[] = {
	{.pme_name="DCACHE_LINE_EVICTED",
	 .pme_code = 0x00000600,
	 .pme_counters = 0x2,
	 .pme_desc = "Data cache line evicted"
	},
	{.pme_name="LOADS_EXECED",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "Load/pref(x)/sync/cache-ops executed"
	},
	{.pme_name="INSN_SCHEDULED",
	 .pme_code = 0x0000000a,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction scheduled"
	},
	{.pme_name="DUAL_ISSUED_INSNS",
	 .pme_code = 0x0000000e,
	 .pme_counters = 0x1,
	 .pme_desc = "Dual issued instructions executed"
	},
	{.pme_name="BRANCHES_MISSPREDICTED",
	 .pme_code = 0x00000800,
	 .pme_counters = 0x2,
	 .pme_desc = "Branch mispredicted"
	},
	{.pme_name="CONFLICT_STALL_M_STAGE",
	 .pme_code = 0x00000a00,
	 .pme_counters = 0x2,
	 .pme_desc = "Instruction stall in M stage due to scheduling conflicts"
	},
	{.pme_name="STORES_EXECED",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Stores (including conditional stores) executed"
	},
	{.pme_name="DCACHE_MISS",
	 .pme_code = 0x00000900,
	 .pme_counters = 0x2,
	 .pme_desc = "Data cache miss"
	},
	{.pme_name="INSN_FETCHED",
	 .pme_code = 0x00000001,
	 .pme_counters = 0x1,
	 .pme_desc = "Instructions fetched"
	},
	{.pme_name="TLB_MISS_EXCEPTIONS",
	 .pme_code = 0x00000700,
	 .pme_counters = 0x2,
	 .pme_desc = "TLB miss exceptions"
	},
	{.pme_name="COP2_INSNS_EXECED",
	 .pme_code = 0x00000f00,
	 .pme_counters = 0x2,
	 .pme_desc = "COP2 instructions executed"
	},
	{.pme_name="FAILED_COND_STORES",
	 .pme_code = 0x00000005,
	 .pme_counters = 0x1,
	 .pme_desc = "Failed conditional stores"
	},
	{.pme_name="INSNS_EXECED",
	 .pme_code = 0x0000010f,
	 .pme_counters = 0x3,
	 .pme_desc = "Instructions executed"
	},
	{.pme_name="ICACHE_MISS",
	 .pme_code = 0x00000009,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction cache miss"
	},
	{.pme_name="COND_STORES_EXECED",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Conditional stores executed"
	},
	{.pme_name="FP_INSNS_EXECED",
	 .pme_code = 0x00000500,
	 .pme_counters = 0x2,
	 .pme_desc = "Floating-point instructions executed"
	},
	{.pme_name="DTLB_MISSES",
	 .pme_code = 0x00000008,
	 .pme_counters = 0x1,
	 .pme_desc = "DTLB miss"
	},
	{.pme_name="BRANCHES_EXECED",
	 .pme_code = 0x00000006,
	 .pme_counters = 0x1,
	 .pme_desc = "Branches executed"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Cycles"
	},
	{.pme_name="ITLB_MISSES",
	 .pme_code = 0x00000007,
	 .pme_counters = 0x1,
	 .pme_desc = "ITLB miss"
	},
};

static pme_gen_mips64_entry_t gen_mips64_r10000_pe[] = {
	{.pme_name="BRANCHES_RESOLVED",
	 .pme_code = 0x00000006,
	 .pme_counters = 0x1,
	 .pme_desc = "Branches resolved"
	},
	{.pme_name="TLB_REFILL_EXCEPTIONS",
	 .pme_code = 0x00000700,
	 .pme_counters = 0x2,
	 .pme_desc = "TLB refill exceptions"
	},
	{.pme_name="EXTERNAL_INTERVENTION_RQ",
	 .pme_code = 0x0000000c,
	 .pme_counters = 0x1,
	 .pme_desc = "External intervention requests"
	},
	{.pme_name="STORES_GRADUATED",
	 .pme_code = 0x00000300,
	 .pme_counters = 0x2,
	 .pme_desc = "Stores graduated"
	},
	{.pme_name="SCACHE_WAY_MISPREDICTED_INSN",
	 .pme_code = 0x0000000b,
	 .pme_counters = 0x1,
	 .pme_desc = "Secondary cache way mispredicted (instruction)"
	},
	{.pme_name="INSTRUCTION_CACHE_MISSES",
	 .pme_code = 0x00000009,
	 .pme_counters = 0x1,
	 .pme_desc = "Instruction cache misses"
	},
	{.pme_name="SCACHE_MISSES_DATA",
	 .pme_code = 0x00000a00,
	 .pme_counters = 0x2,
	 .pme_desc = "Secondary cache misses (data)"
	},
	{.pme_name="QUADWORDS_WB_FROM_PRIMARY_DCACHE",
	 .pme_code = 0x00000600,
	 .pme_counters = 0x2,
	 .pme_desc = "Quadwords written back from primary data cache"
	},
	{.pme_name="EXTERNAL_INVALIDATE_RQ_HITS_SCACHE",
	 .pme_code = 0x00000d00,
	 .pme_counters = 0x2,
	 .pme_desc = "External invalidate request is determined to have hit in secondary cache"
	},
	{.pme_name="LOAD_PREFETC_SYNC_CACHEOP_ISSUED",
	 .pme_code = 0x00000002,
	 .pme_counters = 0x1,
	 .pme_desc = "Load / prefetch / sync / CacheOp issued"
	},
	{.pme_name="STORES_OR_STORE_PREF_TO_SHD_SCACHE_BLOCKS",
	 .pme_code = 0x00000f00,
	 .pme_counters = 0x2,
	 .pme_desc = "Stores or prefetches with store hint to Shared secondary cache blocks"
	},
	{.pme_name="STORE_COND_ISSUED",
	 .pme_code = 0x00000004,
	 .pme_counters = 0x1,
	 .pme_desc = "Store conditional issued"
	},
	{.pme_name="BRANCHES_MISPREDICTED",
	 .pme_code = 0x00000800,
	 .pme_counters = 0x2,
	 .pme_desc = "Branches mispredicted"
	},
	{.pme_name="EXTERNAL_INVALIDATE_RQ",
	 .pme_code = 0x0000000d,
	 .pme_counters = 0x1,
	 .pme_desc = "External invalidate requests"
	},
	{.pme_name="LOAD_PREFETC_SYNC_CACHEOP_GRADUATED",
	 .pme_code = 0x00000200,
	 .pme_counters = 0x2,
	 .pme_desc = "Load / prefetch / sync / CacheOp graduated"
	},
	{.pme_name="INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000001,
	 .pme_counters = 0x1,
	 .pme_desc = "Instructions issued"
	},
	{.pme_name="INSTRUCTION_GRADUATED",
	 .pme_code = 0x0000000f,
	 .pme_counters = 0x1,
	 .pme_desc = "Instructions graduated"
	},
	{.pme_name="EXTERNAL_INTERVENTION_RQ_HITS_SCACHE",
	 .pme_code = 0x00000c00,
	 .pme_counters = 0x2,
	 .pme_desc = "External intervention request is determined to have hit in secondary cache"
	},
	{.pme_name="SCACHE_MISSES_INSTRUCTION",
	 .pme_code = 0x0000000a,
	 .pme_counters = 0x1,
	 .pme_desc = "Secondary cache misses (instruction)"
	},
	{.pme_name="SCACHE_LOAD_STORE_CACHEOP_OPERATIONS",
	 .pme_code = 0x00000900,
	 .pme_counters = 0x2,
	 .pme_desc = "Secondary cache load / store and cache-ops operations"
	},
	{.pme_name="STORES_OR_STORE_PREF_TO_CLEANEXCLUSIVE_SCACHE_BLOCKS",
	 .pme_code = 0x00000e00,
	 .pme_counters = 0x2,
	 .pme_desc = "Stores or prefetches with store hint to CleanExclusive secondary cache blocks"
	},
	{.pme_name="INSTRUCTIONS_GRADUATED",
	 .pme_code = 0x00000100,
	 .pme_counters = 0x2,
	 .pme_desc = "Instructions graduated"
	},
	{.pme_name="FP_INSTRUCTON_GRADUATED",
	 .pme_code = 0x00000500,
	 .pme_counters = 0x2,
	 .pme_desc = "Floating-point instructions graduated"
	},
	{.pme_name="STORES_ISSUED",
	 .pme_code = 0x00000003,
	 .pme_counters = 0x1,
	 .pme_desc = "Stores issued"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Cycles"
	},
	{.pme_name="CORRECTABLE_ECC_ERRORS_SCACHE",
	 .pme_code = 0x00000008,
	 .pme_counters = 0x1,
	 .pme_desc = "Correctable ECC errors on secondary cache data"
	},
	{.pme_name="QUADWORDS_WB_FROM_SCACHE",
	 .pme_code = 0x00000007,
	 .pme_counters = 0x1,
	 .pme_desc = "Quadwords written back from secondary cache"
	},
	{.pme_name="STORE_COND_GRADUATED",
	 .pme_code = 0x00000400,
	 .pme_counters = 0x2,
	 .pme_desc = "Store conditional graduated"
	},
	{.pme_name="FUNCTIONAL_UNIT_COMPLETION_CYCLES",
	 .pme_code = 0x0000000e,
	 .pme_counters = 0x1,
	 .pme_desc = "Functional unit completion cycles"
	},
	{.pme_name="FAILED_STORE_CONDITIONAL",
	 .pme_code = 0x00000005,
	 .pme_counters = 0x1,
	 .pme_desc = "Failed store conditional"
	},
	{.pme_name="SCACHE_WAY_MISPREDICTED_DATA",
	 .pme_code = 0x00000b00,
	 .pme_counters = 0x2,
	 .pme_desc = "Secondary cache way mispredicted (data)"
	},
};

static pme_gen_mips64_entry_t gen_mips64_r12000_pe[] = {
	{.pme_name="INTERVENTION_REQUESTS",
	 .pme_code = 0x0c0c0c0c,
	 .pme_counters = 0xf,
	 .pme_desc = "External intervention requests"
	},
	{.pme_name="QUADWORDS",
	 .pme_code = 0x16161616,
	 .pme_counters = 0xf,
	 .pme_desc = "Quadwords written back from primary data cache"
	},
	{.pme_name="MISPREDICTED_BRANCHES",
	 .pme_code = 0x18181818,
	 .pme_counters = 0xf,
	 .pme_desc = "Mispredicted branches"
	},
	{.pme_name="DECODED_STORES",
	 .pme_code = 0x03030303,
	 .pme_counters = 0xf,
	 .pme_desc = "Decoded stores"
	},
	{.pme_name="TLB_MISSES",
	 .pme_code = 0x17171717,
	 .pme_counters = 0xf,
	 .pme_desc = "TLB misses"
	},
	{.pme_name="GRADUATED_FP_INSTRUCTIONS",
	 .pme_code = 0x15151515,
	 .pme_counters = 0xf,
	 .pme_desc = "Graduated floating point instructions"
	},
	{.pme_name="EXTERNAL_REQUESTS",
	 .pme_code = 0x0d0d0d0d,
	 .pme_counters = 0xf,
	 .pme_desc = "External invalidate requests"
	},
	{.pme_name="GRADUATED_STORES",
	 .pme_code = 0x13131313,
	 .pme_counters = 0xf,
	 .pme_desc = "Graduated stores"
	},
	{.pme_name="PREFETCH_MISSES_IN_DCACHE",
	 .pme_code = 0x11111111,
	 .pme_counters = 0xf,
	 .pme_desc = "Primary data cache misses by prefetch instructions"
	},
	{.pme_name="STORE_PREFETCH_EXCLUSIVE_SHARED_SC_BLOCK",
	 .pme_code = 0x1f1f1f1f,
	 .pme_counters = 0xf,
	 .pme_desc = "Store/prefetch exclusive to shared block in secondary"
	},
	{.pme_name="DECODED_LOADS",
	 .pme_code = 0x02020202,
	 .pme_counters = 0xf,
	 .pme_desc = "Decoded loads"
	},
	{.pme_name="GRADUATED_STORE_CONDITIONALS",
	 .pme_code = 0x14141414,
	 .pme_counters = 0xf,
	 .pme_desc = "Graduated store conditionals"
	},
	{.pme_name="INSTRUCTION_SECONDARY_CACHE_MISSES",
	 .pme_code = 0x0a0a0a0a,
	 .pme_counters = 0xf,
	 .pme_desc = "Secondary cache misses (instruction)"
	},
	{.pme_name="STATE_OF_EXTERNAL_INVALIDATION_HIT",
	 .pme_code = 0x1d1d1d1d,
	 .pme_counters = 0xf,
	 .pme_desc = "State of external invalidation hits in secondary cache"
	},
	{.pme_name="SECONDARY_CACHE_WAY_MISSPREDICTED",
	 .pme_code = 0x0b0b0b0b,
	 .pme_counters = 0xf,
	 .pme_desc = "Secondary cache way mispredicted (instruction)"
	},
	{.pme_name="DECODED_INSTRUCTIONS",
	 .pme_code = 0x01010101,
	 .pme_counters = 0xf,
	 .pme_desc = "Decoded instructions"
	},
	{.pme_name="SCACHE_MISSES",
	 .pme_code = 0x1a1a1a1a,
	 .pme_counters = 0xf,
	 .pme_desc = "Secondary cache misses (data)"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x09090909,
	 .pme_counters = 0xf,
	 .pme_desc = "Instruction cache misses"
	},
	{.pme_name="SCACHE_WAY_MISPREDICTION",
	 .pme_code = 0x1b1b1b1b,
	 .pme_counters = 0xf,
	 .pme_desc = "Misprediction from scache way prediction table (data)"
	},
	{.pme_name="STATE_OF_SCACHE_INTERVENTION_HIT",
	 .pme_code = 0x1c1c1c1c,
	 .pme_counters = 0xf,
	 .pme_desc = "State of external intervention hit in secondary cache"
	},
	{.pme_name="GRADUATED_LOADS",
	 .pme_code = 0x12121212,
	 .pme_counters = 0xf,
	 .pme_desc = "Graduated loads"
	},
	{.pme_name="PREFETCH_INSTRUCTIONS_EXECUTED",
	 .pme_code = 0x10101010,
	 .pme_counters = 0xf,
	 .pme_desc = "Executed prefetch instructions"
	},
	{.pme_name="MISS_TABLE_OCCUPANCY",
	 .pme_code = 0x04040404,
	 .pme_counters = 0xf,
	 .pme_desc = "Miss Handling Table Occupancy"
	},
	{.pme_name="INSTRUCTIONS_GRADUATED",
	 .pme_code = 0x0f0f0f0f,
	 .pme_counters = 0xf,
	 .pme_desc = "Instructions graduated"
	},
	{.pme_name="QUADWORDS_WRITEBACK_FROM_SC",
	 .pme_code = 0x07070707,
	 .pme_counters = 0xf,
	 .pme_desc = "Quadwords written back from secondary cache"
	},
	{.pme_name="CORRECTABLE_ECC_ERRORS",
	 .pme_code = 0x08080808,
	 .pme_counters = 0xf,
	 .pme_desc = "Correctable ECC errors on secondary cache data"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0xf,
	 .pme_desc = "Cycles"
	},
	{.pme_name="RESOLVED_BRANCH_CONDITIONAL",
	 .pme_code = 0x06060606,
	 .pme_counters = 0xf,
	 .pme_desc = "Resolved conditional branches"
	},
	{.pme_name="STORE_PREFETCH_EXCLUSIVE_TO_CLEAN_SC_BLOCK",
	 .pme_code = 0x1e1e1e1e,
	 .pme_counters = 0xf,
	 .pme_desc = "Store/prefetch exclusive to clean block in secondary cache"
	},
	{.pme_name="FAILED_STORE_CONDITIONAL",
	 .pme_code = 0x05050505,
	 .pme_counters = 0xf,
	 .pme_desc = "Failed store conditional"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x19191919,
	 .pme_counters = 0xf,
	 .pme_desc = "Primary data cache misses"
	},
};

static pme_gen_mips64_entry_t gen_mips64_rm7000_pe[] = {
	{.pme_name="SLIP_CYCLES_PENDING_NON_BLKING_LOAD",
	 .pme_code = 0x00001a1a,
	 .pme_counters = 0x3,
	 .pme_desc = "Slip cycles due to pending non-blocking loads"
	},
	{.pme_name="STORE_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000505,
	 .pme_counters = 0x3,
	 .pme_desc = "Store instructions issued"
	},
	{.pme_name="BRANCH_PREFETCHES",
	 .pme_code = 0x00000707,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch prefetches"
	},
	{.pme_name="PCACHE_WRITEBACKS",
	 .pme_code = 0x00001414,
	 .pme_counters = 0x3,
	 .pme_desc = "Primary cache writebacks"
	},
	{.pme_name="STALL_CYCLES_PENDING_NON_BLKING_LOAD",
	 .pme_code = 0x00001f1f,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles due to pending non-blocking loads - stall start of exception"
	},
	{.pme_name="STALL_CYCLES",
	 .pme_code = 0x00000909,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles"
	},
	{.pme_name="CACHE_MISSES",
	 .pme_code = 0x00001616,
	 .pme_counters = 0x3,
	 .pme_desc = "Cache misses"
	},
	{.pme_name="DUAL_ISSUED_PAIRS",
	 .pme_code = 0x00000606,
	 .pme_counters = 0x3,
	 .pme_desc = "Dual issued pairs"
	},
	{.pme_name="SLIP_CYCLES_DUE_MULTIPLIER_BUSY",
	 .pme_code = 0x00001818,
	 .pme_counters = 0x3,
	 .pme_desc = "Slip Cycles due to multiplier busy"
	},
	{.pme_name="INTEGER_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Integer instructions issued"
	},
	{.pme_name="SCACHE_WRITEBACKS",
	 .pme_code = 0x00001313,
	 .pme_counters = 0x3,
	 .pme_desc = "Secondary cache writebacks"
	},
	{.pme_name="DCACHE_MISS_STALL_CYCLES",
	 .pme_code = 0x00001515,
	 .pme_counters = 0x3,
	 .pme_desc = "Dcache miss stall cycles (cycles where both cache miss tokens taken and a third try is requested)"
	},
	{.pme_name="MULTIPLIER_STALL_CYCLES",
	 .pme_code = 0x00001e1e,
	 .pme_counters = 0x3,
	 .pme_desc = "Multiplier stall cycles"
	},
	{.pme_name="WRITE_BUFFER_FULL_STALL_CYCLES",
	 .pme_code = 0x00001c1c,
	 .pme_counters = 0x3,
	 .pme_desc = "Write buffer full stall cycles"
	},
	{.pme_name="FP_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "Floating-point instructions issued"
	},
	{.pme_name="JTLB_DATA_MISSES",
	 .pme_code = 0x00001010,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint TLB data misses"
	},
	{.pme_name="FP_EXCEPTION_STALL_CYCLES",
	 .pme_code = 0x00001717,
	 .pme_counters = 0x3,
	 .pme_desc = "FP possible exception cycles"
	},
	{.pme_name="SCACHE_MISSES",
	 .pme_code = 0x00000a0a,
	 .pme_counters = 0x3,
	 .pme_desc = "Secondary cache misses"
	},
	{.pme_name="BRANCHES_ISSUED",
	 .pme_code = 0x00001212,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches issued"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00000b0b,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction cache misses"
	},
	{.pme_name="INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "Total instructions issued"
	},
	{.pme_name="JTLB_INSTRUCTION_MISSES",
	 .pme_code = 0x00000f0f,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint TLB instruction misses"
	},
	{.pme_name="LOAD_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Load instructions issued"
	},
	{.pme_name="EXTERNAL_CACHE_MISSES",
	 .pme_code = 0x00000808,
	 .pme_counters = 0x3,
	 .pme_desc = "External Cache Misses"
	},
	{.pme_name="BRANCHES_TAKEN",
	 .pme_code = 0x00001111,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches taken"
	},
	{.pme_name="DTLB_MISSES",
	 .pme_code = 0x00000d0d,
	 .pme_counters = 0x3,
	 .pme_desc = "Data TLB misses"
	},
	{.pme_name="CACHE_INSTRUCTION_STALL_CYCLES",
	 .pme_code = 0x00001d1d,
	 .pme_counters = 0x3,
	 .pme_desc = "Cache instruction stall cycles"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Clock cycles"
	},
	{.pme_name="COP0_SLIP_CYCLES",
	 .pme_code = 0x00001919,
	 .pme_counters = 0x3,
	 .pme_desc = "Coprocessor 0 slip cycles"
	},
	{.pme_name="ITLB_MISSES",
	 .pme_code = 0x00000e0e,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction TLB misses"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x00000c0c,
	 .pme_counters = 0x3,
	 .pme_desc = "Data cache misses"
	},
};

static pme_gen_mips64_entry_t gen_mips64_rm9000_pe[] = {
	{.pme_name="FP_POSSIBLE_EXCEPTION_CYCLES",
	 .pme_code = 0x00001717,
	 .pme_counters = 0x3,
	 .pme_desc = "Floating-point possible exception cycles"
	},
	{.pme_name="STORE_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000505,
	 .pme_counters = 0x3,
	 .pme_desc = "Store instructions issued"
	},
	{.pme_name="STALL_CYCLES",
	 .pme_code = 0x00000909,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles"
	},
	{.pme_name="L2_WRITEBACKS",
	 .pme_code = 0x00001313,
	 .pme_counters = 0x3,
	 .pme_desc = "L2 cache writebacks"
	},
	{.pme_name="NONBLOCKING_LOAD_SLIP_CYCLES",
	 .pme_code = 0x00001a1a,
	 .pme_counters = 0x3,
	 .pme_desc = "Slip cycles due to pending non-blocking loads"
	},
	{.pme_name="NONBLOCKING_LOAD_PENDING_EXCEPTION_STALL_CYCLES",
	 .pme_code = 0x00001e1e,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles due to pending non-blocking loads - stall start of exception"
	},
	{.pme_name="BRANCH_MISSPREDICTS",
	 .pme_code = 0x00000707,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch mispredictions"
	},
	{.pme_name="DCACHE_MISS_STALL_CYCLES",
	 .pme_code = 0x00001515,
	 .pme_counters = 0x3,
	 .pme_desc = "Dcache-miss stall cycles"
	},
	{.pme_name="WRITE_BUFFER_FULL_STALL_CYCLES",
	 .pme_code = 0x00001b1b,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles due to a full write buffer"
	},
	{.pme_name="INT_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Integer instructions issued"
	},
	{.pme_name="FP_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "Floating-point instructions issued"
	},
	{.pme_name="JTLB_DATA_MISSES",
	 .pme_code = 0x00001010,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint TLB data misses"
	},
	{.pme_name="L2_CACHE_MISSES",
	 .pme_code = 0x00000a0a,
	 .pme_counters = 0x3,
	 .pme_desc = "L2 cache misses"
	},
	{.pme_name="DCACHE_WRITEBACKS",
	 .pme_code = 0x00001414,
	 .pme_counters = 0x3,
	 .pme_desc = "Dcache writebacks"
	},
	{.pme_name="BRANCHES_ISSUED",
	 .pme_code = 0x00001212,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch instructions issued"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00000b0b,
	 .pme_counters = 0x3,
	 .pme_desc = "Icache misses"
	},
	{.pme_name="INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "Instructions issued"
	},
	{.pme_name="MULTIPLIER_BUSY_SLIP_CYCLES",
	 .pme_code = 0x00001818,
	 .pme_counters = 0x3,
	 .pme_desc = "Slip cycles due to busy multiplier"
	},
	{.pme_name="INSTRUCTIONS_DUAL_ISSUED",
	 .pme_code = 0x00000606,
	 .pme_counters = 0x3,
	 .pme_desc = "Dual-issued instruction pairs"
	},
	{.pme_name="CACHE_INSN_STALL_CYCLES",
	 .pme_code = 0x00001c1c,
	 .pme_counters = 0x3,
	 .pme_desc = "Stall cycles due to cache instructions"
	},
	{.pme_name="JTLB_INSTRUCTION_MISSES",
	 .pme_code = 0x00000f0f,
	 .pme_counters = 0x3,
	 .pme_desc = "Joint TLB instruction misses"
	},
	{.pme_name="LOAD_INSTRUCTIONS_ISSUED",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Load instructions issued"
	},
	{.pme_name="CACHE_REMISSES",
	 .pme_code = 0x00001616,
	 .pme_counters = 0x3,
	 .pme_desc = "Cache remisses"
	},
	{.pme_name="BRANCHES_TAKEN",
	 .pme_code = 0x00001111,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches taken"
	},
	{.pme_name="DTLB_MISSES",
	 .pme_code = 0x00000d0d,
	 .pme_counters = 0x3,
	 .pme_desc = "Data TLB misses"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Processor clock cycles"
	},
	{.pme_name="COP0_SLIP_CYCLES",
	 .pme_code = 0x00001919,
	 .pme_counters = 0x3,
	 .pme_desc = "Co-processor 0 slip cycles"
	},
	{.pme_name="ITLB_MISSES",
	 .pme_code = 0x00000e0e,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction TLB misses"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x00000c0c,
	 .pme_counters = 0x3,
	 .pme_desc = "Dcache misses"
	},
};

static pme_gen_mips64_entry_t gen_mips64_sb1_pe[] = {
	{.pme_name="DATA_DEPENDENCY_REPLAY",
	 .pme_code = 0x1e1e1e1e,
	 .pme_counters = 0xf,
	 .pme_desc = "Data dependency replay"
	},
	{.pme_name="DCACHE_READ_MISS",
	 .pme_code = 0x0f0f0f00,
	 .pme_counters = 0xe,
	 .pme_desc = "Dcache read results in a miss"
	},
	{.pme_name="R_RESP_OTHER_CORE_D_MOD",
	 .pme_code = 0x19191900,
	 .pme_counters = 0xe,
	 .pme_desc = "Read response comes from the other core with D_MOD set"
	},
	{.pme_name="RQ_LENGTH",
	 .pme_code = 0x01010100,
	 .pme_counters = 0xe,
	 .pme_desc = "Read queue length"
	},
	{.pme_name="READ_RQ_NOPS_SENT_TO_ABUS",
	 .pme_code = 0x14141400,
	 .pme_counters = 0xe,
	 .pme_desc = "Read requests and NOPs sent to ZB Abus"
	},
	{.pme_name="R_RESP_OTHER_CORE",
	 .pme_code = 0x18181800,
	 .pme_counters = 0xe,
	 .pme_desc = "Read response comes from the other core"
	},
	{.pme_name="SNOOP_RQ_HITS",
	 .pme_code = 0x16161600,
	 .pme_counters = 0xe,
	 .pme_desc = "Snoop request hits anywhere"
	},
	{.pme_name="LOAD_SURVIVED_STAGE4",
	 .pme_code = 0x08080800,
	 .pme_counters = 0xe,
	 .pme_desc = "Load survived stage 4"
	},
	{.pme_name="BRANCH_PREDICTED_TAKEN",
	 .pme_code = 0x2e2e2e00,
	 .pme_counters = 0xe,
	 .pme_desc = "Predicted taken conditional branch"
	},
	{.pme_name="ISSUE_L1",
	 .pme_code = 0x29292900,
	 .pme_counters = 0xe,
	 .pme_desc = "Issue to L0"
	},
	{.pme_name="ANY_REPLAY",
	 .pme_code = 0x1f1f1f1f,
	 .pme_counters = 0xf,
	 .pme_desc = "Any replay except mispredict"
	},
	{.pme_name="LD_ST_HITS_PREFETCH_IN_QUEUE",
	 .pme_code = 0x06060600,
	 .pme_counters = 0xe,
	 .pme_desc = "Load/store hits prefetch in read queue"
	},
	{.pme_name="NOT_DATA_READY",
	 .pme_code = 0x23232300,
	 .pme_counters = 0xe,
	 .pme_desc = "Not data ready"
	},
	{.pme_name="DCFIFO",
	 .pme_code = 0x1c1c1c1c,
	 .pme_counters = 0xf,
	 .pme_desc = "DCFIFO"
	},
	{.pme_name="ISSUE_E1",
	 .pme_code = 0x2b2b2b00,
	 .pme_counters = 0xe,
	 .pme_desc = "Issue to E1"
	},
	{.pme_name="PREFETCH_HITS_CACHE_OR_READ_Q",
	 .pme_code = 0x05050500,
	 .pme_counters = 0xe,
	 .pme_desc = "Prefetch hits in cache or read queue"
	},
	{.pme_name="BRANCH_STAGE4",
	 .pme_code = 0x2c2c2c00,
	 .pme_counters = 0xe,
	 .pme_desc = "Branch survived stage 4"
	},
	{.pme_name="SNOOP_ADDR_Q_FULL",
	 .pme_code = 0x17171700,
	 .pme_counters = 0xe,
	 .pme_desc = "Snoop address queue is full"
	},
	{.pme_name="CONSUMER_WAITING_FOR_LOAD",
	 .pme_code = 0x22222200,
	 .pme_counters = 0xe,
	 .pme_desc = "load consumer waiting for dfill"
	},
	{.pme_name="VICTIM_WRITEBACK",
	 .pme_code = 0x0d0d0d00,
	 .pme_counters = 0xe,
	 .pme_desc = "A writeback occurs due to replacement"
	},
	{.pme_name="BRANCH_MISSPREDICTS",
	 .pme_code = 0x2f2f2f00,
	 .pme_counters = 0xe,
	 .pme_desc = "Branch mispredicts"
	},
	{.pme_name="UPGRADE_SHARED_TO_EXCLUSIVE",
	 .pme_code = 0x07070700,
	 .pme_counters = 0xe,
	 .pme_desc = "A line is upgraded from shared to exclusive"
	},
	{.pme_name="READ_HITS_READ_Q",
	 .pme_code = 0x04040400,
	 .pme_counters = 0xe,
	 .pme_desc = "Read hits in read queue"
	},
	{.pme_name="INSN_STAGE4",
	 .pme_code = 0x27272700,
	 .pme_counters = 0xe,
	 .pme_desc = "One or more instructions survives stage 4"
	},
	{.pme_name="UNCACHED_RQ_LENGTH",
	 .pme_code = 0x02020200,
	 .pme_counters = 0xe,
	 .pme_desc = "Number of valid uncached entries in read queue"
	},
	{.pme_name="READ_RQ_SENT_TO_ABUS",
	 .pme_code = 0x17171700,
	 .pme_counters = 0xe,
	 .pme_desc = "Read requests sent to ZB Abus"
	},
	{.pme_name="DCACHE_FILL_SHARED_LINE",
	 .pme_code = 0x0b0b0b00,
	 .pme_counters = 0xe,
	 .pme_desc = "Dcache is filled with shared line"
	},
	{.pme_name="ISSUE_CONFLICT_DUE_IMISS",
	 .pme_code = 0x25252500,
	 .pme_counters = 0xe,
	 .pme_desc = "issue conflict due to imiss using LS0"
	},
	{.pme_name="NO_VALID_INSN",
	 .pme_code = 0x21212100,
	 .pme_counters = 0xe,
	 .pme_desc = "No valid instr to issue"
	},
	{.pme_name="ISSUE_E0",
	 .pme_code = 0x2a2a2a00,
	 .pme_counters = 0xe,
	 .pme_desc = "Issue to E0"
	},
	{.pme_name="INSN_SURVIVED_STAGE7",
	 .pme_code = 0x00000000,
	 .pme_counters = 0xe,
	 .pme_desc = "Instruction survived stage 7"
	},
	{.pme_name="BRANCH_REALLY_TAKEN",
	 .pme_code = 0x2d2d2d00,
	 .pme_counters = 0xe,
	 .pme_desc = "Conditional branch was really taken"
	},
	{.pme_name="STORE_COND_FAILED",
	 .pme_code = 0x1a1a1a00,
	 .pme_counters = 0xe,
	 .pme_desc = "Failed store conditional"
	},
	{.pme_name="MAX_ISSUE",
	 .pme_code = 0x20202000,
	 .pme_counters = 0xe,
	 .pme_desc = "Max issue"
	},
	{.pme_name="BIU_STALLS_ON_ZB_ADDR_BUS",
	 .pme_code = 0x11111100,
	 .pme_counters = 0xe,
	 .pme_desc = "BIU stalls on ZB addr bus"
	},
	{.pme_name="STORE_SURVIVED_STAGE4",
	 .pme_code = 0x09090900,
	 .pme_counters = 0xe,
	 .pme_desc = "Store survived stage 4"
	},
	{.pme_name="RESOURCE_CONSTRAINT",
	 .pme_code = 0x24242400,
	 .pme_counters = 0xe,
	 .pme_desc = "Resource (L0/1 E0/1) constraint"
	},
	{.pme_name="DCACHE_FILL_REPLAY",
	 .pme_code = 0x1b1b1b1b,
	 .pme_counters = 0xf,
	 .pme_desc = "Dcache fill replay"
	},
	{.pme_name="BIU_STALLS_ON_ZB_DATA_BUS",
	 .pme_code = 0x12121200,
	 .pme_counters = 0xe,
	 .pme_desc = "BIU stalls on ZB data bus"
	},
	{.pme_name="ISSUE_CONFLICT_DUE_DFILL",
	 .pme_code = 0x26262600,
	 .pme_counters = 0xe,
	 .pme_desc = "issue conflict due to dfill using LS0/1"
	},
	{.pme_name="WRITEBACK_RETURNS",
	 .pme_code = 0x0f0f0f00,
	 .pme_counters = 0xe,
	 .pme_desc = "Number of instruction returns"
	},
	{.pme_name="DCACHE_FILLED_SHD_NONC_EXC",
	 .pme_code = 0x0a0a0a00,
	 .pme_counters = 0xe,
	 .pme_desc = "Dcache is filled (shared, nonc, exclusive)"
	},
	{.pme_name="ISSUE_L0",
	 .pme_code = 0x28282800,
	 .pme_counters = 0xe,
	 .pme_desc = "Issue to L0"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x10101010,
	 .pme_counters = 0xf,
	 .pme_desc = "Elapsed cycles"
	},
	{.pme_name="MBOX_RQ_WHEN_BIU_BUSY",
	 .pme_code = 0x0e0e0e00,
	 .pme_counters = 0xe,
	 .pme_desc = "MBOX requests to BIU when BIU busy"
	},
	{.pme_name="MBOX_REPLAY",
	 .pme_code = 0x1d1d1d1d,
	 .pme_counters = 0xf,
	 .pme_desc = "MBOX replay"
	},
};

static pme_gen_mips64_entry_t gen_mips64_vr5432_pe[] = {
	{.pme_name="INSTRUCTIONS_EXECUTED",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "(Instructions executed)/2 and truncated"
	},
	{.pme_name="JTLB_REFILLS",
	 .pme_code = 0x00000707,
	 .pme_counters = 0x3,
	 .pme_desc = "JTLB refills"
	},
	{.pme_name="BRANCHES",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch execution (no jumps or jump registers)"
	},
	{.pme_name="FP_INSTRUCTIONS",
	 .pme_code = 0x00000505,
	 .pme_counters = 0x3,
	 .pme_desc = "(FP instruction execution) / 2 and truncated excluding cp1 loads and stores"
	},
	{.pme_name="BRANCHES_MISPREDICTED",
	 .pme_code = 0x00000a0a,
	 .pme_counters = 0x3,
	 .pme_desc = "Branches mispredicted"
	},
	{.pme_name="DOUBLEWORDS_FLUSHED",
	 .pme_code = 0x00000606,
	 .pme_counters = 0x3,
	 .pme_desc = "Doublewords flushed to main memory (no uncached stores)"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00000909,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction cache misses (no D-cache misses)"
	},
	{.pme_name="LOAD_PREF_CACHE_INSTRUCTIONS",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "Load, prefetch/CacheOps execution (no sync)"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Processor cycles (PClock)"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x00000808,
	 .pme_counters = 0x3,
	 .pme_desc = "Data cache misses (no I-cache misses)"
	},
	{.pme_name="STORES",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Store execution"
	},
};

static pme_gen_mips64_entry_t gen_mips64_vr5500_pe[] = {
	{.pme_name="INSTRUCTIONS_EXECUTED",
	 .pme_code = 0x00000101,
	 .pme_counters = 0x3,
	 .pme_desc = "Instructions executed"
	},
	{.pme_name="JTLB_REFILLS",
	 .pme_code = 0x00000707,
	 .pme_counters = 0x3,
	 .pme_desc = "TLB refill"
	},
	{.pme_name="BRANCHES",
	 .pme_code = 0x00000404,
	 .pme_counters = 0x3,
	 .pme_desc = "Execution of branch instruction"
	},
	{.pme_name="FP_INSTRUCTIONS",
	 .pme_code = 0x00000505,
	 .pme_counters = 0x3,
	 .pme_desc = "Execution of floating-point instruction"
	},
	{.pme_name="BRANCHES_MISPREDICTED",
	 .pme_code = 0x00000a0a,
	 .pme_counters = 0x3,
	 .pme_desc = "Branch prediction miss"
	},
	{.pme_name="DOUBLEWORDS_FLUSHED",
	 .pme_code = 0x00000606,
	 .pme_counters = 0x3,
	 .pme_desc = "Doubleword flush to main memory"
	},
	{.pme_name="ICACHE_MISSES",
	 .pme_code = 0x00000909,
	 .pme_counters = 0x3,
	 .pme_desc = "Instruction cache miss"
	},
	{.pme_name="LOAD_PREF_CACHE_INSTRUCTIONS",
	 .pme_code = 0x00000202,
	 .pme_counters = 0x3,
	 .pme_desc = "Execution of load/prefetch/cache instruction"
	},
	{.pme_name="CYCLES",
	 .pme_code = 0x00000000,
	 .pme_counters = 0x3,
	 .pme_desc = "Processor clock cycles"
	},
	{.pme_name="DCACHE_MISSES",
	 .pme_code = 0x00000808,
	 .pme_counters = 0x3,
	 .pme_desc = "Data cache miss"
	},
	{.pme_name="STORES",
	 .pme_code = 0x00000303,
	 .pme_counters = 0x3,
	 .pme_desc = "Execution of store instruction"
	},
};

