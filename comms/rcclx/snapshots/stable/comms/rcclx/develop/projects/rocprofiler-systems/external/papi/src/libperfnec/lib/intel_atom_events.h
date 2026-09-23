/*
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmai.com>
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */

/* table 18.11 */
#define INTEL_ATOM_MESI \
		{ .pme_uname = "MESI",\
		  .pme_udesc = "Any cacheline access",\
		  .pme_ucode = 0xf\
		},\
		{ .pme_uname = "I_STATE",\
		  .pme_udesc = "Invalid cacheline",\
		  .pme_ucode = 0x1\
		},\
		{ .pme_uname = "S_STATE",\
		  .pme_udesc = "Shared cacheline",\
		  .pme_ucode = 0x2\
		},\
		{ .pme_uname = "E_STATE",\
		  .pme_udesc = "Exclusive cacheline",\
		  .pme_ucode = 0x4\
		},\
		{ .pme_uname = "M_STATE",\
		  .pme_udesc = "Modified cacheline",\
		  .pme_ucode = 0x8\
		}

/* table 18.9 */
#define INTEL_ATOM_AGENT \
		{ .pme_uname = "THIS_AGENT",\
		  .pme_udesc = "This agent",\
		  .pme_ucode = 0x00\
		},\
		{ .pme_uname = "ALL_AGENTS",\
		  .pme_udesc = "Any agent on the bus",\
		  .pme_ucode = 0x20\
		}

/* table 18.8 */
#define INTEL_ATOM_CORE \
		{ .pme_uname = "SELF",\
		  .pme_udesc = "This core",\
		  .pme_ucode = 0x40\
		},\
		{ .pme_uname = "BOTH_CORES",\
		  .pme_udesc = "Both cores",\
		  .pme_ucode = 0xc0\
		}

/* table 18.10 */
#define INTEL_ATOM_PREFETCH \
		{ .pme_uname = "ANY",\
		  .pme_udesc = "All inclusive",\
		  .pme_ucode = 0x30\
		},\
		{ .pme_uname = "PREFETCH",\
		  .pme_udesc = "Hardware prefetch only",\
		  .pme_ucode = 0x10\
		}

static pme_intel_atom_entry_t intel_atom_pe[]={
	/*
	 * BEGIN architectural perfmon events
	 */
/* 0 */{.pme_name  = "UNHALTED_CORE_CYCLES",
	.pme_code  = 0x003c,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED1,
	.pme_desc  = "Unhalted core cycles",
       },
/* 1 */{.pme_name  = "UNHALTED_REFERENCE_CYCLES",
	.pme_code  = 0x013c,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED2_ONLY,
	.pme_desc  = "Unhalted reference cycles. Measures bus cycles"
	},
/* 2 */{.pme_name  = "INSTRUCTIONS_RETIRED",
	.pme_code  = 0xc0,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED0|PFMLIB_INTEL_ATOM_PEBS,
	.pme_desc  = "Instructions retired"
	},
/* 3 */{.pme_name = "LAST_LEVEL_CACHE_REFERENCES",
	.pme_code = 0x4f2e,
	.pme_desc = "Last level of cache references"
	},
/* 4 */{.pme_name = "LAST_LEVEL_CACHE_MISSES",
	.pme_code = 0x412e,
	.pme_desc = "Last level of cache misses",
       },
/* 5  */{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	.pme_code  = 0xc4,
	.pme_desc  = "Branch instructions retired"
	},
/* 6  */{.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	.pme_code  = 0xc5,
	.pme_flags = PFMLIB_INTEL_ATOM_PEBS,
	.pme_desc  = "Mispredicted branch instruction retired"
	},
	
	/*
	 * BEGIN non architectural events
	 */
	{ .pme_name   = "SIMD_INSTR_RETIRED",
	  .pme_desc   = "SIMD Instructions retired",
	  .pme_code   = 0xCE,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "L2_REJECT_BUSQ",
	  .pme_desc   = "Rejected L2 cache requests",
	  .pme_code   = 0x30,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_MESI,
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH
	  },
	  .pme_numasks = 9,
	}, 
	{ .pme_name   = "SIMD_SAT_INSTR_RETIRED",
	  .pme_desc   = "Saturated arithmetic instructions retired",
	  .pme_code   = 0xCF,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "ICACHE",
	  .pme_desc   = "Instruction fetches",
	  .pme_code   = 0x80,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "ACCESSES",
		  .pme_udesc  = "Instruction fetches, including uncacheacble fetches",
		  .pme_ucode  = 0x3,
		}, 
		{ .pme_uname  = "MISSES",
		  .pme_udesc  = "count all instructions fetches that miss tha icache or produce memory requests. This includes uncacheache fetches. Any instruction fetch miss is counted only once and not once for every cycle it is outstanding",
		  .pme_ucode  = 0x2,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_LOCK",
	  .pme_desc   = "L2 locked accesses",
	  .pme_code   = 0x2B,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_MESI,
		INTEL_ATOM_CORE	
	  },
	  .pme_numasks = 7
	}, 
	{ .pme_name   = "UOPS_RETIRED",
	  .pme_desc   = "Micro-ops retired",
	  .pme_code   = 0xC2,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY",
		  .pme_udesc  = "Micro-ops retired",
		  .pme_ucode  = 0x10,
		}, 
		{ .pme_uname  = "STALLED_CYCLES",
		  .pme_udesc  = "Cycles no micro-ops retired",
		  .pme_ucode  = 0x1d010, /* inv=1 cnt_mask=1 */
		}, 
		{ .pme_uname  = "STALLS",
		  .pme_udesc  = "Periods no micro-ops retired",
		  .pme_ucode  = 0x1d410, /* inv=1 edge=1, cnt_mask=1 */
		}, 
	  },
	  .pme_numasks = 3
	}, 
	{ .pme_name   = "L2_M_LINES_OUT",
	  .pme_desc   = "Modified lines evicted from the L2 cache",
	  .pme_code   = 0x27,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "SIMD_COMP_INST_RETIRED",
	  .pme_desc   = "Retired computational Streaming SIMD Extensions (SSE) instructions",
	  .pme_code   = 0xCA,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "PACKED_SINGLE",
		  .pme_udesc  = "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "SCALAR_SINGLE",
		  .pme_udesc  = "Retired computational Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .pme_ucode  = 0x2,
		}, 
		{ .pme_uname  = "PACKED_DOUBLE",
		  .pme_udesc  = "Retired computational Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .pme_ucode  = 0x4,
		}, 
		{ .pme_uname  = "SCALAR_DOUBLE",
		  .pme_udesc  = "Retired computational Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .pme_ucode  = 0x8,
		}, 
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "SNOOP_STALL_DRV",
	  .pme_desc   = "Bus stalled for snoops",
	  .pme_code   = 0x7E,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT,	
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_TRANS_BURST",
	  .pme_desc   = "Burst (full cache-line) bus transactions",
	  .pme_code   = 0x6E,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT,
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "SIMD_SAT_UOP_EXEC",
	  .pme_desc   = "SIMD saturated arithmetic micro-ops executed",
	  .pme_code   = 0xB1,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "S",
		  .pme_udesc  = "SIMD saturated arithmetic micro-ops executed",
		  .pme_ucode  = 0x0,
		}, 
		{ .pme_uname  = "AR",
		  .pme_udesc  = "SIMD saturated arithmetic micro-ops retired",
		  .pme_ucode  = 0x80,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "BUS_TRANS_IO",
	  .pme_desc   = "IO bus transactions",
	  .pme_code   = 0x6C,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_TRANS_RFO",
	  .pme_desc   = "RFO bus transactions",
	  .pme_code   = 0x66,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "SIMD_ASSIST",
	  .pme_desc   = "SIMD assists invoked",
	  .pme_code   = 0xCD,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "INST_RETIRED",
	  .pme_desc   = "Instructions retired",
	  .pme_code   = 0xC0,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY_P",
		  .pme_udesc  = "Instructions retired using generic counter (precise event)",
		  .pme_ucode  = 0x0,
	  	  .pme_flags = PFMLIB_INTEL_ATOM_PEBS
		}, 
	  },
	  .pme_numasks = 1
	}, 
	{ .pme_name   = "L1D_CACHE",
	  .pme_desc   = "L1 Cacheable Data Reads",
	  .pme_code   = 0x40,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "LD",
		  .pme_udesc  = "L1 Cacheable Data Reads",
		  .pme_ucode  = 0x21,
		}, 
		{ .pme_uname  = "ST",
		  .pme_udesc  = "L1 Cacheable Data Writes",
		  .pme_ucode  = 0x22,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "MUL",
	  .pme_desc   = "Multiply operations executed",
	  .pme_code   = 0x12,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "S",
		  .pme_udesc  = "Multiply operations executed",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "AR",
		  .pme_udesc  = "Multiply operations retired",
		  .pme_ucode  = 0x81,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "DIV",
	  .pme_desc   = "Divide operations executed",
	  .pme_code   = 0x13,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "S",
		  .pme_udesc  = "Divide operations executed",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "AR",
		  .pme_udesc  = "Divide operations retired",
		  .pme_ucode  = 0x81,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "BUS_TRANS_P",
	  .pme_desc   = "Partial bus transactions",
	  .pme_code   = 0x6b,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT,
		INTEL_ATOM_CORE,
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_IO_WAIT",
	  .pme_desc   = "IO requests waiting in the bus queue",
	  .pme_code   = 0x7F,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_M_LINES_IN",
	  .pme_desc   = "L2 cache line modifications",
	  .pme_code   = 0x25,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_LINES_IN",
	  .pme_desc   = "L2 cache misses",
	  .pme_code   = 0x24,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUSQ_EMPTY",
	  .pme_desc   = "Bus queue is empty",
	  .pme_code   = 0x7D,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_IFETCH",
	  .pme_desc   = "L2 cacheable instruction fetch requests",
	  .pme_code   = 0x28,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_MESI,
		INTEL_ATOM_CORE	
	  },
	  .pme_numasks = 7
	}, 
	{ .pme_name   = "BUS_HITM_DRV",
	  .pme_desc   = "HITM signal asserted",
	  .pme_code   = 0x7B,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "ITLB",
	  .pme_desc   = "ITLB hits",
	  .pme_code   = 0x82,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "FLUSH",
		  .pme_udesc  = "ITLB flushes",
		  .pme_ucode  = 0x4,
		}, 
		{ .pme_uname  = "MISSES",
		  .pme_udesc  = "ITLB misses",
		  .pme_ucode  = 0x2,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "BUS_TRANS_MEM",
	  .pme_desc   = "Memory bus transactions",
	  .pme_code   = 0x6F,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT,
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_TRANS_PWR",
	  .pme_desc   = "Partial write bus transaction",
	  .pme_code   = 0x6A,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT,
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BR_INST_DECODED",
	  .pme_desc   = "Branch instructions decoded",
	  .pme_code   = 0x1E0,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "BUS_TRANS_INVAL",
	  .pme_desc   = "Invalidate bus transactions",
	  .pme_code   = 0x69,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "SIMD_UOP_TYPE_EXEC",
	  .pme_desc   = "SIMD micro-ops executed",
	  .pme_code   = 0xB3,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "MUL_S",
		  .pme_udesc  = "SIMD packed multiply micro-ops executed",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "MUL_AR",
		  .pme_udesc  = "SIMD packed multiply micro-ops retired",
		  .pme_ucode  = 0x81,
		}, 
		{ .pme_uname  = "SHIFT_S",
		  .pme_udesc  = "SIMD packed shift micro-ops executed",
		  .pme_ucode  = 0x2,
		}, 
		{ .pme_uname  = "SHIFT_AR",
		  .pme_udesc  = "SIMD packed shift micro-ops retired",
		  .pme_ucode  = 0x82,
		}, 
		{ .pme_uname  = "PACK_S",
		  .pme_udesc  = "SIMD packed micro-ops executed",
		  .pme_ucode  = 0x4,
		}, 
		{ .pme_uname  = "PACK_AR",
		  .pme_udesc  = "SIMD packed micro-ops retired",
		  .pme_ucode  = 0x84,
		}, 
		{ .pme_uname  = "UNPACK_S",
		  .pme_udesc  = "SIMD unpacked micro-ops executed",
		  .pme_ucode  = 0x8,
		}, 
		{ .pme_uname  = "UNPACK_AR",
		  .pme_udesc  = "SIMD unpacked micro-ops retired",
		  .pme_ucode  = 0x88,
		}, 
		{ .pme_uname  = "LOGICAL_S",
		  .pme_udesc  = "SIMD packed logical micro-ops executed",
		  .pme_ucode  = 0x10,
		}, 
		{ .pme_uname  = "LOGICAL_AR",
		  .pme_udesc  = "SIMD packed logical micro-ops retired",
		  .pme_ucode  = 0x90,
		}, 
		{ .pme_uname  = "ARITHMETIC_S",
		  .pme_udesc  = "SIMD packed arithmetic micro-ops executed",
		  .pme_ucode  = 0x20,
		}, 
		{ .pme_uname  = "ARITHMETIC_AR",
		  .pme_udesc  = "SIMD packed arithmetic micro-ops retired",
		  .pme_ucode  = 0xA0,
		}, 
	  },
	  .pme_numasks = 12
	}, 
	{ .pme_name   = "SIMD_INST_RETIRED",
	  .pme_desc   = "Retired Streaming SIMD Extensions (SSE)",
	  .pme_code   = 0xC7,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "PACKED_SINGLE",
		  .pme_udesc  = "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "SCALAR_SINGLE",
		  .pme_udesc  = "Retired Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .pme_ucode  = 0x2,
		}, 
		{ .pme_uname  = "PACKED_DOUBLE",
		  .pme_udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .pme_ucode  = 0x4,
		}, 
		{ .pme_uname  = "SCALAR_DOUBLE",
		  .pme_udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .pme_ucode  = 0x8,
		}, 
		{ .pme_uname  = "VECTOR",
		  .pme_udesc  = "Retired Streaming SIMD Extensions 2 (SSE2) vector instructions",
		  .pme_ucode  = 0x10,
		}, 
		{ .pme_uname  = "ANY",
		  .pme_udesc  = "Retired Streaming SIMD instructions",
		  .pme_ucode  = 0x1F,
		}, 
	  },
	  .pme_numasks = 6
	}, 
	{ .pme_name   = "CYCLES_DIV_BUSY",
	  .pme_desc   = "Cycles the divider is busy",
	  .pme_code   = 0x14,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "PREFETCH",
	  .pme_desc   = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
	  .pme_code   = 0x7,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "PREFETCHT0",
		  .pme_udesc  = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
		  .pme_ucode  = 0x01,
		}, 
		{ .pme_uname  = "SW_L2",
		  .pme_udesc  = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions executed",
		  .pme_ucode  = 0x06,
		}, 
		{ .pme_uname  = "PREFETCHNTA",
		  .pme_udesc  = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		  .pme_ucode  = 0x08,
		}, 
	  },
	  .pme_numasks = 3
	}, 
	{ .pme_name   = "L2_RQSTS",
	  .pme_desc   = "L2 cache requests",
	  .pme_code   = 0x2E,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH,
		INTEL_ATOM_MESI
	  },
	  .pme_numasks = 9
	}, 
	{ .pme_name   = "SIMD_UOPS_EXEC",
	  .pme_desc   = "SIMD micro-ops executed (excluding stores)",
	  .pme_code   = 0xB0,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "S",
		  .pme_udesc  = "number of SMD saturated arithmetic micro-ops executed",
		  .pme_ucode  = 0x0,
		}, 
		{ .pme_uname  = "AR",
		  .pme_udesc  = "number of SIMD saturated arithmetic micro-ops retired",
		  .pme_ucode  = 0x80,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "HW_INT_RCV",
	  .pme_desc   = "Hardware interrupts received",
	  .pme_code   = 0xC8,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "BUS_TRANS_BRD",
	  .pme_desc   = "Burst read bus transactions",
	  .pme_code   = 0x65,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT,
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BOGUS_BR",
	  .pme_desc   = "Bogus branches",
	  .pme_code   = 0xE4,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "BUS_DATA_RCV",
	  .pme_desc   = "Bus cycles while processor receives data",
	  .pme_code   = 0x64,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "MACHINE_CLEARS",
	  .pme_desc   = "Self-Modifying Code detected",
	  .pme_code   = 0xC3,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "SMC",
		  .pme_udesc  = "Self-Modifying Code detected",
		  .pme_ucode  = 0x1,
		}, 
	  },
	  .pme_numasks = 1
	}, 
	{ .pme_name   = "BR_INST_RETIRED",
	  .pme_desc   = "Retired branch instructions",
	  .pme_code   = 0xC4,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY",
		  .pme_udesc  = "Retired branch instructions",
		  .pme_ucode  = 0x0,
		}, 
		{ .pme_uname  = "PRED_NOT_TAKEN",
		  .pme_udesc  = "Retired branch instructions that were predicted not-taken",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "MISPRED_NOT_TAKEN",
		  .pme_udesc  = "Retired branch instructions that were mispredicted not-taken",
		  .pme_ucode  = 0x2,
		}, 
		{ .pme_uname  = "PRED_TAKEN",
		  .pme_udesc  = "Retired branch instructions that were predicted taken",
		  .pme_ucode  = 0x4,
		}, 
		{ .pme_uname  = "MISPRED_TAKEN",
		  .pme_udesc  = "Retired branch instructions that were mispredicted taken",
		  .pme_ucode  = 0x8,
		}, 
		{ .pme_uname  = "MISPRED",
		  .pme_udesc  = "Retired mispredicted branch instructions (precise event)",
	  	  .pme_flags = PFMLIB_INTEL_ATOM_PEBS,
		  .pme_ucode  = 0xA,
		}, 
		{ .pme_uname  = "TAKEN",
		  .pme_udesc  = "Retired taken branch instructions",
		  .pme_ucode  = 0xC,
		}, 
		{ .pme_uname  = "ANY1",
		  .pme_udesc  = "Retired branch instructions",
		  .pme_ucode  = 0xF,
		}, 
	  },
	  .pme_numasks = 8
	}, 
	{ .pme_name   = "L2_ADS",
	  .pme_desc   = "Cycles L2 address bus is in use",
	  .pme_code   = 0x21,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "EIST_TRANS",
	  .pme_desc   = "Number of Enhanced Intel SpeedStep(R) Technology (EIST) transitions",
	  .pme_code   = 0x3A,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "BUS_TRANS_WB",
	  .pme_desc   = "Explicit writeback bus transactions",
	  .pme_code   = 0x67,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "MACRO_INSTS",
	  .pme_desc   = "Macro instructions decoded",
	  .pme_code   = 0xAA,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "NON_CISC_DECODED",
		  .pme_udesc  = "Non-CISC macro instructions decoded",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "ALL_DECODED",
		  .pme_udesc  = "All Instructions decoded",
		  .pme_ucode  = 0x3,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_LINES_OUT",
	  .pme_desc   = "L2 cache lines evicted",
	  .pme_code   = 0x26,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "L2_LD",
	  .pme_desc   = "L2 cache reads",
	  .pme_code   = 0x29,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_PREFETCH,
		INTEL_ATOM_MESI
	  },
	  .pme_numasks = 9
	}, 
	{ .pme_name   = "SEGMENT_REG_LOADS",
	  .pme_desc   = "Number of segment register loads",
	  .pme_code   = 0x6,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY",
		  .pme_udesc  = "Number of segment register loads",
		  .pme_ucode  = 0x80,
		}, 
	  },
	  .pme_numasks = 1
	}, 
	{ .pme_name   = "L2_NO_REQ",
	  .pme_desc   = "Cycles no L2 cache requests are pending",
	  .pme_code   = 0x32,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "THERMAL_TRIP",
	  .pme_desc   = "Number of thermal trips",
	  .pme_code   = 0xC03B,
	  .pme_flags  = 0,
	}, 
	{ .pme_name   = "EXT_SNOOP",
	  .pme_desc   = "External snoops",
	  .pme_code   = 0x77,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_MESI,
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 7
	}, 
	{ .pme_name   = "BACLEARS",
	  .pme_desc   = "BACLEARS asserted",
	  .pme_code   = 0xE6,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY",
		  .pme_udesc  = "BACLEARS asserted",
		  .pme_ucode  = 0x1,
		}, 
	  },
	  .pme_numasks = 1
	}, 
	{ .pme_name   = "CYCLES_INT_MASKED",
	  .pme_desc   = "Cycles during which interrupts are disabled",
	  .pme_code   = 0xC6,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "CYCLES_INT_MASKED",
		  .pme_udesc  = "Cycles during which interrupts are disabled",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "CYCLES_INT_PENDING_AND_MASKED",
		  .pme_udesc  = "Cycles during which interrupts are pending and disabled",
		  .pme_ucode  = 0x2,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "FP_ASSIST",
	  .pme_desc   = "Floating point assists",
	  .pme_code   = 0x11,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "S",
		  .pme_udesc  = "Floating point assists for executed instructions",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "AR",
		  .pme_udesc  = "Floating point assists for retired instructions",
		  .pme_ucode  = 0x81,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_ST",
	  .pme_desc   = "L2 store requests",
	  .pme_code   = 0x2A,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_MESI,
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 7
	}, 
	{ .pme_name   = "BUS_TRANS_DEF",
	  .pme_desc   = "Deferred bus transactions",
	  .pme_code   = 0x6D,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "DATA_TLB_MISSES",
	  .pme_desc   = "Memory accesses that missed the DTLB",
	  .pme_code   = 0x8,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "DTLB_MISS",
		  .pme_udesc  = "Memory accesses that missed the DTLB",
		  .pme_ucode  = 0x7,
		}, 
		{ .pme_uname  = "DTLB_MISS_LD",
		  .pme_udesc  = "DTLB misses due to load operations",
		  .pme_ucode  = 0x5,
		}, 
		{ .pme_uname  = "L0_DTLB_MISS_LD",
		  .pme_udesc  = "L0 (micro-TLB) misses due to load operations",
		  .pme_ucode  = 0x9,
		}, 
		{ .pme_uname  = "DTLB_MISS_ST",
		  .pme_udesc  = "DTLB misses due to store operations",
		  .pme_ucode  = 0x6,
		}, 
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_BNR_DRV",
	  .pme_desc   = "Number of Bus Not Ready signals asserted",
	  .pme_code   = 0x61,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "STORE_FORWARDS",
	  .pme_desc   = "All store forwards",
	  .pme_code   = 0x2,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "GOOD",
		  .pme_udesc  = "Good store forwards",
		  .pme_ucode  = 0x81,
		}, 
	  },
	  .pme_numasks = 1
	}, 
	{ .pme_name  = "CPU_CLK_UNHALTED",
	  .pme_code  = 0x3c,
	  .pme_desc  =  "Core cycles when core is not halted",
	  .pme_flags = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
	  .pme_umasks = {
		{ .pme_uname = "CORE_P",
		  .pme_udesc = "Core cycles when core is not halted",
		  .pme_ucode = 0x0,
		},
		{ .pme_uname = "BUS",
		  .pme_udesc = "Bus cycles when core is not halted. This event can give a measurement of the elapsed time. This events has a constant ratio with CPU_CLK_UNHALTED:REF event, which is the maximum bus to processor frequency ratio",
		  .pme_ucode = 0x1,
		},
		{ .pme_uname = "NO_OTHER",
		  .pme_udesc = "Bus cycles when core is active and other is halted",
		  .pme_ucode = 0x2,
		},

	   },
	   .pme_numasks = 3
	},
	{ .pme_name   = "BUS_TRANS_ANY",
	  .pme_desc   = "All bus transactions",
	  .pme_code   = 0x70,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE,
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "MEM_LOAD_RETIRED",
	  .pme_desc   = "Retired loads that hit the L2 cache (precise event)",
	  .pme_code   = 0xCB,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		{ .pme_uname  = "L2_HIT",
		  .pme_udesc  = "Retired loads that hit the L2 cache (precise event)",
		  .pme_ucode  = 0x1,
	  	  .pme_flags = PFMLIB_INTEL_ATOM_PEBS
		}, 
		{ .pme_uname  = "L2_MISS",
		  .pme_udesc  = "Retired loads that miss the L2 cache (precise event)",
		  .pme_ucode  = 0x2,
	  	  .pme_flags = PFMLIB_INTEL_ATOM_PEBS
		}, 
		{ .pme_uname  = "DTLB_MISS",
		  .pme_udesc  = "Retired loads that miss the DTLB (precise event)",
		  .pme_ucode  = 0x4,
	  	  .pme_flags = PFMLIB_INTEL_ATOM_PEBS
		}, 
	  },
	  .pme_numasks = 3
	}, 
	{ .pme_name   = "X87_COMP_OPS_EXE",
	  .pme_desc   = "Floating point computational micro-ops executed",
	  .pme_code   = 0x10,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "ANY_S",
		  .pme_udesc  = "Floating point computational micro-ops executed",
		  .pme_ucode  = 0x1,
		}, 
		{ .pme_uname  = "ANY_AR",
		  .pme_udesc  = "Floating point computational micro-ops retired",
		  .pme_ucode  = 0x81,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "PAGE_WALKS",
	  .pme_desc   = "Number of page-walks executed",
	  .pme_code   = 0xC,
	  .pme_flags  = PFMLIB_INTEL_ATOM_UMASK_NCOMBO,
    	  .pme_umasks = { 
		{ .pme_uname  = "WALKS",
		  .pme_udesc  = "Number of page-walks executed",
		  .pme_ucode  = 0x3 | 1ul << 10,
		}, 
		{ .pme_uname  = "CYCLES",
		  .pme_udesc  = "Duration of page-walks in core cycles",
		  .pme_ucode  = 0x3,
		}, 
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "BUS_LOCK_CLOCKS",
	  .pme_desc   = "Bus cycles when a LOCK signal is asserted",
	  .pme_code   = 0x63,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT,
		INTEL_ATOM_CORE	
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_REQUEST_OUTSTANDING",
	  .pme_desc   = "Outstanding cacheable data read bus requests duration",
	  .pme_code   = 0x60,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT,
		INTEL_ATOM_CORE	
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_TRANS_IFETCH",
	  .pme_desc   = "Instruction-fetch bus transactions",
	  .pme_code   = 0x68,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT,
		INTEL_ATOM_CORE	
	  },
	  .pme_numasks = 4
	}, 
	{ .pme_name   = "BUS_HIT_DRV",
	  .pme_desc   = "HIT signal asserted",
	  .pme_code   = 0x7A,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "BUS_DRDY_CLOCKS",
	  .pme_desc   = "Bus cycles when data is sent on the bus",
	  .pme_code   = 0x62,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_AGENT
	  },
	  .pme_numasks = 2
	}, 
	{ .pme_name   = "L2_DBUS_BUSY",
	  .pme_desc   = "Cycles the L2 cache data bus is busy",
	  .pme_code   = 0x22,
	  .pme_flags  = 0,
    	  .pme_umasks = { 
		INTEL_ATOM_CORE
	  },
	  .pme_numasks = 2
	},
};
#define PME_INTEL_ATOM_UNHALTED_CORE_CYCLES		0
#define PME_INTEL_ATOM_INSTRUCTIONS_RETIRED		2
#define PME_INTEL_ATOM_EVENT_COUNT	   		(sizeof(intel_atom_pe)/sizeof(pme_intel_atom_entry_t))

