/*
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */

#define INTEL_CORE_MESI_UMASKS \
		{ .pme_uname = "MESI",\
		  .pme_udesc = "Any cacheline access (default)",\
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

#define INTEL_CORE_SPECIFICITY_UMASKS \
		{ .pme_uname = "SELF",\
		  .pme_udesc = "This core",\
		  .pme_ucode = 0x40\
		},\
		{ .pme_uname = "BOTH_CORES",\
		  .pme_udesc = "Both cores",\
		  .pme_ucode = 0xc0\
		}

#define INTEL_CORE_HW_PREFETCH_UMASKS \
		{ .pme_uname = "ANY",\
		  .pme_udesc = "All inclusive",\
		  .pme_ucode = 0x30\
		},\
		{ .pme_uname = "PREFETCH",\
		  .pme_udesc = "Hardware prefetch only",\
		  .pme_ucode = 0x10\
		}

#define INTEL_CORE_AGENT_UMASKS \
		{ .pme_uname = "THIS_AGENT",\
		  .pme_udesc = "This agent",\
		  .pme_ucode = 0x00\
		},\
		{ .pme_uname = "ALL_AGENTS",\
		  .pme_udesc = "Any agent on the bus",\
		  .pme_ucode = 0x20\
		}


static pme_core_entry_t core_pe[]={
	/*
	 * BEGIN: architected Core events
	 */
	{.pme_name = "UNHALTED_CORE_CYCLES",
	 .pme_code = 0x003c,
	 .pme_flags = PFMLIB_CORE_FIXED1,
	 .pme_desc =  "count core clock cycles whenever the clock signal on the specific core is running (not halted). Alias to event CPU_CLK_UNHALTED:CORE_P"
	},
	{.pme_name = "INSTRUCTIONS_RETIRED",
	 .pme_code = 0x00c0,
	 .pme_flags = PFMLIB_CORE_FIXED0,
	 .pme_desc =  "count the number of instructions at retirement. Alias to event INST_RETIRED:ANY_P",
	},
	{.pme_name = "UNHALTED_REFERENCE_CYCLES",
	 .pme_code = 0x013c,
	 .pme_flags = PFMLIB_CORE_FIXED2_ONLY,
	 .pme_desc =  "Unhalted reference cycles. Alias to event CPU_CLK_UNHALTED:REF",
	},
	{.pme_name = "LAST_LEVEL_CACHE_REFERENCES",
	 .pme_code = 0x4f2e,
	 .pme_desc =  "count each request originating from the core to reference a cache line in the last level cache. The count may include speculation, but excludes cache line fills due to hardware prefetch. Alias to L2_RQSTS:SELF_DEMAND_MESI",
	},
	{.pme_name = "LAST_LEVEL_CACHE_MISSES",
	 .pme_code = 0x412e,
	 .pme_desc =  "count each cache miss condition for references to the last level cache. The event count may include speculation, but excludes cache line fills due to hardware prefetch. Alias to event L2_RQSTS:SELF_DEMAND_I_STATE",
	},
	{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	 .pme_code = 0x00c4,
	 .pme_desc =  "count branch instructions at retirement. Specifically, this event counts the retirement of the last micro-op of a branch instruction. Alias to event BR_INST_RETIRED:ANY",
	},
	{.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	 .pme_code = 0x00c5,
	 .pme_desc =  "count mispredicted branch instructions at retirement. Specifically, this event counts at retirement of the last micro-op of a branch instruction in the architectural path of the execution and experienced misprediction in the branch prediction hardware. Alias to BR_INST_RETIRED:MISPRED",
	},
	/*
	 * END: architected events
	 */
	/*
	 * BEGIN: Core 2 Duo events
	 */
	{ .pme_name = "RS_UOPS_DISPATCHED_CYCLES",
	  .pme_code = 0xa1,
	  .pme_flags = PFMLIB_CORE_PMC0,
	  .pme_desc =  "Cycles micro-ops dispatched for execution",
	  .pme_umasks = {
		{ .pme_uname = "PORT_0",
		  .pme_udesc = "on port 0",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "PORT_1",
		  .pme_udesc = "on port 1",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "PORT_2",
		  .pme_udesc = "on port 2",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "PORT_3",
		  .pme_udesc = "on port 3",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "PORT_4",
		  .pme_udesc = "on port 4",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "PORT_5",
		  .pme_udesc = "on port 5",
		  .pme_ucode = 0x20
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "on any port",
		  .pme_ucode = 0x3f
		},
	   },	
	   .pme_numasks = 7

	},
	{ .pme_name = "RS_UOPS_DISPATCHED",
	  .pme_code = 0xa0,
	  .pme_desc =  "Number of micro-ops dispatched for execution",
	},
	{ .pme_name = "RS_UOPS_DISPATCHED_NONE",
	  .pme_code = 0xa0 | (1 << 23 | 1 << 24),
	  .pme_desc =  "Number of of cycles in which no micro-ops is dispatched for execution",
	},
	{ .pme_name = "LOAD_BLOCK",
	  .pme_code = 0x3,
	  .pme_flags = 0,
	  .pme_desc =  "Loads blocked",
	  .pme_umasks = {
		{ .pme_uname = "STA",
		  .pme_udesc = "Loads blocked by a preceding store with unknown address",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "STD",
		  .pme_udesc = "Loads blocked by a preceding store with unknown data",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "OVERLAP_STORE",
		  .pme_udesc = "Loads that partially overlap an earlier store, or 4K aliased with a previous store",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "UNTIL_RETIRE",
		  .pme_udesc = "Loads blocked until retirement",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "L1D",
		  .pme_udesc = "Loads blocked by the L1 data cache",
		  .pme_ucode = 0x20
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "SB_DRAIN_CYCLES",
	  .pme_code = 0x104,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles while stores are blocked due to store buffer drain"
	},
	{ .pme_name = "STORE_BLOCK",
	  .pme_code = 0x4,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles while store is waiting",
	  .pme_umasks = {
		{ .pme_uname = "ORDER",
		  .pme_udesc = "Cycles while store is waiting for a preceding store to be globally observed",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "SNOOP",
		  .pme_udesc = "A store is blocked due to a conflict with an external or internal snoop",
		  .pme_ucode = 0x8
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "SEGMENT_REG_LOADS",
	  .pme_code = 0x6,
	  .pme_flags = 0,
	  .pme_desc =  "Number of segment register loads"
	},
	{ .pme_name = "SSE_PRE_EXEC",
	  .pme_code = 0x7,
	  .pme_flags = 0,
	  .pme_desc =  "Streaming SIMD Extensions (SSE) Prefetch instructions executed",
	  .pme_umasks = {
		{ .pme_uname = "NTA",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "L1",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions executed",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "L2",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions executed",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "STORES",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) Weakly-ordered store instructions executed",
		  .pme_ucode = 0x3
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "DTLB_MISSES",
	  .pme_code = 0x8,
	  .pme_flags = 0,
	  .pme_desc =  "Memory accesses that missed the DTLB",
	  .pme_umasks = {
		{ .pme_uname = "ANY",
		  .pme_udesc = "Any memory access that missed the DTLB",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "MISS_LD",
		  .pme_udesc = "DTLB misses due to load operations",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "L0_MISS_LD",
		  .pme_udesc = "L0 DTLB misses due to load operations",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "MISS_ST",
		  .pme_udesc = "DTLB misses due to store operations",
		  .pme_ucode = 0x8
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "MEMORY_DISAMBIGUATION",
	  .pme_code = 0x9,
	  .pme_flags = 0,
	  .pme_desc =  "Memory disambiguation",
	  .pme_umasks = {
		{ .pme_uname = "RESET",
		  .pme_udesc = "Memory disambiguation reset cycles",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SUCCESS",
		  .pme_udesc = "Number of loads that were successfully disambiguated",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "PAGE_WALKS",
	  .pme_code = 0xc,
	  .pme_flags = 0,
	  .pme_desc =  "Number of page-walks executed",
	  .pme_umasks = {
		{ .pme_uname = "COUNT",
		  .pme_udesc = "Number of page-walks executed",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "CYCLES",
		  .pme_udesc = "Duration of page-walks in core cycles",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "FP_COMP_OPS_EXE",
	  .pme_code = 0x10,
	  .pme_flags = PFMLIB_CORE_PMC0,
	  .pme_desc =  "Floating point computational micro-ops executed"
	},
	{ .pme_name = "FP_ASSIST",
	  .pme_code = 0x11,
	  .pme_flags = PFMLIB_CORE_PMC1,
	  .pme_desc =  "Floating point assists"
	},
	{ .pme_name = "MUL",
	  .pme_code = 0x12,
	  .pme_flags = PFMLIB_CORE_PMC1,
	  .pme_desc =  "Multiply operations executed"
	},
	{ .pme_name = "DIV",
	  .pme_code = 0x13,
	  .pme_flags = PFMLIB_CORE_PMC1,
	  .pme_desc =  "Divide operations executed"
	},
	{ .pme_name = "CYCLES_DIV_BUSY",
	  .pme_code = 0x14,
	  .pme_flags = PFMLIB_CORE_PMC0,
	  .pme_desc =  "Cycles the divider is busy"
	},
	{ .pme_name = "IDLE_DURING_DIV",
	  .pme_code = 0x18,
	  .pme_flags = PFMLIB_CORE_PMC0,
	  .pme_desc =  "Cycles the divider is busy and all other execution units are idle"
	},
	{ .pme_name = "DELAYED_BYPASS",
	  .pme_code = 0x19,
	  .pme_flags = PFMLIB_CORE_PMC1,
	  .pme_desc =  "Delayed bypass",
	  .pme_umasks = {
		{ .pme_uname = "FP",
		  .pme_udesc = "Delayed bypass to FP operation",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "SIMD",
		  .pme_udesc = "Delayed bypass to SIMD operation",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "LOAD",
		  .pme_udesc = "Delayed bypass to load operation",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 3
	},
	{ .pme_name = "L2_ADS",
	  .pme_code = 0x21,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Cycles L2 address bus is in use",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "L2_DBUS_BUSY_RD",
	  .pme_code = 0x23,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Cycles the L2 transfers data to the core",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "L2_LINES_IN",
	  .pme_code = 0x24,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "L2 cache misses",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "L2_M_LINES_IN",
	  .pme_code = 0x25,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "L2 cache line modifications",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "L2_LINES_OUT",
	  .pme_code = 0x26,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "L2 cache lines evicted",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "L2_M_LINES_OUT",
	  .pme_code = 0x27,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Modified lines evicted from the L2 cache",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "L2_IFETCH",
	  .pme_code = 0x28,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "L2 cacheable instruction fetch requests",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 7
	},
	{ .pme_name = "L2_LD",
	  .pme_code = 0x29,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "L2 cache reads",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 9
	},
	{ .pme_name = "L2_ST",
	  .pme_code = 0x2a,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "L2 store requests",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 7
	},
	{ .pme_name = "L2_LOCK",
	  .pme_code = 0x2b,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "L2 locked accesses",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 7
	},
	{ .pme_name = "L2_RQSTS",
	  .pme_code = 0x2e,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "L2 cache requests",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 9
	},
	{ .pme_name = "L2_REJECT_BUSQ",
	  .pme_code = 0x30,
	  .pme_flags = PFMLIB_CORE_CSPEC|PFMLIB_CORE_MESI,
	  .pme_desc =  "Rejected L2 cache requests",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS,
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_HW_PREFETCH_UMASKS
	   },
	   .pme_numasks = 9
	},
	{ .pme_name = "L2_NO_REQ",
	  .pme_code = 0x32,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Cycles no L2 cache requests are pending",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "EIST_TRANS",
	  .pme_code = 0x3a,
	  .pme_flags = 0,
	  .pme_desc =  "Number of Enhanced Intel SpeedStep(R) Technology (EIST) transitions"
	},
	{ .pme_name = "THERMAL_TRIP",
	  .pme_code = 0xc03b,
	  .pme_flags = 0,
	  .pme_desc =  "Number of thermal trips"
	},
	{ .pme_name = "CPU_CLK_UNHALTED",
	  .pme_code = 0x3c,
	  .pme_flags = PFMLIB_CORE_UMASK_NCOMBO,
	  .pme_desc =  "Core cycles when core is not halted",
	  .pme_umasks = {
		{ .pme_uname = "CORE_P",
		  .pme_udesc = "Core cycles when core is not halted",
		  .pme_ucode = 0x0,
		},
		{ .pme_uname = "REF",
		  .pme_udesc = "Reference cycles. This event is not affected by core changes such as P-states or TM2 transitions but counts at the same frequency as the time stamp counter. This event can approximate elapsed time. This event has a constant ratio with the CPU_CLK_UNHALTED:BUS event",
		  .pme_ucode = 0x1,
	  	  .pme_flags = PFMLIB_CORE_FIXED2_ONLY /* Can only be measured on FIXED_CTR2 */
		},
		{ .pme_uname = "BUS",
		  .pme_udesc = "Bus cycles when core is not halted. This event can give a measurement of the elapsed time. This events has a constant ratio with CPU_CLK_UNHALTED:REF event, which is the maximum bus to processor frequency ratio",
		  .pme_ucode = 0x1,
		},
		{ .pme_uname = "NO_OTHER",
		  .pme_udesc = "Bus cycles when core is active and the other is halted",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "L1D_CACHE_LD",
	  .pme_code = 0x40,
	  .pme_flags = PFMLIB_CORE_MESI,
	  .pme_desc =  "L1 cacheable data reads",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "L1D_CACHE_ST",
	  .pme_code = 0x41,
	  .pme_flags = PFMLIB_CORE_MESI,
	  .pme_desc =  "L1 cacheable data writes",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "L1D_CACHE_LOCK",
	  .pme_code = 0x42,
	  .pme_flags = PFMLIB_CORE_MESI,
	  .pme_desc =  "L1 data cacheable locked reads",
	  .pme_umasks = {
		INTEL_CORE_MESI_UMASKS
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "L1D_ALL_REF",
	  .pme_code = 0x143,
	  .pme_flags = 0,
	  .pme_desc =  "All references to the L1 data cache"
	},
	{ .pme_name = "L1D_ALL_CACHE_REF",
	  .pme_code = 0x243,
	  .pme_flags = 0,
	  .pme_desc =  "L1 Data cacheable reads and writes"
	},
	{ .pme_name = "L1D_REPL",
	  .pme_code = 0xf45,
	  .pme_flags = 0,
	  .pme_desc =  "Cache lines allocated in the L1 data cache"
	},
	{ .pme_name = "L1D_M_REPL",
	  .pme_code = 0x46,
	  .pme_flags = 0,
	  .pme_desc =  "Modified cache lines allocated in the L1 data cache"
	},
	{ .pme_name = "L1D_M_EVICT",
	  .pme_code = 0x47,
	  .pme_flags = 0,
	  .pme_desc =  "Modified cache lines evicted from the L1 data cache"
	},
	{ .pme_name = "L1D_PEND_MISS",
	  .pme_code = 0x48,
	  .pme_flags = 0,
	  .pme_desc =  "Total number of outstanding L1 data cache misses at any cycle"
	},
	{ .pme_name = "L1D_SPLIT",
	  .pme_code = 0x49,
	  .pme_flags = 0,
	  .pme_desc =  "Cache line split from L1 data cache",
	  .pme_umasks = {
		{ .pme_uname = "LOADS",
		  .pme_udesc = "Cache line split loads from the L1 data cache",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "STORES",
		  .pme_udesc = "Cache line split stores to the L1 data cache",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "SSE_PRE_MISS",
	  .pme_code = 0x4b,
	  .pme_flags = 0,
	  .pme_desc =  "Streaming SIMD Extensions (SSE) instructions missing all cache levels",
	  .pme_umasks = {
		{ .pme_uname = "NTA",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) Prefetch NTA instructions missing all cache levels",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "L1",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) PrefetchT0 instructions missing all cache levels",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "L2",
		  .pme_udesc = "Streaming SIMD Extensions (SSE) PrefetchT1 and PrefetchT2 instructions missing all cache levels",
		  .pme_ucode = 0x2
		},
	   },
	   .pme_numasks = 3
	},
	{ .pme_name = "LOAD_HIT_PRE",
	  .pme_code = 0x4c,
	  .pme_flags = 0,
	  .pme_desc =  "Load operations conflicting with a software prefetch to the same address"
	},
	{ .pme_name = "L1D_PREFETCH",
	  .pme_code = 0x4e,
	  .pme_flags = 0,
	  .pme_desc =  "L1 data cache prefetch",
	  .pme_umasks = {
		{ .pme_uname = "REQUESTS",
		  .pme_udesc = "L1 data cache prefetch requests",
		  .pme_ucode = 0x10
		}
	   },
	   .pme_numasks = 1
	},
	{ .pme_name = "BUS_REQUEST_OUTSTANDING",
	  .pme_code = 0x60,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Number of pending full cache line read transactions on the bus occurring in each cycle",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_BNR_DRV",
	  .pme_code = 0x61,
	  .pme_flags = 0,
	  .pme_desc =  "Number of Bus Not Ready signals asserted",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BUS_DRDY_CLOCKS",
	  .pme_code = 0x62,
	  .pme_flags = 0,
	  .pme_desc =  "Bus cycles when data is sent on the bus",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BUS_LOCK_CLOCKS",
	  .pme_code = 0x63,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Bus cycles when a LOCK signal is asserted",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_DATA_RCV",
	  .pme_code = 0x64,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Bus cycles while processor receives data",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BUS_TRANS_BRD",
	  .pme_code = 0x65,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Burst read bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_RFO",
	  .pme_code = 0x66,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "RFO bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_WB",
	  .pme_code = 0x67,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Explicit writeback bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_IFETCH",
	  .pme_code = 0x68,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Instruction-fetch bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_INVAL",
	  .pme_code = 0x69,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Invalidate bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_PWR",
	  .pme_code = 0x6a,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Partial write bus transaction",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_P",
	  .pme_code = 0x6b,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Partial bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_IO",
	  .pme_code = 0x6c,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "IO bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_DEF",
	  .pme_code = 0x6d,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Deferred bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_BURST",
	  .pme_code = 0x6e,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Burst (full cache-line) bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_MEM",
	  .pme_code = 0x6f,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Memory bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_TRANS_ANY",
	  .pme_code = 0x70,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "All bus transactions",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "EXT_SNOOP",
	  .pme_code = 0x77,
	  .pme_flags = 0,
	  .pme_desc =  "External snoops responses",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS,
		{ .pme_uname = "ANY",
		  .pme_udesc = "Any external snoop response",
		  .pme_ucode = 0xb
		},
		{ .pme_uname = "CLEAN",
		  .pme_udesc = "External snoop CLEAN response",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "HIT",
		  .pme_udesc = "External snoop HIT response",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "HITM",
		  .pme_udesc = "External snoop HITM response",
		  .pme_ucode = 0x8
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "CMP_SNOOP",
	  .pme_code = 0x78,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "L1 data cache is snooped by other core",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		{ .pme_uname = "ANY",
		  .pme_udesc = "L1 data cache is snooped by other core",
		  .pme_ucode = 0x03
		},
		{ .pme_uname = "SHARE",
		  .pme_udesc = "L1 data cache is snooped for sharing by other core",
		  .pme_ucode = 0x01
		},
		{ .pme_uname = "INVALIDATE",
		  .pme_udesc = "L1 data cache is snooped for Invalidation by other core",
		  .pme_ucode = 0x02
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "BUS_HIT_DRV",
	  .pme_code = 0x7a,
	  .pme_flags = 0,
	  .pme_desc =  "HIT signal asserted",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BUS_HITM_DRV",
	  .pme_code = 0x7b,
	  .pme_flags = 0,
	  .pme_desc =  "HITM signal asserted",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BUSQ_EMPTY",
	  .pme_code = 0x7d,
	  .pme_flags = 0,
	  .pme_desc =  "Bus queue is empty",
	  .pme_umasks = {
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "SNOOP_STALL_DRV",
	  .pme_code = 0x7e,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "Bus stalled for snoops",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS,
		INTEL_CORE_AGENT_UMASKS
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "BUS_IO_WAIT",
	  .pme_code = 0x7f,
	  .pme_flags = PFMLIB_CORE_CSPEC,
	  .pme_desc =  "IO requests waiting in the bus queue",
	  .pme_umasks = {
		INTEL_CORE_SPECIFICITY_UMASKS
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "L1I_READS",
	  .pme_code = 0x80,
	  .pme_flags = 0,
	  .pme_desc =  "Instruction fetches"
	},
	{ .pme_name = "L1I_MISSES",
	  .pme_code = 0x81,
	  .pme_flags = 0,
	  .pme_desc =  "Instruction Fetch Unit misses"
	},
	{ .pme_name = "ITLB",
	  .pme_code = 0x82,
	  .pme_flags = 0,
	  .pme_desc =  "ITLB small page misses",
	  .pme_umasks = {
		{ .pme_uname = "SMALL_MISS",
		  .pme_udesc = "ITLB small page misses",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "LARGE_MISS",
		  .pme_udesc = "ITLB large page misses",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "FLUSH",
		  .pme_udesc = "ITLB flushes",
		  .pme_ucode = 0x40
		},
		{ .pme_uname = "MISSES",
		  .pme_udesc = "ITLB misses",
		  .pme_ucode = 0x12
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "INST_QUEUE",
	  .pme_code = 0x83,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles during which the instruction queue is full",
	  .pme_umasks = {
		{ .pme_uname = "FULL",
		  .pme_udesc = "Cycles during which the instruction queue is full",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 1
	},
	{ .pme_name = "CYCLES_L1I_MEM_STALLED",
	  .pme_code = 0x86,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles during which instruction fetches are stalled"
	},
	{ .pme_name = "ILD_STALL",
	  .pme_code = 0x87,
	  .pme_flags = 0,
	  .pme_desc =  "Instruction Length Decoder stall cycles due to a length changing prefix"
	},
	{ .pme_name = "BR_INST_EXEC",
	  .pme_code = 0x88,
	  .pme_flags = 0,
	  .pme_desc =  "Branch instructions executed"
	},
	{ .pme_name = "BR_MISSP_EXEC",
	  .pme_code = 0x89,
	  .pme_flags = 0,
	  .pme_desc =  "Mispredicted branch instructions executed"
	},
	{ .pme_name = "BR_BAC_MISSP_EXEC",
	  .pme_code = 0x8a,
	  .pme_flags = 0,
	  .pme_desc =  "Branch instructions mispredicted at decoding"
	},
	{ .pme_name = "BR_CND_EXEC",
	  .pme_code = 0x8b,
	  .pme_flags = 0,
	  .pme_desc =  "Conditional branch instructions executed"
	},
	{ .pme_name = "BR_CND_MISSP_EXEC",
	  .pme_code = 0x8c,
	  .pme_flags = 0,
	  .pme_desc =  "Mispredicted conditional branch instructions executed"
	},
	{ .pme_name = "BR_IND_EXEC",
	  .pme_code = 0x8d,
	  .pme_flags = 0,
	  .pme_desc =  "Indirect branch instructions executed"
	},
	{ .pme_name = "BR_IND_MISSP_EXEC",
	  .pme_code = 0x8e,
	  .pme_flags = 0,
	  .pme_desc =  "Mispredicted indirect branch instructions executed"
	},
	{ .pme_name = "BR_RET_EXEC",
	  .pme_code = 0x8f,
	  .pme_flags = 0,
	  .pme_desc =  "RET instructions executed"
	},
	{ .pme_name = "BR_RET_MISSP_EXEC",
	  .pme_code = 0x90,
	  .pme_flags = 0,
	  .pme_desc =  "Mispredicted RET instructions executed"
	},
	{ .pme_name = "BR_RET_BAC_MISSP_EXEC",
	  .pme_code = 0x91,
	  .pme_flags = 0,
	  .pme_desc =  "RET instructions executed mispredicted at decoding"
	},
	{ .pme_name = "BR_CALL_EXEC",
	  .pme_code = 0x92,
	  .pme_flags = 0,
	  .pme_desc =  "CALL instructions executed"
	},
	{ .pme_name = "BR_CALL_MISSP_EXEC",
	  .pme_code = 0x93,
	  .pme_flags = 0,
	  .pme_desc =  "Mispredicted CALL instructions executed"
	},
	{ .pme_name = "BR_IND_CALL_EXEC",
	  .pme_code = 0x94,
	  .pme_flags = 0,
	  .pme_desc =  "Indirect CALL instructions executed"
	},
	{ .pme_name = "BR_TKN_BUBBLE_1",
	  .pme_code = 0x97,
	  .pme_flags = 0,
	  .pme_desc =  "Branch predicted taken with bubble I"
	},
	{ .pme_name = "BR_TKN_BUBBLE_2",
	  .pme_code = 0x98,
	  .pme_flags = 0,
	  .pme_desc =  "Branch predicted taken with bubble II"
	},
#if 0
	/*
	 * Looks like event 0xa1 supersedes this one
	 */
	{ .pme_name = "RS_UOPS_DISPATCHED",
	  .pme_code = 0xa0,
	  .pme_flags = 0,
	  .pme_desc =  "Micro-ops dispatched for execution"
	},
#endif
	{ .pme_name = "MACRO_INSTS",
	  .pme_code = 0xaa,
	  .pme_flags = 0,
	  .pme_desc =  "Instructions decoded",
	  .pme_umasks = {
		{ .pme_uname = "DECODED",
		  .pme_udesc = "Instructions decoded",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "CISC_DECODED",
		  .pme_udesc = "CISC instructions decoded",
		  .pme_ucode = 0x8
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "ESP",
	  .pme_code = 0xab,
	  .pme_flags = 0,
	  .pme_desc =  "ESP register content synchronization",
	  .pme_umasks = {
		{ .pme_uname = "SYNCH",
		  .pme_udesc = "ESP register content synchronization",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "ADDITIONS",
		  .pme_udesc = "ESP register automatic additions",
		  .pme_ucode = 0x2
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "SIMD_UOPS_EXEC",
	  .pme_code = 0xb0,
	  .pme_flags = 0,
	  .pme_desc =  "SIMD micro-ops executed (excluding stores)"
	},
	{ .pme_name = "SIMD_SAT_UOP_EXEC",
	  .pme_code = 0xb1,
	  .pme_flags = 0,
	  .pme_desc =  "SIMD saturated arithmetic micro-ops executed"
	},
	{ .pme_name = "SIMD_UOP_TYPE_EXEC",
	  .pme_code = 0xb3,
	  .pme_flags = 0,
	  .pme_desc =  "SIMD packed multiply micro-ops executed",
	  .pme_umasks = {
		{ .pme_uname = "MUL",
		  .pme_udesc = "SIMD packed multiply micro-ops executed",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SHIFT",
		  .pme_udesc = "SIMD packed shift micro-ops executed",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "PACK",
		  .pme_udesc = "SIMD pack micro-ops executed",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "UNPACK",
		  .pme_udesc = "SIMD unpack micro-ops executed",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "LOGICAL",
		  .pme_udesc = "SIMD packed logical micro-ops executed",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "ARITHMETIC",
		  .pme_udesc = "SIMD packed arithmetic micro-ops executed",
		  .pme_ucode = 0x20
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "INST_RETIRED",
	  .pme_code = 0xc0,
	  .pme_desc =  "Instructions retired",
	  .pme_umasks = {
		{ .pme_uname = "ANY_P",
		  .pme_udesc = "Instructions retired (precise event)",
		  .pme_ucode = 0x0,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		},
		{ .pme_uname = "LOADS",
		  .pme_udesc = "Instructions retired, which contain a load",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "STORES",
		  .pme_udesc = "Instructions retired, which contain a store",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "OTHER",
		  .pme_udesc = "Instructions retired, with no load or store operation",
		  .pme_ucode = 0x4
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "X87_OPS_RETIRED",
	  .pme_code = 0xc1,
	  .pme_flags = 0,
	  .pme_desc =  "FXCH instructions retired",
	  .pme_umasks = {
		{ .pme_uname = "FXCH",
		  .pme_udesc = "FXCH instructions retired",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Retired floating-point computational operations (precise event)",
		  .pme_ucode = 0xfe,
		  .pme_flags = PFMLIB_CORE_PEBS
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "UOPS_RETIRED",
	  .pme_code = 0xc2,
	  .pme_flags = 0,
	  .pme_desc =  "Fused load+op or load+indirect branch retired",
	  .pme_umasks = {
		{ .pme_uname = "LD_IND_BR",
		  .pme_udesc = "Fused load+op or load+indirect branch retired",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "STD_STA",
		  .pme_udesc = "Fused store address + data retired",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "MACRO_FUSION",
		  .pme_udesc = "Retired instruction pairs fused into one micro-op",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "NON_FUSED",
		  .pme_udesc = "Non-fused micro-ops retired",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "FUSED",
		  .pme_udesc = "Fused micro-ops retired",
		  .pme_ucode = 0x7
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Micro-ops retired",
		  .pme_ucode = 0xf
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "MACHINE_NUKES",
	  .pme_code = 0xc3,
	  .pme_flags = 0,
	  .pme_desc =  "Self-Modifying Code detected",
	  .pme_umasks = {
		{ .pme_uname = "SMC",
		  .pme_udesc = "Self-Modifying Code detected",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "MEM_ORDER",
		  .pme_udesc = "Execution pipeline restart due to memory ordering conflict or memory disambiguation misprediction",
		  .pme_ucode = 0x4
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "BR_INST_RETIRED",
	  .pme_code = 0xc4,
	  .pme_flags = 0,
	  .pme_desc =  "Retired branch instructions",
	  .pme_umasks = {
		{ .pme_uname = "ANY",
		  .pme_udesc = "Retired branch instructions",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "PRED_NOT_TAKEN",
		  .pme_udesc = "Retired branch instructions that were predicted not-taken",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "MISPRED_NOT_TAKEN",
		  .pme_udesc = "Retired branch instructions that were mispredicted not-taken",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "PRED_TAKEN",
		  .pme_udesc = "Retired branch instructions that were predicted taken",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "MISPRED_TAKEN",
		  .pme_udesc = "Retired branch instructions that were mispredicted taken",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "TAKEN",
		  .pme_udesc = "Retired taken branch instructions",
		  .pme_ucode = 0xc
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "BR_INST_RETIRED_MISPRED",
	  .pme_code = 0x00c5,
	  .pme_desc =  "Retired mispredicted branch instructions (precise_event)",
	  .pme_flags = PFMLIB_CORE_PEBS
	},
	{ .pme_name = "CYCLES_INT_MASKED",
	  .pme_code = 0x1c6,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles during which interrupts are disabled"
	},
	{ .pme_name = "CYCLES_INT_PENDING_AND_MASKED",
	  .pme_code = 0x2c6,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles during which interrupts are pending and disabled"
	},
	{ .pme_name = "SIMD_INST_RETIRED",
	  .pme_code = 0xc7,
	  .pme_flags = 0,
	  .pme_desc =  "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
	  .pme_umasks = {
		{ .pme_uname = "PACKED_SINGLE",
		  .pme_udesc = "Retired Streaming SIMD Extensions (SSE) packed-single instructions",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SCALAR_SINGLE",
		  .pme_udesc = "Retired Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "PACKED_DOUBLE",
		  .pme_udesc = "Retired Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "SCALAR_DOUBLE",
		  .pme_udesc = "Retired Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "VECTOR",
		  .pme_udesc = "Retired Streaming SIMD Extensions 2 (SSE2) vector integer instructions",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Retired Streaming SIMD instructions (precise event)",
		  .pme_ucode = 0x1f,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "HW_INT_RCV",
	  .pme_code = 0xc8,
	  .pme_desc =  "Hardware interrupts received"
	},
	{ .pme_name = "ITLB_MISS_RETIRED",
	  .pme_code = 0xc9,
	  .pme_flags = 0,
	  .pme_desc =  "Retired instructions that missed the ITLB"
	},
	{ .pme_name = "SIMD_COMP_INST_RETIRED",
	  .pme_code = 0xca,
	  .pme_flags = 0,
	  .pme_desc =  "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
	  .pme_umasks = {
		{ .pme_uname = "PACKED_SINGLE",
		  .pme_udesc = "Retired computational Streaming SIMD Extensions (SSE) packed-single instructions",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SCALAR_SINGLE",
		  .pme_udesc = "Retired computational Streaming SIMD Extensions (SSE) scalar-single instructions",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "PACKED_DOUBLE",
		  .pme_udesc = "Retired computational Streaming SIMD Extensions 2 (SSE2) packed-double instructions",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "SCALAR_DOUBLE",
		  .pme_udesc = "Retired computational Streaming SIMD Extensions 2 (SSE2) scalar-double instructions",
		  .pme_ucode = 0x8
		}
	   },
	   .pme_numasks = 4
	},
	{ .pme_name = "MEM_LOAD_RETIRED",
	  .pme_code = 0xcb,
	  .pme_desc =  "Retired loads that miss the L1 data cache",
	  .pme_flags = PFMLIB_CORE_PMC0, 
	  .pme_umasks = {
		{ .pme_uname = "L1D_MISS",
		  .pme_udesc = "Retired loads that miss the L1 data cache (precise event)",
		  .pme_ucode = 0x1,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		},
		{ .pme_uname = "L1D_LINE_MISS",
		  .pme_udesc = "L1 data cache line missed by retired loads (precise event)",
		  .pme_ucode = 0x2,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		},
		{ .pme_uname = "L2_MISS",
		  .pme_udesc = "Retired loads that miss the L2 cache (precise event)",
		  .pme_ucode = 0x4,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		},
		{ .pme_uname = "L2_LINE_MISS",
		  .pme_udesc = "L2 cache line missed by retired loads (precise event)",
		  .pme_ucode = 0x8,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		},
		{ .pme_uname = "DTLB_MISS",
		  .pme_udesc = "Retired loads that miss the DTLB (precise event)",
		  .pme_ucode = 0x10,
	  	  .pme_flags = PFMLIB_CORE_PEBS
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "FP_MMX_TRANS",
	  .pme_code = 0xcc,
	  .pme_flags = PFMLIB_CORE_PEBS,
	  .pme_desc =  "Transitions from MMX (TM) Instructions to Floating Point Instructions",
	  .pme_umasks = {
		{ .pme_uname = "TO_FP",
		  .pme_udesc = "Transitions from MMX (TM) Instructions to Floating Point Instructions",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "TO_MMX",
		  .pme_udesc = "Transitions from Floating Point to MMX (TM) Instructions",
		  .pme_ucode = 0x1
		}
	   },
	   .pme_numasks = 2
	},
	{ .pme_name = "SIMD_ASSIST",
	  .pme_code = 0xcd,
	  .pme_flags = 0,
	  .pme_desc =  "SIMD assists invoked"
	},
	{ .pme_name = "SIMD_INSTR_RETIRED",
	  .pme_code = 0xce,
	  .pme_flags = 0,
	  .pme_desc =  "SIMD Instructions retired"
	},
	{ .pme_name = "SIMD_SAT_INSTR_RETIRED",
	  .pme_code = 0xcf,
	  .pme_flags = 0,
	  .pme_desc =  "Saturated arithmetic instructions retired"
	},
	{ .pme_name = "RAT_STALLS",
	  .pme_code = 0xd2,
	  .pme_flags = 0,
	  .pme_desc =  "ROB read port stalls cycles",
	  .pme_umasks = {
		{ .pme_uname = "ROB_READ_PORT",
		  .pme_udesc = "ROB read port stalls cycles",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "PARTIAL_CYCLES",
		  .pme_udesc = "Partial register stall cycles",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "FLAGS",
		  .pme_udesc = "Flag stall cycles",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "FPSW",
		  .pme_udesc = "FPU status word stall",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "All RAT stall cycles",
		  .pme_ucode = 0xf
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "SEG_RENAME_STALLS",
	  .pme_code = 0xd4,
	  .pme_flags = 0,
	  .pme_desc =  "Segment rename stalls - ES ",
	  .pme_umasks = {
		{ .pme_uname = "ES",
		  .pme_udesc = "Segment rename stalls - ES ",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "DS",
		  .pme_udesc = "Segment rename stalls - DS",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "FS",
		  .pme_udesc = "Segment rename stalls - FS",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "GS",
		  .pme_udesc = "Segment rename stalls - GS",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Any (ES/DS/FS/GS) segment rename stall",
		  .pme_ucode = 0xf
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "SEG_REG_RENAMES",
	  .pme_code = 0xd5,
	  .pme_flags = 0,
	  .pme_desc =  "Segment renames - ES",
	  .pme_umasks = {
		{ .pme_uname = "ES",
		  .pme_udesc = "Segment renames - ES",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "DS",
		  .pme_udesc = "Segment renames - DS",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "FS",
		  .pme_udesc = "Segment renames - FS",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "GS",
		  .pme_udesc = "Segment renames - GS",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Any (ES/DS/FS/GS) segment rename",
		  .pme_ucode = 0xf
		}
	   },
	   .pme_numasks = 5
	},
	{ .pme_name = "RESOURCE_STALLS",
	  .pme_code = 0xdc,
	  .pme_flags = 0,
	  .pme_desc =  "Cycles during which the ROB is full",
	  .pme_umasks = {
		{ .pme_uname = "ROB_FULL",
		  .pme_udesc = "Cycles during which the ROB is full",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "RS_FULL",
		  .pme_udesc = "Cycles during which the RS is full",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "LD_ST",
		  .pme_udesc = "Cycles during which the pipeline has exceeded load or store limit or waiting to commit all stores",
		  .pme_ucode = 0x4
		},
		{ .pme_uname = "FPCW",
		  .pme_udesc = "Cycles stalled due to FPU control word write",
		  .pme_ucode = 0x8
		},
		{ .pme_uname = "BR_MISS_CLEAR",
		  .pme_udesc = "Cycles stalled due to branch misprediction",
		  .pme_ucode = 0x10
		},
		{ .pme_uname = "ANY",
		  .pme_udesc = "Resource related stalls",
		  .pme_ucode = 0x1f
		}
	   },
	   .pme_numasks = 6
	},
	{ .pme_name = "BR_INST_DECODED",
	  .pme_code = 0xe0,
	  .pme_flags = 0,
	  .pme_desc =  "Branch instructions decoded"
	},
	{ .pme_name = "BOGUS_BR",
	  .pme_code = 0xe4,
	  .pme_flags = 0,
	  .pme_desc =  "Bogus branches"
	},
	{ .pme_name = "BACLEARS",
	  .pme_code = 0xe6,
	  .pme_flags = 0,
	  .pme_desc =  "BACLEARS asserted"
	},
	{ .pme_name = "PREF_RQSTS_UP",
	  .pme_code = 0xf0,
	  .pme_flags = 0,
	  .pme_desc =  "Upward prefetches issued from the DPL"
	},
	{ .pme_name = "PREF_RQSTS_DN",
	  .pme_code = 0xf8,
	  .pme_flags = 0,
	  .pme_desc =  "Downward prefetches issued from the DPL"
	}
};
#define PME_CORE_UNHALTED_CORE_CYCLES 0
#define PME_CORE_INSTRUCTIONS_RETIRED 1
#define PME_CORE_EVENT_COUNT	  (sizeof(core_pe)/sizeof(pme_core_entry_t))
