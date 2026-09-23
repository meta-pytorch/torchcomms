/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 * Contributions by James Ralph <ralph@eecs.utk.edu>
 *
 * Based on:
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


#define INTEL_COREDUO_MESI_UMASKS \
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

#define INTEL_COREDUO_SPECIFICITY_UMASKS \
	{ .pme_uname = "SELF",\
		  .pme_udesc = "This core",\
		  .pme_ucode = 0x40\
		},\
	{ .pme_uname = "BOTH_CORES",\
		  .pme_udesc = "Both cores",\
		  .pme_ucode = 0xc0\
		}

#define INTEL_COREDUO_HW_PREFETCH_UMASKS \
	{ .pme_uname = "ANY",\
		  .pme_udesc = "All inclusive",\
		  .pme_ucode = 0x30\
		},\
	{ .pme_uname = "PREFETCH",\
		  .pme_udesc = "Hardware prefetch only",\
		  .pme_ucode = 0x10\
		}

#define INTEL_COREDUO_AGENT_UMASKS \
	{ .pme_uname = "THIS_AGENT",\
		  .pme_udesc = "This agent",\
		  .pme_ucode = 0x00\
		},\
	{ .pme_uname = "ALL_AGENTS",\
		  .pme_udesc = "Any agent on the bus",\
		  .pme_ucode = 0x20\
		}

static pme_coreduo_entry_t coreduo_pe[]={
  /*
   * BEGIN architectural perfmon events
   */
  /* 0 */{
	.pme_name = "UNHALTED_CORE_CYCLES",
	.pme_code    = 0x003c,
	.pme_desc    = "Unhalted core cycles",
  },
  /* 1 */{
	.pme_name = "UNHALTED_REFERENCE_CYCLES",
	.pme_code = 0x013c,
	.pme_desc = "Unhalted reference cycles. Measures bus cycles"
  },
  /* 2 */{
	.pme_name = "INSTRUCTIONS_RETIRED",
	.pme_code = 0xc0,
	.pme_desc = "Instructions retired"
  },
  /* 3 */{
	.pme_name = "LAST_LEVEL_CACHE_REFERENCES",
	.pme_code = 0x4f2e,
	.pme_desc = "Last level of cache references"
  },
  /* 4 */{
	.pme_name = "LAST_LEVEL_CACHE_MISSES",
	.pme_code = 0x412e,
	.pme_desc = "Last level of cache misses",
  },
  /* 5  */{
	.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	.pme_code = 0xc4,
	.pme_desc = "Branch instructions retired"
  },
  /* 6  */{
	.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	.pme_code = 0xc5,
	.pme_desc = "Mispredicted branch instruction retired"
  },

  /*
   * BEGIN non architectural events
   */

  { .pme_code = 0x3,
	.pme_name = "LD_BLOCKS",
	.pme_desc = "Load operations delayed due to store buffer blocks. The preceding store may be blocked due to unknown address, unknown data, or conflict due to partial overlap between the load and store.",
  },
  { .pme_code = 0x4,
	.pme_name = "SD_DRAINS",
	.pme_desc = "Cycles while draining store buffers",
  },
  { .pme_code = 0x5,
	.pme_name = "MISALIGN_MEM_REF",
	.pme_desc = "Misaligned data memory references (MOB splits of loads and stores).",
  },
  { .pme_code = 0x6,
	.pme_name = "SEG_REG_LOADS",
	.pme_desc = "Segment register loads",
  },
  { .pme_code = 0x7,
	.pme_name = "SSE_PREFETCH",
	.pme_flags = 0,
	.pme_desc = "Streaming SIMD Extensions (SSE) Prefetch instructions executed",
	.pme_umasks = {
	{ .pme_uname = "NTA",
		.pme_udesc =  "Streaming SIMD Extensions (SSE) Prefetch NTA instructions executed",
		.pme_ucode = 0x0
	  },
	{ .pme_uname = "T1",
		.pme_udesc = "SSE software prefetch instruction PREFE0xTCT1 retired",
		.pme_ucode = 0x01
	  },
	{ .pme_uname = "T2",
		.pme_udesc = "SSE software prefetch instruction PREFE0xTCT2 retired",
		.pme_ucode = 0x02
	  },
	},
	.pme_numasks = 3
  },
  { .pme_name = "SSE_NTSTORES_RET",
    .pme_desc = "SSE streaming store instruction retired",
    .pme_code = 0x0307
  },
  { .pme_code = 0x10,
	.pme_name = "FP_COMPS_OP_EXE",
	.pme_desc = "FP computational Instruction executed. FADD, FSUB, FCOM, FMULs, MUL, IMUL, FDIVs, DIV, IDIV, FPREMs, FSQRT are included; but exclude FADD or FMUL used in the middle of a transcendental instruction.",
  },
  { .pme_code = 0x11,
	.pme_name = "FP_ASSIST",
	.pme_desc = "FP exceptions experienced microcode assists",
	.pme_flags = PFMLIB_COREDUO_PMC1
  },
  { .pme_code = 0x12,
	.pme_name = "MUL",
	.pme_desc = "Multiply operations (a speculative count, including FP and integer multiplies).",
	.pme_flags = PFMLIB_COREDUO_PMC1
  },
  { .pme_code = 0x13,
	.pme_name = "DIV",
	.pme_desc = "Divide operations (a speculative count, including FP and integer multiplies). ",
	.pme_flags = PFMLIB_COREDUO_PMC1   
  },
  { .pme_code = 0x14,
	.pme_name = "CYCLES_DIV_BUSY",
	.pme_desc = "Cycles the divider is busy ",
	.pme_flags = PFMLIB_COREDUO_PMC0 
  },
  { .pme_code = 0x21,
	.pme_name = "L2_ADS",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "L2 Address strobes ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x22,
	.pme_name = "DBUS_BUSY",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Core cycle during which data buswas busy (increments by 4)",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x23,
	.pme_name = "DBUS_BUSY_RD",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Cycles data bus is busy transferring data to a core (increments by 4) ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x24,
	.pme_name = "L2_LINES_IN",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "L2 cache lines allocated",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x25,
	.pme_name = "L2_M_LINES_IN",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "L2 Modified-state cache lines allocated",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x26,
	.pme_name = "L2_LINES_OUT",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "L2 cache lines evicted ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x27,
	.pme_name = "L2_M_LINES_OUT",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "L2 Modified-state cache lines evicted ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x28,
	.pme_name = "L2_IFETCH",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_desc = "L2 instruction fetches from nstruction fetch unit (includes speculative fetches) ",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 7
  },
  { .pme_code = 0x29,
	.pme_name = "L2_LD",
	.pme_desc = "L2 cache reads (includes speculation) ",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 7
  },
  { .pme_code = 0x2A,
	.pme_name = "L2_ST",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_desc = "L2 cache writes (includes speculation)",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 7
  },
  { .pme_code = 0x2E,
	.pme_name = "L2_RQSTS",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_desc = "L2 cache reference requests ",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 9
  },
  { .pme_code = 0x30,
	.pme_name = "L2_REJECT_CYCLES",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_desc = "Cycles L2 is busy and rejecting new requests.",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 9
  },
  { .pme_code = 0x32,
	.pme_name = "L2_NO_REQUEST_CYCLES",
	.pme_flags = PFMLIB_COREDUO_CSPEC|PFMLIB_COREDUO_MESI,
	.pme_desc = "Cycles there is no request to access L2.",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_HW_PREFETCH_UMASKS
	},
	.pme_numasks = 9
  },
  { .pme_code = 0x3A,
	.pme_name = "EST_TRANS_ALL",
	.pme_desc = "Any Intel Enhanced SpeedStep(R) Technology transitions",
  },
  { .pme_code = 0x103A,
	.pme_name = "EST_TRANS_ALL",
	.pme_desc = "Intel Enhanced SpeedStep Technology frequency transitions",
  },
  { .pme_code = 0x3B,
	.pme_name = "THERMAL_TRIP",
	.pme_desc = "Duration in a thermal trip based on the current core clock ",
	.pme_umasks = {
	{ .pme_uname = "CYCLES",
		.pme_udesc = "Duration in a thermal trip based on the current core clock",
		.pme_ucode = 0xC0
	  },
	{ .pme_uname = "TRIPS",
		.pme_udesc = "Number of thermal trips",
		.pme_ucode = 0xC0 | (1<<10) /* Edge detect pin (Figure 18-13) */
	  }
	},
	.pme_numasks = 2 
  },
  {
	.pme_name = "CPU_CLK_UNHALTED",
	.pme_code = 0x3c,
	.pme_desc = "Core cycles when core is not halted",
	.pme_umasks = {
	{ .pme_uname = "NONHLT_REF_CYCLES",
		.pme_udesc = "Non-halted bus cycles",
		.pme_ucode = 0x01
	  },
	{ .pme_uname = "SERIAL_EXECUTION_CYCLES",
		.pme_udesc ="Non-halted bus cycles of this core executing code while the other core is halted",
		.pme_ucode = 0x02
	  }
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x40,
	.pme_name = "DCACHE_CACHE_LD",
	.pme_desc = "L1 cacheable data read operations",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS
	},
	.pme_numasks = 5
  },
  { .pme_code = 0x41,
	.pme_name = "DCACHE_CACHE_ST",
	.pme_desc = "L1 cacheable data write operations",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS
	},
	.pme_numasks = 5
  },
  { .pme_code = 0x42,
	.pme_name = "DCACHE_CACHE_LOCK",
	.pme_desc = "L1 cacheable lock read operations to invalid state",
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS
	},
	.pme_numasks = 5
  },
  { .pme_code = 0x0143,
	.pme_name = "DATA_MEM_REF",
	.pme_desc = "L1 data read and writes of cacheable and non-cacheable types",
  },
  { .pme_code = 0x0244,
	.pme_name = "DATA_MEM_CACHE_REF",
	.pme_desc = "L1 data cacheable read and write operations.",
  },
  { .pme_code = 0x0f45,
	.pme_name = "DCACHE_REPL",
	.pme_desc = "L1 data cache line replacements",
  },
  { .pme_code = 0x46,
	.pme_name = "DCACHE_M_REPL",
	.pme_desc = "L1 data M-state cache line  allocated",
  },
  { .pme_code = 0x47,
	.pme_name = "DCACHE_M_EVICT",
	.pme_desc = "L1 data M-state cache line evicted",
  },
  { .pme_code = 0x48,
	.pme_name = "DCACHE_PEND_MISS",
	.pme_desc = "Weighted cycles of L1 miss outstanding",
  },
  { .pme_code = 0x49,
	.pme_name = "DTLB_MISS",
	.pme_desc = "Data references that missed TLB",
  },
  { .pme_code = 0x4B,
	.pme_name = "SSE_PRE_MISS",
	.pme_flags = 0,
	.pme_desc = "Streaming SIMD Extensions (SSE) instructions missing all cache levels",
	.pme_umasks = {
	{ .pme_uname = "NTA_MISS",
		.pme_udesc = "PREFETCHNTA missed all caches",
		.pme_ucode = 0x00
	  },
	{ .pme_uname = "T1_MISS",
		.pme_udesc = "PREFETCHT1 missed all caches",
		.pme_ucode = 0x01
	  },
	{ .pme_uname = "T2_MISS",
		.pme_udesc = "PREFETCHT2 missed all caches",
		.pme_ucode = 0x02
	  },
	{ .pme_uname = "STORES_MISS",
		.pme_udesc = "SSE streaming store instruction missed all caches",
		.pme_ucode = 0x03
	  }
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x4F,
	.pme_name = "L1_PREF_REQ",
	.pme_desc = "L1 prefetch requests due to DCU cache misses",
  },
  { .pme_code = 0x60,
	.pme_name = "BUS_REQ_OUTSTANDING",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Weighted cycles of cacheable bus data read requests. This event counts full-line read request from DCU or HW prefetcher, but not RFO, write, instruction fetches, or others.",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
	/* TODO: umasks bit 12 to include HWP or exclude HWP separately. */,
  },
  { .pme_code = 0x61,
	.pme_name = "BUS_BNR_CLOCKS",
	.pme_desc = "External bus cycles while BNR asserted",
  },
  { .pme_code = 0x62,
	.pme_name = "BUS_DRDY_CLOCKS",
	.pme_desc = "External bus cycles while DRDY asserted",
	.pme_umasks = {
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x63,
	.pme_name = "BUS_LOCKS_CLOCKS",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "External bus cycles while bus lock signal asserted",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x4064,
	.pme_name = "BUS_DATA_RCV",
	.pme_desc = "External bus cycles while bus lock signal asserted",
  },
  { .pme_code = 0x65,
	.pme_name = "BUS_TRANS_BRD",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Burst read bus transactions (data or code)",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x66,
	.pme_name = "BUS_TRANS_RFO",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed read for ownership ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x68,
	.pme_name = "BUS_TRANS_IFETCH",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed instruction fetch transactions",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
 
  },
  { .pme_code = 0x69,
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_name = "BUS_TRANS_INVAL",
	.pme_desc = "Completed invalidate transactions",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x6A,
	.pme_name = "BUS_TRANS_PWR",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed partial write transactions",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x6B,
	.pme_name = "BUS_TRANS_P",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed partial transactions (include partial read + partial write + line write)",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x6C,
	.pme_name = "BUS_TRANS_IO",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed I/O transactions (read and write)",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 4
  },
  { .pme_code = 0x206D,
	.pme_name = "BUS_TRANS_DEF",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed defer transactions ",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0xc067,
	.pme_name = "BUS_TRANS_WB",
	.pme_desc = "Completed writeback transactions from DCU (does not include L2 writebacks)",
	.pme_umasks = {
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0xc06E,
	.pme_name = "BUS_TRANS_BURST",
	.pme_desc = "Completed burst transactions (full line transactions include reads, write, RFO, and writebacks) ",
	/* TODO .pme_umasks = 0xC0, */
	.pme_umasks = {
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0xc06F,
	.pme_name = "BUS_TRANS_MEM",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Completed memory transactions. This includes Bus_Trans_Burst + Bus_Trans_P + Bus_Trans_Inval.",
	.pme_umasks = {
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0xc070,
	.pme_name = "BUS_TRANS_ANY",
	.pme_desc = "Any completed bus transactions",
	.pme_umasks = {
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x77,
	.pme_name = "BUS_SNOOPS",
	.pme_desc = "External bus cycles while bus lock signal asserted",
	.pme_flags = PFMLIB_COREDUO_MESI,
	.pme_umasks = {
	  INTEL_COREDUO_MESI_UMASKS,
	  INTEL_COREDUO_AGENT_UMASKS
	},
	.pme_numasks = 7
  },
  { .pme_code = 0x0178,
	.pme_name = "DCU_SNOOP_TO_SHARE",
	.pme_desc = "DCU snoops to share-state L1 cache line due to L1 misses ",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x7D,
	.pme_name = "BUS_NOT_IN_USE",
	.pme_flags = PFMLIB_COREDUO_CSPEC,
	.pme_desc = "Number of cycles there is no transaction from the core",
	.pme_umasks = {
	  INTEL_COREDUO_SPECIFICITY_UMASKS
	},
	.pme_numasks = 2
  },
  { .pme_code = 0x7E,
	.pme_name = "BUS_SNOOP_STALL",
	.pme_desc = "Number of bus cycles while bus snoop is stalled"
  },
  { .pme_code = 0x80,
	.pme_name = "ICACHE_READS",
	.pme_desc = "Number of instruction fetches from ICache, streaming buffers (both cacheable and uncacheable fetches)"
  },
  { .pme_code = 0x81,
	.pme_name = "ICACHE_MISSES",
	.pme_desc = "Number of instruction fetch misses from ICache, streaming buffers."
  },
  { .pme_code = 0x85,
	.pme_name = "ITLB_MISSES",
	.pme_desc = "Number of iITLB misses"
  },
  { .pme_code = 0x86,
	.pme_name = "IFU_MEM_STALL",
	.pme_desc = "Cycles IFU is stalled while waiting for data from memory"
  },
  { .pme_code = 0x87,
	.pme_name = "ILD_STALL",
	.pme_desc = "Number of instruction length decoder stalls (Counts number of LCP stalls)"
  },
  { .pme_code = 0x88,
	.pme_name = "BR_INST_EXEC",
	.pme_desc = "Branch instruction executed (includes speculation)."
  },
  { .pme_code = 0x89,
	.pme_name = "BR_MISSP_EXEC",
	.pme_desc = "Branch instructions executed and mispredicted at execution  (includes branches that do not have prediction or mispredicted)"
  },
  { .pme_code = 0x8A,
	.pme_name = "BR_BAC_MISSP_EXEC",
	.pme_desc = "Branch instructions executed that were mispredicted at front end"
  },
  { .pme_code = 0x8B,
	.pme_name = "BR_CND_EXEC",
	.pme_desc = "Conditional branch instructions executed"
  },
  { .pme_code = 0x8C,
	.pme_name = "BR_CND_MISSP_EXEC",
	.pme_desc = "Conditional branch instructions executed that were mispredicted"
  },
  { .pme_code = 0x8D,
	.pme_name = "BR_IND_EXEC",
	.pme_desc = "Indirect branch instructions executed"
  },
  { .pme_code = 0x8E,
	.pme_name = "BR_IND_MISSP_EXEC",
	.pme_desc = "Indirect branch instructions executed that were mispredicted"
  },
  { .pme_code = 0x8F,
	.pme_name = "BR_RET_EXEC",
	.pme_desc = "Return branch instructions executed"
  },
  { .pme_code = 0x90,
	.pme_name = "BR_RET_MISSP_EXEC",
	.pme_desc = "Return branch instructions executed that were mispredicted"
  },
  { .pme_code = 0x91,
	.pme_name = "BR_RET_BAC_MISSP_EXEC",
	.pme_desc = "Return branch instructions executed that were mispredicted at the front end"
  },
  { .pme_code = 0x92,
	.pme_name = "BR_CALL_EXEC",
	.pme_desc = "Return call instructions executed"
  },
  { .pme_code = 0x93,
	.pme_name = "BR_CALL_MISSP_EXEC",
	.pme_desc = "Return call instructions executed that were mispredicted"
  },
  { .pme_code = 0x94,
	.pme_name = "BR_IND_CALL_EXEC",
	.pme_desc = "Indirect call branch instructions executed"
  },
  { .pme_code = 0xA2,
	.pme_name = "RESOURCE_STALL",
	.pme_desc = "Cycles while there is a resource related stall (renaming, buffer entries) as seen by allocator"
  },
  { .pme_code = 0xB0,
	.pme_name = "MMX_INSTR_EXEC",
	.pme_desc = "Number of MMX instructions executed (does not include MOVQ and MOVD stores)"
  },
  { .pme_code = 0xB1,
	.pme_name = "SIMD_INT_SAT_EXEC",
	.pme_desc = "Number of SIMD Integer saturating instructions executed"
  },
  { .pme_code = 0xB3,
	.pme_name = "SIMD_INT_INSTRUCTIONS",
	.pme_desc = "Number of SIMD Integer instructions executed",
	.pme_umasks = {
	{ .pme_uname = "MUL",
		.pme_udesc = "Number of SIMD Integer packed multiply instructions executed",
		.pme_ucode = 0x01
	  },
	{ .pme_uname = "SHIFT",
		.pme_udesc = "Number of SIMD Integer packed shift instructions executed",
		.pme_ucode = 0x02
	  },
	{ .pme_uname = "PACK",
		.pme_udesc = "Number of SIMD Integer pack operations instruction executed",
		.pme_ucode = 0x04
	  },
	{ .pme_uname = "UNPACK",
		.pme_udesc = "Number of SIMD Integer unpack instructions executed",
		.pme_ucode = 0x08
	  },
	{ .pme_uname = "LOGICAL",
		.pme_udesc = "Number of SIMD Integer packed logical instructions executed",
		.pme_ucode = 0x10
	  },
	{ .pme_uname = "ARITHMETIC",
		.pme_udesc = "Number of SIMD Integer packed arithmetic instructions executed",
		.pme_ucode = 0x20
	  }
	},
	.pme_numasks = 6
  },
  { .pme_code = 0xC0,
	.pme_name = "INSTR_RET",
	.pme_desc = "Number of instruction retired (Macro fused instruction count as 2)"
  },
  { .pme_code = 0xC1,
	.pme_name = "FP_COMP_INSTR_RET",
	.pme_desc = "Number of FP compute instructions retired (X87 instruction or instruction that contain X87 operations)",
	.pme_flags = PFMLIB_COREDUO_PMC0
  },
  { .pme_code = 0xC2,
	.pme_name = "UOPS_RET",
	.pme_desc = "Number of micro-ops retired (include fused uops)"
  },
  { .pme_code = 0xC3,
	.pme_name = "SMC_DETECTED",
	.pme_desc = "Number of times self-modifying code condition detected"
  },
  { .pme_code = 0xC4,
	.pme_name = "BR_INSTR_RET",
	.pme_desc = "Number of branch instructions retired"
  },
  { .pme_code = 0xC5,
	.pme_name = "BR_MISPRED_RET",
	.pme_desc = "Number of mispredicted branch instructions retired"
  },
  { .pme_code = 0xC6,
	.pme_name = "CYCLES_INT_MASKED",
	.pme_desc = "Cycles while interrupt is disabled"
  },
  { .pme_code = 0xC7,
	.pme_name = "CYCLES_INT_PEDNING_MASKED",
	.pme_desc = "Cycles while interrupt is disabled and interrupts are pending"
  },
  { .pme_code = 0xC8,
	.pme_name = "HW_INT_RX",
	.pme_desc = "Number of hardware interrupts received"
  },
  { .pme_code = 0xC9,
	.pme_name = "BR_TAKEN_RET",
	.pme_desc = "Number of taken branch instruction retired"
  },
  { .pme_code = 0xCA,
	.pme_name = "BR_MISPRED_TAKEN_RET",
	.pme_desc = "Number of taken and mispredicted branch instructions retired"
  },
  { .pme_code = 0xCC,
	.pme_name = "FP_MMX_TRANS",
	.pme_name = "MMX_FP_TRANS",
	.pme_desc = "Transitions from MMX (TM) Instructions to Floating Point Instructions",
	.pme_umasks = {
	{ .pme_uname = "TO_FP",
		.pme_udesc = "Number of transitions from MMX to X87",
		.pme_ucode = 0x00
	  },
	{ .pme_uname = "TO_MMX",
		.pme_udesc = "Number of transitions from X87 to MMX",
		.pme_ucode = 0x01
	  }
	},
	.pme_numasks = 2
  },
  { .pme_code = 0xCD,
	.pme_name = "MMX_ASSIST",
	.pme_desc = "Number of EMMS executed"
  },
  { .pme_code = 0xCE,
	.pme_name = "MMX_INSTR_RET",
	.pme_desc = "Number of MMX instruction retired"
  },
  { .pme_code = 0xD0,
	.pme_name = "INSTR_DECODED",
	.pme_desc = "Number of instruction decoded"
  },
  { .pme_code = 0xD7,
	.pme_name = "ESP_UOPS",
	.pme_desc = "Number of ESP folding instruction decoded"
  },
  { .pme_code = 0xD8,
	.pme_name = "SSE_INSTRUCTIONS_RETIRED",
	.pme_desc = "Number of SSE/SSE2 instructions retired (packed and scalar)",
	.pme_umasks = {
	{ .pme_uname = "SINGLE",
		.pme_udesc = "Number of SSE/SSE2 single precision instructions retired (packed and scalar)",
		.pme_ucode = 0x00
	  },
	{ .pme_uname = "SCALAR_SINGLE",
		.pme_udesc = "Number of SSE/SSE2 scalar single precision instructions retired",
		.pme_ucode = 0x01,
	  },
	{ .pme_uname = "PACKED_DOUBLE",
		.pme_udesc = "Number of SSE/SSE2 packed double percision instructions retired",
		.pme_ucode = 0x02,
	  },
	{ .pme_uname = "DOUBLE",
		.pme_udesc = "Number of SSE/SSE2 scalar double percision instructions retired",
		.pme_ucode = 0x03,
	  },
	{ .pme_uname = "INT_128",
		.pme_udesc = "Number of SSE2 128 bit integer  instructions retired",
		.pme_ucode = 0x04,
	 },
	},
	.pme_numasks = 5
  },
  { .pme_code = 0xD9,
	.pme_name = "SSE_COMP_INSTRUCTIONS_RETIRED",
	.pme_desc = "Number of computational SSE/SSE2 instructions retired (does not include AND, OR, XOR)",
	.pme_umasks = {
	{ .pme_uname = "PACKED_SINGLE",
		.pme_udesc = "Number of SSE/SSE2 packed single precision compute instructions retired (does not include AND, OR, XOR)",
		.pme_ucode = 0x00
	  },
	{ .pme_uname = "SCALAR_SINGLE",
		.pme_udesc = "Number of SSE/SSE2 scalar single precision compute instructions retired (does not include AND, OR, XOR)",
		.pme_ucode = 0x01
	  },
	{ .pme_uname = "PACKED_DOUBLE",
		.pme_udesc = "Number of SSE/SSE2 packed double precision compute instructions retired (does not include AND, OR, XOR)",
		.pme_ucode = 0x02
	  },
	{ .pme_uname = "SCALAR_DOUBLE",
		.pme_udesc = "Number of SSE/SSE2 scalar double precision compute instructions retired (does not include AND, OR, XOR)",
		.pme_ucode = 0x03
	  }
	},
	.pme_numasks = 4
  },
  { .pme_code = 0xDA,
	.pme_name = "FUSED_UOPS",
	.pme_desc = "fused uops retired",
	.pme_umasks = {
  	{ .pme_uname = "ALL",
		.pme_udesc = "All fused uops retired",
		.pme_ucode = 0x00
	  },
  	{ .pme_uname = "LOADS",
		.pme_udesc = "Fused load uops retired",
		.pme_ucode = 0x01
	},
  	{ .pme_uname = "STORES",
		.pme_udesc = "Fused load uops retired",
		.pme_ucode = 0x02
	 },
	},
	.pme_numasks = 3
  },
  { .pme_code = 0xDB,
	.pme_name = "UNFUSION",
	.pme_desc = "Number of unfusion events in the ROB (due to exception)"
  },
  { .pme_code = 0xE0,
	.pme_name = "BR_INSTR_DECODED",
	.pme_desc = "Branch instructions decoded"
  },
  { .pme_code = 0xE2,
	.pme_name = "BTB_MISSES",
	.pme_desc = "Number of branches the BTB did not produce a prediction"
  },
  { .pme_code = 0xE4,
	.pme_name = "BR_BOGUS",
	.pme_desc = "Number of bogus branches"
  },
  { .pme_code = 0xE6,
	.pme_name = "BACLEARS",
	.pme_desc = "Number of BAClears asserted"
  },
  { .pme_code = 0xF0,
	.pme_name = "PREF_RQSTS_UP",
	.pme_desc = "Number of hardware prefetch requests issued in forward streams"
  },
  { .pme_code = 0xF8,
	.pme_name = "PREF_RQSTS_DN",
	.pme_desc = "Number of hardware prefetch requests issued in backward streams"
  }
};


#define PME_COREDUO_UNHALTED_CORE_CYCLES 0
#define PME_COREDUO_INSTRUCTIONS_RETIRED 2
#define PME_COREDUO_EVENT_COUNT (sizeof(coreduo_pe)/sizeof(pme_coreduo_entry_t))
