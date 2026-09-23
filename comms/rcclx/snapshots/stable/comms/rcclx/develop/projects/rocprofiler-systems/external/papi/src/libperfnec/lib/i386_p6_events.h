/*
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
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
#define I386_P6_MESI_UMASKS \
	.pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	.pme_numasks = 4, \
	.pme_umasks = { \
		{ .pme_uname = "I", \
		  .pme_udesc = "invalid state", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "S", \
		  .pme_udesc = "shared state", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "E", \
		  .pme_udesc = "exclusive state", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "M", \
		  .pme_udesc = "modified state", \
		  .pme_ucode = 0x8 \
		}}

#define I386_PM_MESI_PREFETCH_UMASKS \
	.pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	.pme_numasks = 7, \
	.pme_umasks = { \
		{ .pme_uname = "I", \
		  .pme_udesc = "invalid state", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "S", \
		  .pme_udesc = "shared state", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "E", \
		  .pme_udesc = "exclusive state", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "M", \
		  .pme_udesc = "modified state", \
		  .pme_ucode = 0x8 \
		}, \
		{ .pme_uname = "EXCL_HW_PREFETCH", \
		  .pme_udesc = "exclude hardware prefetched lines", \
		  .pme_ucode = 0x0 \
		}, \
		{ .pme_uname = "ONLY_HW_PREFETCH", \
		  .pme_udesc = "only hardware prefetched lines", \
		  .pme_ucode = 0x1 << 4 \
		}, \
		{ .pme_uname = "NON_HW_PREFETCH", \
		  .pme_udesc = "non hardware prefetched lines", \
		  .pme_ucode = 0x2 << 4 \
		}} 


#define I386_P6_PII_ONLY_PME \
	{.pme_name = "MMX_INSTR_EXEC",\
	 .pme_code = 0xb0,\
	 .pme_desc = "Number of MMX instructions executed"\
	},\
	{.pme_name = "MMX_INSTR_RET",\
	 .pme_code = 0xce,\
	 .pme_desc = "Number of MMX instructions retired"\
	}\

#define I386_P6_PII_PIII_PME \
	{.pme_name = "MMX_SAT_INSTR_EXEC",\
	 .pme_code = 0xb1,\
	 .pme_desc = "Number of MMX saturating instructions executed"\
	},\
	{.pme_name = "MMX_UOPS_EXEC",\
	 .pme_code = 0xb2,\
	 .pme_desc = "Number of MMX micro-ops executed"\
	},\
	{.pme_name = "MMX_INSTR_TYPE_EXEC",\
	 .pme_code = 0xb3,\
	 .pme_desc = "Number of MMX instructions executed by type",\
	 .pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	 .pme_numasks = 6, \
	 .pme_umasks = { \
		{ .pme_uname = "MUL", \
		  .pme_udesc = "MMX packed multiply instructions executed", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "SHIFT", \
		  .pme_udesc = "MMX packed shift instructions executed", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "PACK", \
		  .pme_udesc = "MMX pack operation instructions executed", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "UNPACK", \
		  .pme_udesc = "MMX unpack operation instructions executed", \
		  .pme_ucode = 0x8 \
		}, \
		{ .pme_uname = "LOGICAL", \
		  .pme_udesc = "MMX packed logical instructions executed", \
		  .pme_ucode = 0x10 \
		}, \
		{ .pme_uname = "ARITH", \
		  .pme_udesc = "MMX packed arithmetic instructions executed", \
		  .pme_ucode = 0x20 \
		} \
	 }\
	},\
	{.pme_name = "FP_MMX_TRANS",\
	 .pme_code = 0xcc,\
	 .pme_desc = "Number of MMX transitions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "TO_FP", \
		  .pme_udesc = "from MMX instructions to floating-point instructions", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "TO_MMX", \
		  .pme_udesc = "from floating-point instructions to MMX instructions", \
		  .pme_ucode = 0x01 \
		}\
	 }\
	},\
	{.pme_name = "MMX_ASSIST",\
	 .pme_code = 0xcd,\
	 .pme_desc = "Number of MMX micro-ops executed"\
	},\
	{.pme_name = "SEG_RENAME_STALLS",\
	 .pme_code = 0xd4,\
	 .pme_desc = "Number of Segment Register Renaming Stalls", \
	 .pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	 .pme_numasks = 4, \
	 .pme_umasks = { \
		{ .pme_uname = "ES", \
		  .pme_udesc = "Segment register ES", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "DS", \
		  .pme_udesc = "Segment register DS", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "FS", \
		  .pme_udesc = "Segment register FS", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "GS", \
		  .pme_udesc = "Segment register GS", \
		  .pme_ucode = 0x8 \
		} \
	 }\
	},\
	{.pme_name = "SEG_REG_RENAMES",\
	 .pme_code = 0xd5,\
	 .pme_desc = "Number of Segment Register Renames", \
	 .pme_flags   = PFMLIB_I386_P6_UMASK_COMBO, \
	 .pme_numasks = 4, \
	 .pme_umasks = { \
		{ .pme_uname = "ES", \
		  .pme_udesc = "Segment register ES", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "DS", \
		  .pme_udesc = "Segment register DS", \
		  .pme_ucode = 0x2 \
		}, \
		{ .pme_uname = "FS", \
		  .pme_udesc = "Segment register FS", \
		  .pme_ucode = 0x4 \
		}, \
		{ .pme_uname = "GS", \
		  .pme_udesc = "Segment register GS", \
		  .pme_ucode = 0x8 \
		} \
	 }\
	},\
	{.pme_name = "RET_SEG_RENAMES",\
	 .pme_code = 0xd6,\
	 .pme_desc = "Number of segment register rename events retired"\
	} \

#define I386_P6_PIII_PME \
	{.pme_name = "EMON_KNI_PREF_DISPATCHED",\
	 .pme_code = 0x07,\
	 .pme_desc = "Number of Streaming SIMD extensions prefetch/weakly-ordered instructions dispatched " \
		     "(speculative prefetches are included in counting). Pentium III and later",\
	 .pme_numasks = 4, \
	 .pme_umasks = { \
		{ .pme_uname = "NTA", \
		  .pme_udesc = "prefetch NTA", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "T1", \
		  .pme_udesc = "prefetch T1", \
		  .pme_ucode = 0x01 \
		}, \
		{ .pme_uname = "T2", \
		  .pme_udesc = "prefetch T2", \
		  .pme_ucode = 0x02 \
		}, \
		{ .pme_uname = "WEAK", \
		  .pme_udesc = "weakly ordered stores", \
		  .pme_ucode = 0x03 \
		} \
	 } \
	},\
	{.pme_name = "EMON_KNI_PREF_MISS",\
	 .pme_code = 0x4b,\
	 .pme_desc = "Number of prefetch/weakly-ordered instructions that miss all caches. Pentium III and later",\
	 .pme_numasks = 4, \
	 .pme_umasks = { \
		{ .pme_uname = "NTA", \
		  .pme_udesc = "prefetch NTA", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "T1", \
		  .pme_udesc = "prefetch T1", \
		  .pme_ucode = 0x01 \
		}, \
		{ .pme_uname = "T2", \
		  .pme_udesc = "prefetch T2", \
		  .pme_ucode = 0x02 \
		}, \
		{ .pme_uname = "WEAK", \
		  .pme_udesc = "weakly ordered stores", \
		  .pme_ucode = 0x03 \
		} \
	 } \
	} \


#define I386_P6_CPU_CLK_UNHALTED \
	{.pme_name = "CPU_CLK_UNHALTED",\
	 .pme_code = 0x79,\
	 .pme_desc =  "Number cycles during which the processor is not halted"\
	}\


#define I386_P6_NOT_PM_PME \
	{.pme_name = "L2_LD",\
	 .pme_code = 0x29,\
	 .pme_desc =  "Number of L2 data loads. This event indicates that a normal, unlocked, load memory access "\
	 	"was received by the L2. It includes only L2 cacheable memory accesses; it does not include I/O "\
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include "\
		"L2 cacheable TLB miss memory accesses",\
	 I386_P6_MESI_UMASKS\
	},\
	{.pme_name = "L2_LINES_IN",\
	 .pme_code = 0x24,\
	 .pme_desc =  "Number of lines allocated in the L2"\
	},\
	{.pme_name = "L2_LINES_OUT",\
	 .pme_code = 0x26,\
	 .pme_desc =  "Number of lines removed from the L2 for any reason"\
	},\
	{.pme_name = "L2_M_LINES_OUTM",\
	 .pme_code = 0x27,\
	 .pme_desc =  "Number of modified lines removed from the L2 for any reason"\
	}\


#define I386_P6_PIII_NOT_PM_PME \
	{.pme_name = "EMON_KNI_INST_RETIRED",\
	 .pme_code = 0xd8,\
	 .pme_desc = "Number of SSE instructions retired. Pentium III and later",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "PACKED_SCALAR", \
		  .pme_udesc = "packed and scalar instructions", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "SCALAR", \
		  .pme_udesc = "scalar only", \
		  .pme_ucode = 0x01 \
		} \
	 } \
	},\
	{.pme_name = "EMON_KNI_COMP_INST_RET",\
	 .pme_code = 0xd9,\
	 .pme_desc = "Number of SSE computation instructions retired. Pentium III and later",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "PACKED_SCALAR", \
		  .pme_udesc = "packed and scalar instructions", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "SCALAR", \
		  .pme_udesc = "scalar only", \
		  .pme_ucode = 0x01 \
		} \
	 } \
	}\



#define I386_P6_COMMON_PME \
	{.pme_name = "INST_RETIRED",\
	 .pme_code = 0xc0,\
	 .pme_desc = "Number of instructions retired"\
	},\
	{.pme_name = "DATA_MEM_REFS",\
	 .pme_code = 0x43,\
	 .pme_desc = "All loads from any memory type. All stores to any memory type"\
		"Each part of a split is counted separately. The internal logic counts not only memory loads and stores"\
		" but also internal retries. 80-bit floating point accesses are double counted, since they are decomposed"\
		" into a 16-bit exponent load and a 64-bit mantissa load. Memory accesses are only counted when they are "\
		" actually performed (such as a load that gets squashed because a previous cache miss is outstanding to the"\
		" same address, and which finally gets performe, is only counted once). Does ot include I/O accesses or other"\
		" non-memory accesses"\
	},\
	{.pme_name = "DCU_LINES_IN",\
	 .pme_code = 0x45,\
	 .pme_desc = "Total lines allocated in the DCU"\
	},\
	{.pme_name = "DCU_M_LINES_IN",\
	 .pme_code = 0x46,\
	 .pme_desc = "Number of M state lines allocated in the DCU"\
	},\
	{.pme_name = "DCU_M_LINES_OUT",\
	 .pme_code = 0x47,\
	 .pme_desc = "Number of M state lines evicted from the DCU. This includes evictions via snoop HITM, intervention"\
	 	     " or replacement"\
	},\
	{.pme_name = "DCU_MISS_OUTSTANDING",\
	 .pme_code = 0x48,\
	 .pme_desc = "Weighted number of cycle while a DCU miss is outstanding, incremented by the number of cache misses"\
	 	     " at any particular time. Cacheable read requests only are considered. Uncacheable requests are excluded"\
		     " Read-for-ownerships are counted, as well as line fills, invalidates, and stores"\
	},\
	{.pme_name = "IFU_IFETCH",\
	 .pme_code = 0x80,\
	 .pme_desc = "Number of instruction fetches, both cacheable and noncacheable including UC fetches"\
	},\
	{.pme_name = "IFU_IFETCH_MISS",\
	 .pme_code = 0x81,\
	 .pme_desc = "Number of instruction fetch misses. All instructions fetches that do not hit the IFU (i.e., that"\
	 	     " produce memory requests). Includes UC accesses"\
	},\
	{.pme_name = "ITLB_MISS",\
	 .pme_code = 0x85,\
	 .pme_desc = "Number of ITLB misses"\
	},\
	{.pme_name = "IFU_MEM_STALL",\
	 .pme_code = 0x86,\
	 .pme_desc = "Number of cycles instruction fetch is stalled for any reason. Includs IFU cache misses, ITLB misses,"\
	 	     " ITLB faults, and other minor stalls"\
	},\
	{.pme_name = "ILD_STALL",\
	 .pme_code = 0x87,\
	 .pme_desc = "Number of cycles that the instruction length decoder is stalled"\
	},\
	{.pme_name = "L2_IFETCH",\
	 .pme_code = 0x28,\
	 .pme_desc =  "Number of L2 instruction fetches. This event indicates that a normal instruction fetch was received by"\
	 	" the L2. The count includes only L2 cacheable instruction fetches: it does not include UC instruction fetches"\
		" It does not include ITLB miss accesses",\
	 I386_P6_MESI_UMASKS \
	}, \
	{.pme_name = "L2_ST",\
	 .pme_code = 0x2a,\
	 .pme_desc =  "Number of L2 data stores. This event indicates that a normal, unlocked, store memory access "\
	 	"was received by the L2. Specifically, it indictes that the DCU sent a read-for ownership request to " \
		"the L2. It also includes Invalid to Modified reqyests sent by the DCU to the L2. " \
		"It includes only L2 cacheable memory accesses;  it does not include I/O " \
		"accesses, other non-memory accesses, or memory accesses such as UC/WT memory accesses. It does include " \
		"L2 cacheable TLB miss memory accesses", \
	 I386_P6_MESI_UMASKS \
	},\
	{.pme_name = "L2_M_LINES_INM",\
	 .pme_code = 0x25,\
	 .pme_desc = "Number of modified lines allocated in the L2"\
	},\
	{.pme_name = "L2_RQSTS",\
	 .pme_code = 0x2e,\
	 .pme_desc = "Total number of L2 requests",\
	 I386_P6_MESI_UMASKS \
	},\
	{.pme_name = "L2_ADS",\
	 .pme_code = 0x21,\
	 .pme_desc = "Number of L2 address strobes"\
	},\
	{.pme_name = "L2_DBUS_BUSY",\
	 .pme_code = 0x22,\
	 .pme_desc = "Number of cycles during which the L2 cache data bus was busy"\
	},\
	{.pme_name = "L2_DBUS_BUSY_RD",\
	 .pme_code = 0x23,\
	 .pme_desc = "Number of cycles during which the data bus was busy transferring read data from L2 to the processor"\
	},\
	{.pme_name = "BUS_DRDY_CLOCKS",\
	 .pme_code = 0x62,\
	 .pme_desc = "Number of clocks during which DRDY# is asserted. " \
		     "Utilization of the external system data bus during data transfers", \
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_LOCK_CLOCKS",\
	 .pme_code = 0x63,\
	 .pme_desc = "Number of clocks during which LOCK# is asserted on the external system bus", \
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_REQ_OUTSTANDING",\
	 .pme_code = 0x60,\
	 .pme_desc = "Number of bus requests outstanding. This counter is incremented " \
		"by the number of cacheable read bus requests outstanding in any given cycle", \
	},\
	{.pme_name = "BUS_TRANS_BRD",\
	 .pme_code = 0x65,\
	 .pme_desc = "Number of burst read transactions", \
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRANS_RFO",\
	 .pme_code = 0x66,\
	 .pme_desc = "Number of completed read for ownership transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRANS_WB",\
	 .pme_code = 0x67,\
	 .pme_desc = "Number of completed write back transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_IFETCH",\
	 .pme_code = 0x68,\
	 .pme_desc = "Number of completed instruction fetch transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_INVAL",\
	 .pme_code = 0x69,\
	 .pme_desc = "Number of completed invalidate transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_PWR",\
	 .pme_code = 0x6a,\
	 .pme_desc = "Number of completed partial write transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRANS_P",\
	 .pme_code = 0x6b,\
	 .pme_desc = "Number of completed partial transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRANS_IO",\
	 .pme_code = 0x6c,\
	 .pme_desc = "Number of completed I/O transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_DEF",\
	 .pme_code = 0x6d,\
	 .pme_desc = "Number of completed deferred transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x1 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x2 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_BURST",\
	 .pme_code = 0x6e,\
	 .pme_desc = "Number of completed burst transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_ANY",\
	 .pme_code = 0x70,\
	 .pme_desc = "Number of all completed bus transactions. Address bus utilization " \
		"can be calculated knowing the minimum address bus occupancy. Includes special cycles, etc.",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_TRAN_MEM",\
	 .pme_code = 0x6f,\
	 .pme_desc = "Number of completed memory transactions",\
	 .pme_numasks = 2, \
	 .pme_umasks = { \
		{ .pme_uname = "SELF", \
		  .pme_udesc = "clocks when processor is driving bus", \
		  .pme_ucode = 0x00 \
		}, \
		{ .pme_uname = "ANY", \
		  .pme_udesc = "clocks when any agent is driving bus", \
		  .pme_ucode = 0x20 \
		} \
	 } \
	},\
	{.pme_name = "BUS_DATA_RECV",\
	 .pme_code = 0x64,\
	 .pme_desc = "Number of bus clock cycles during which this processor is receiving data"\
	},\
	{.pme_name = "BUS_BNR_DRV",\
	 .pme_code = 0x61,\
	 .pme_desc = "Number of bus clock cycles during which this processor is driving the BNR# pin"\
	},\
	{.pme_name = "BUS_HIT_DRV",\
	 .pme_code = 0x7a,\
	 .pme_desc = "Number of bus clock cycles during which this processor is driving the HIT# pin"\
	},\
	{.pme_name = "BUS_HITM_DRV",\
	 .pme_code = 0x7b,\
	 .pme_desc = "Number of bus clock cycles during which this processor is driving the HITM# pin"\
	},\
	{.pme_name = "BUS_SNOOP_STALL",\
	 .pme_code = 0x7e,\
	 .pme_desc = "Number of clock cycles during which the bus is snoop stalled"\
	},\
	{.pme_name = "FLOPS",\
	 .pme_code = 0xc1,\
	 .pme_desc = "Number of computational floating-point operations retired. " \
		     "Excludes floating-point computational operations that cause traps or assists. " \
		     "Includes internal sub-operations for complex floating-point instructions like transcendentals. " \
		     "Excludes floating point loads and stores", \
	 .pme_flags   = PFMLIB_I386_P6_CTR0_ONLY \
	},\
	{.pme_name = "FP_COMP_OPS_EXE",\
	 .pme_code = 0x10,\
	 .pme_desc = "Number of computational floating-point operations executed. The number of FADD, FSUB, " \
		     "FCOM, FMULs, integer MULs and IMULs, FDIVs, FPREMs, FSQRTS, integer DIVs, and IDIVs. " \
		     "This number does not include the number of cycles, but the number of operations. " \
		     "This event does not distinguish an FADD used in the middle of a transcendental flow " \
		     "from a separate FADD instruction", \
	 .pme_flags   = PFMLIB_I386_P6_CTR0_ONLY \
	},\
	{.pme_name = "FP_ASSIST",\
	 .pme_code = 0x11,\
	 .pme_desc = "Number of floating-point exception cases handled by microcode.", \
	.pme_flags   = PFMLIB_I386_P6_CTR1_ONLY \
	},\
	{.pme_name = "MUL",\
	 .pme_code = 0x12,\
	 .pme_desc = "Number of multiplies." \
		     "This count includes integer as well as FP multiplies and is speculative", \
	 .pme_flags   = PFMLIB_I386_P6_CTR1_ONLY \
	},\
	{.pme_name = "DIV",\
	 .pme_code = 0x13,\
	 .pme_desc = "Number of divides." \
		     "This count includes integer as well as FP divides and is speculative", \
	 .pme_flags   = PFMLIB_I386_P6_CTR1_ONLY \
	},\
	{.pme_name = "CYCLES_DIV_BUSY",\
	 .pme_code = 0x14,\
	 .pme_desc = "Number of cycles during which the divider is busy, and cannot accept new divides. " \
		     "This includes integer and FP divides, FPREM, FPSQRT, etc. and is speculative", \
	 .pme_flags   = PFMLIB_I386_P6_CTR0_ONLY \
	},\
	{.pme_name = "LD_BLOCKS",\
	 .pme_code = 0x03,\
	 .pme_desc = "Number of load operations delayed due to store buffer blocks. Includes counts " \
		     "caused by preceding stores whose addresses are unknown, preceding stores whose addresses " \
		     "are known but whose data is unknown, and preceding stores that conflicts with the load " \
		     "but which incompletely overlap the load" \
	},\
	{.pme_name = "SB_DRAINS",\
	 .pme_code = 0x04,\
	 .pme_desc = "Number of store buffer drain cycles. Incremented every cycle the store buffer is draining. " \
		     "Draining is caused by serializing operations like CPUID, synchronizing operations " \
		     "like XCHG, interrupt acknowledgment, as well as other conditions (such as cache flushing)."\
	},\
	{.pme_name = "MISALIGN_MEM_REF",\
	 .pme_code = 0x05,\
	 .pme_desc = "Number of misaligned data memory references. Incremented by 1 every cycle during "\
		     "which, either the processor's load or store pipeline dispatches a misaligned micro-op "\
		     "Counting is performed if it is the first or second half or if it is blocked, squashed, "\
		     "or missed. In this context, misaligned means crossing a 64-bit boundary"\
	},\
	{.pme_name = "UOPS_RETIRED",\
	 .pme_code = 0xc2,\
	 .pme_desc =  "Number of micro-ops retired"\
	},\
	{.pme_name = "INST_DECODED",\
	 .pme_code = 0xd0,\
	 .pme_desc = "Number of instructions decoded"\
	},\
	{.pme_name = "HW_INT_RX",\
	 .pme_code = 0xc8,\
	 .pme_desc = "Number of hardware interrupts received"\
	},\
	{.pme_name = "CYCLES_INT_MASKED",\
	 .pme_code = 0xc6,\
	 .pme_desc = "Number of processor cycles for which interrupts are disabled"\
	},\
	{.pme_name = "CYCLES_INT_PENDING_AND_MASKED",\
	 .pme_code = 0xc7,\
	 .pme_desc = "Number of processor cycles for which interrupts are disabled and interrupts are pending."\
	},\
	{.pme_name = "BR_INST_RETIRED",\
	 .pme_code = 0xc4,\
	 .pme_desc = "Number of branch instructions retired"\
	},\
	{.pme_name = "BR_MISS_PRED_RETIRED",\
	 .pme_code = 0xc5,\
	 .pme_desc = "Number of mispredicted branches retired"\
	},\
	{.pme_name = "BR_TAKEN_RETIRED",\
	 .pme_code = 0xc9,\
	 .pme_desc = "Number of taken branches retired"\
	},\
	{.pme_name = "BR_MISS_PRED_TAKEN_RET",\
	 .pme_code = 0xca,\
	 .pme_desc = "Number of taken mispredicted branches retired"\
	},\
	{.pme_name = "BR_INST_DECODED",\
	 .pme_code = 0xe0,\
	 .pme_desc = "Number of branch instructions decoded"\
	},\
	{.pme_name = "BTB_MISSES",\
	 .pme_code = 0xe2,\
	 .pme_desc = "Number of branches for which the BTB did not produce a prediction"\
	},\
	{.pme_name = "BR_BOGUS",\
	 .pme_code = 0xe4,\
	 .pme_desc = "Number of bogus branches"\
	},\
	{.pme_name = "BACLEARS",\
	 .pme_code = 0xe6,\
	 .pme_desc = "Number of times BACLEAR is asserted. This is the number of times that " \
		     "a static branch prediction was made, in which the branch decoder decided " \
		     "to make a branch prediction because the BTB did not" \
	},\
	{.pme_name = "RESOURCE_STALLS",\
	 .pme_code = 0xa2,\
	 .pme_desc = "Incremented by 1 during every cycle for which there is a resource related stall. " \
		     "Includes register renaming buffer entries, memory buffer entries. Does not include " \
		     "stalls due to bus queue full, too many cache misses, etc. In addition to resource " \
		     "related stalls, this event counts some other events. Includes stalls arising during " \
		     "branch misprediction recovery, such as if retirement of the mispredicted branch is " \
		     "delayed and stalls arising while store buffer is draining from synchronizing operations" \
	},\
	{.pme_name = "PARTIAL_RAT_STALLS",\
	 .pme_code = 0xd2,\
	 .pme_desc = "Number of cycles or events for partial stalls. This includes flag partial stalls"\
	},\
	{.pme_name = "SEGMENT_REG_LOADS",\
	 .pme_code = 0x06,\
	 .pme_desc = "Number of segment register loads."\
	}\



/*
 * Pentium Pro Processor Event Table
 */
static pme_i386_p6_entry_t i386_ppro_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,       /* generic p6 */
	I386_P6_NOT_PM_PME,       /* generic p6 that conflict with Pentium M */
};

#define PME_I386_PPRO_CPU_CLK_UNHALTED 0
#define PME_I386_PPRO_INST_RETIRED 1
#define PME_I386_PPRO_EVENT_COUNT 	(sizeof(i386_ppro_pe)/sizeof(pme_i386_p6_entry_t))


/*
 * Pentium II Processor Event Table
 */
static pme_i386_p6_entry_t i386_pII_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,   /* generic p6 */
	I386_P6_PII_ONLY_PME, /* pentium II only */
        I386_P6_PII_PIII_PME, /* pentium II and later */
	I386_P6_NOT_PM_PME,   /* generic p6 that conflict with Pentium M */
};

#define PME_I386_PII_CPU_CLK_UNHALTED 0
#define PME_I386_PII_INST_RETIRED 1
#define PME_I386_PII_EVENT_COUNT 	(sizeof(i386_pII_pe)/sizeof(pme_i386_p6_entry_t))


/*
 * Pentium III Processor Event Table
 */
static pme_i386_p6_entry_t i386_pIII_pe []={
        I386_P6_CPU_CLK_UNHALTED, /* should be first */
        I386_P6_COMMON_PME,     /* generic p6 */
        I386_P6_PII_PIII_PME,   /* pentium II and later */
	I386_P6_PIII_PME,       /* pentium III and later */
	I386_P6_NOT_PM_PME,     /* generic p6 that conflict with Pentium M */
	I386_P6_PIII_NOT_PM_PME /* pentium III that conflict with Pentium M */
};

#define PME_I386_PIII_CPU_CLK_UNHALTED 0
#define PME_I386_PIII_INST_RETIRED 1
#define PME_I386_PIII_EVENT_COUNT 	(sizeof(i386_pIII_pe)/sizeof(pme_i386_p6_entry_t))


/*
 * Pentium M event table
 * It is different from regular P6 because it supports additional events
 * and also because the semantics of some events is slightly different
 *
 * The library autodetects which table to use during pfmlib_initialize()
 */
static pme_i386_p6_entry_t i386_pm_pe []={
	{.pme_name = "CPU_CLK_UNHALTED",
	 .pme_code = 0x79,
	 .pme_desc = "Number cycles during which the processor is not halted and not in a thermal trip"
	},

	I386_P6_COMMON_PME,     /* generic p6 */
 	I386_P6_PII_PIII_PME,   /* pentium II and later */
 	I386_P6_PIII_PME,       /* pentium III and later */

	{.pme_name = "EMON_EST_TRANS",
	 .pme_code = 0x58,
	 .pme_desc = "Number of Enhanced Intel SpeedStep technology transitions",
	 .pme_numasks = 2,
	 .pme_umasks = {
		{ .pme_uname = "ALL",
		  .pme_udesc = "All transitions",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "FREQ",
		  .pme_udesc = "Only frequency transitions",
		  .pme_ucode = 0x2
		},
	 }
	},
	{.pme_name = "EMON_THERMAL_TRIP",
	 .pme_code = 0x59,
	 .pme_desc = "Duration/occurrences in thermal trip; to count the number of thermal trips; edge detect must be used"
	},
	{.pme_name = "BR_INST_EXEC",
	 .pme_code = 0x088,
	 .pme_desc = "Branch instructions executed (not necessarily retired)"
	},
	{.pme_name = "BR_MISSP_EXEC",
	 .pme_code = 0x89,
	 .pme_desc = "Branch instructions executed that were mispredicted at execution"
	},
	{.pme_name = "BR_BAC_MISSP_EXEC",
	 .pme_code = 0x8a,
	 .pme_desc = "Branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CND_EXEC",
	 .pme_code = 0x8b,
	 .pme_desc = "Conditional branch instructions executed"
	},
	{.pme_name = "BR_CND_MISSP_EXEC",
	 .pme_code = 0x8c,
	 .pme_desc = "Conditional branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_EXEC",
	 .pme_code = 0x8d,
	 .pme_desc = "Indirect branch instructions executed"
	},
	{.pme_name = "BR_IND_MISSP_EXEC",
	 .pme_code = 0x8e,
	 .pme_desc = "Indirect branch instructions executed that were mispredicted"
	},
	{.pme_name = "BR_RET_EXEC",
	 .pme_code = 0x8f,
	 .pme_desc = "Return branch instructions executed"
	},
	{.pme_name = "BR_RET_MISSP_EXEC",
	 .pme_code = 0x90,
	 .pme_desc = "Return branch instructions executed that were mispredicted at Execution"
	},
	{.pme_name = "BR_RET_BAC_MISSP_EXEC",
	 .pme_code = 0x91,
	 .pme_desc = "Return branch instructions executed that were mispredicted at Front End (BAC)"
	},
	{.pme_name = "BR_CALL_EXEC",
	 .pme_code = 0x92,
	 .pme_desc = "CALL instructions executed"
	},
	{.pme_name = "BR_CALL_MISSP_EXEC",
	 .pme_code = 0x93,
	 .pme_desc = "CALL instructions executed that were mispredicted"
	},
	{.pme_name = "BR_IND_CALL_EXEC",
	 .pme_code = 0x94,
	 .pme_desc = "Indirect CALL instructions executed"
	},
	{.pme_name = "EMON_SIMD_INSTR_RETIRED",
	 .pme_code = 0xce,
	 .pme_desc = "Number of retired MMX instructions"
	},
	{.pme_name = "EMON_SYNCH_UOPS",
	 .pme_code = 0xd3,
	 .pme_desc = "Sync micro-ops"
	},
	{.pme_name = "EMON_ESP_UOPS",
	 .pme_code = 0xd7,
	 .pme_desc = "Total number of micro-ops"
	},
	{.pme_name = "EMON_FUSED_UOPS_RET",
	 .pme_code = 0xda,
	 .pme_desc = "Total number of micro-ops",
	 .pme_flags = PFMLIB_I386_P6_UMASK_COMBO,
	 .pme_numasks = 3,
	 .pme_umasks = {
		{ .pme_uname = "ALL",
		  .pme_udesc = "All fused micro-ops",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "LD_OP",
		  .pme_udesc = "Only load+Op micro-ops",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "STD_STA",
		  .pme_udesc = "Only std+sta micro-ops",
		  .pme_ucode = 0x2
		}
	  }
	},
	{.pme_name = "EMON_UNFUSION",
	 .pme_code = 0xdb,
	 .pme_desc = "Number of unfusion events in the ROB, happened on a FP exception to a fused micro-op"
	},
	{.pme_name = "EMON_PREF_RQSTS_UP",
	 .pme_code = 0xf0,
	 .pme_desc = "Number of upward prefetches issued"
	},
	{.pme_name = "EMON_PREF_RQSTS_DN",
	 .pme_code = 0xf8,
	 .pme_desc = "Number of downward prefetches issued"
	},
	{.pme_name = "EMON_SSE_SSE2_INST_RETIRED",
	 .pme_code = 0xd8,
	 .pme_desc =  "Streaming SIMD extensions instructions retired",
	 .pme_numasks = 4,
	 .pme_umasks = {
		{ .pme_uname = "SSE_PACKED_SCALAR_SINGLE",
		  .pme_udesc = "SSE Packed Single and Scalar Single",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "SSE_SCALAR_SINGLE",
		  .pme_udesc = "SSE Scalar Single",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SSE2_PACKED_DOUBLE",
		  .pme_udesc = "SSE2 Packed Double",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "SSE2_SCALAR_DOUBLE",
		  .pme_udesc = "SSE2 Scalar Double",
		  .pme_ucode = 0x3
		}
	 }
	},
	{.pme_name = "EMON_SSE_SSE2_COMP_INST_RETIRED",
	 .pme_code = 0xd9,
	 .pme_desc =  "Computational SSE instructions retired",
	 .pme_numasks = 4,
	 .pme_umasks = {
		{ .pme_uname = "SSE_PACKED_SINGLE",
		  .pme_udesc = "SSE Packed Single",
		  .pme_ucode = 0x0
		},
		{ .pme_uname = "SSE_SCALAR_SINGLE",
		  .pme_udesc = "SSE Scalar Single",
		  .pme_ucode = 0x1
		},
		{ .pme_uname = "SSE2_PACKED_DOUBLE",
		  .pme_udesc = "SSE2 Packed Double",
		  .pme_ucode = 0x2
		},
		{ .pme_uname = "SSE2_SCALAR_DOUBLE",
		  .pme_udesc = "SSE2 Scalar Double",
		  .pme_ucode = 0x3
		}
	 }
	},
	{.pme_name = "L2_LD",
	 .pme_code = 0x29,
	 .pme_desc =  "Number of L2 data loads",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.pme_name = "L2_LINES_IN",
	 .pme_code = 0x24,
	 .pme_desc =  "Number of L2 lines allocated",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.pme_name = "L2_LINES_OUT",
	 .pme_code = 0x26,
	 .pme_desc =  "Number of L2 lines evicted",
	 I386_PM_MESI_PREFETCH_UMASKS
	},
	{.pme_name = "L2_M_LINES_OUT",
	 .pme_code = 0x27,
	 .pme_desc =  "Number of L2 M-state lines evicted",
	 I386_PM_MESI_PREFETCH_UMASKS
	}
};
#define PME_I386_PM_CPU_CLK_UNHALTED 0
#define PME_I386_PM_INST_RETIRED 1
#define PME_I386_PM_EVENT_COUNT 	(sizeof(i386_pm_pe)/sizeof(pme_i386_p6_entry_t))
