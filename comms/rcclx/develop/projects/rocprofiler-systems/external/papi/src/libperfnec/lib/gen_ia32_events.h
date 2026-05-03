/*
 * Copyright (c) 2006-2007 Hewlett-Packard Development Company, L.P.
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


/*
 * architected events for architectural perfmon v1 and v2 as defined by the IA-32 developer's manual
 * Vol 3B, table 18-6 (May 2007)
 */
static pme_gen_ia32_entry_t gen_ia32_all_pe[]={
	{.pme_name = "UNHALTED_CORE_CYCLES",
	 .pme_code = 0x003c,
	 .pme_fixed = 17,
	 .pme_desc =  "count core clock cycles whenever the clock signal on the specific core is running (not halted)"
	},
	{.pme_name = "INSTRUCTIONS_RETIRED",
	 .pme_code = 0x00c0,
	 .pme_fixed = 16,
	 .pme_desc =  "count the number of instructions at retirement. For instructions that consists of multiple micro-ops, this event counts the retirement of the last micro-op of the instruction",
	},
	{.pme_name = "UNHALTED_REFERENCE_CYCLES",
	 .pme_code = 0x013c,
	 .pme_fixed = 18,
	 .pme_desc =  "count reference clock cycles while the clock signal on the specific core is running. The reference clock operates at a fixed frequency, irrespective of core freqeuncy changes due to performance state transitions",
	},
	{.pme_name = "LAST_LEVEL_CACHE_REFERENCES",
	 .pme_code = 0x4f2e,
	 .pme_desc =  "count each request originating from the core to reference a cache line in the last level cache. The count may include speculation, but excludes cache line fills due to hardware prefetch",
	},
	{.pme_name = "LAST_LEVEL_CACHE_MISSES",
	 .pme_code = 0x412e,
	 .pme_desc =  "count each cache miss condition for references to the last level cache. The event count may include speculation, but excludes cache line fills due to hardware prefetch",
	},
	{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	 .pme_code = 0x00c4,
	 .pme_desc =  "count branch instructions at retirement. Specifically, this event counts the retirement of the last micro-op of a branch instruction",
	},
	{.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	 .pme_code = 0x00c5,
	 .pme_desc =  "count mispredicted branch instructions at retirement. Specifically, this event counts at retirement of the last micro-op of a branch instruction in the architectural path of the execution and experienced misprediction in the branch prediction hardware",
	}
};
#define PME_GEN_IA32_UNHALTED_CORE_CYCLES 0
#define PME_GEN_IA32_INSTRUCTIONS_RETIRED 1
#define PFMLIB_GEN_IA32_EVENT_COUNT	  (sizeof(gen_ia32_all_pe)/sizeof(pme_gen_ia32_entry_t))
