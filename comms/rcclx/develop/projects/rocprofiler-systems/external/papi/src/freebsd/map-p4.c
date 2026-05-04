/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-p4.c
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include "freebsd.h"
#include "papiStdEventDefs.h"
#include "map.h"


/****************************************************************************
 P4 SUBSTRATE 
 P4 SUBSTRATE 
 P4 SUBSTRATE (aka Pentium IV)
 P4 SUBSTRATE
 P4 SUBSTRATE
****************************************************************************/

/*
	NativeEvent_Value_P4Processor must match P4Processor_info 
*/

Native_Event_LabelDescription_t P4Processor_info[] =
{
	{ "p4-128bit-mmx-uop", "Count integer SIMD SSE2 instructions that operate on 128 bit SIMD operands." },
	{ "p4-64bit-mmx-uop", "Count MMX instructions that operate on 64 bit SIMD operands." },
	{ "p4-b2b-cycles", "Count back-to-back bys cycles." },
	{ "p4-bnr", "Count bus-not-ready conditions." },
	{ "p4-bpu-fetch-request", "Count instruction fetch requests." },
	{ "p4-branch-retired", "Counts retired branches." },
	{ "p4-bsq-active-entries", "Count the number of entries (clipped at 15) currently active in the BSQ." },
	{ "p4-bsq-allocation", "Count allocations in the bus sequence unit." },
	{ "p4-bsq-cache-reference", "Count cache references as seen by the bus unit." },
	{ "p4-execution-event", "Count the retirement uops through the execution mechanism." },
	{ "p4-front-end-event", "Count the retirement uops through the frontend mechanism." },
	{ "p4-fsb-data-activity", "Count each DBSY or DRDY event." },
	{ "p4-global-power-events", "Count cycles during which the processor is not stopped." },
	{ "p4-instr-retired", "Count all kind of instructions retired during a clock cycle." },
	{ "p4-ioq-active-entries", "Count the number of entries (clipped at 15) in the IOQ that are active." },
	{ "p4-ioq-allocation", "Count various types of transactions on the bus." },
	{ "p4-itlb-reference", "Count translations using the intruction translation look-aside buffer." },
	{ "p4-load-port-replay", "Count replayed events at the load port." },
	{ "p4-mispred-branch-retired", "Count mispredicted IA-32 branch instructions." },
	{ "p4-machine-clear", "Count the number of pipeline clears seen by the processor." },
	{ "p4-memory-cancel", " Count the cancelling of various kinds of requests in the data cache address control unit of the CPU." },
	{ "p4-memory-complete", "Count the completion of load split, store split, uncacheable split and uncacheable load operations." },
	{ "p4-mob-load-replay", "Count load replays triggered by the memory order buffer." },
	{ "p4-packed-dp-uop", "Count packed double-precision uops." },
	{ "p4-packed-sp-uop", "Count packed single-precision uops." },
	{ "p4-page-walk-type", "Count page walks performed by the page miss handler." },
	{ "p4-replay-event", "Count the retirement of tagged uops" },
	{ "p4-resource-stall", "Count the occurrence or latency of stalls in the allocator." },
	{ "p4-response", "Count different types of responses." },
	{ "p4-retired-branch-type", "Count branches retired." },
	{ "p4-retired-mispred-branch-type", "Count mispredicted branches retired." },
	{ "p4-scalar-dp-uop", "Count the number of scalar double-precision uops." },
	{ "p4-scalar-sp-uop", "Count the number of scalar single-precision uops." },
	{ "p4-snoop", "Count snoop traffic." },
	{ "p4-sse-input-assist", "Count the number of times an assist is required to handle problems with the operands for SSE and SSE2 operations." },
	{ "p4-store-port-replay", "Count events replayed at the store port." },
	{ "p4-tc-deliver-mode", "Count the duration in cycles of operating modes of the trace cache and decode engine." },
	{ "p4-tc-ms-xfer", "Count the number of times uop delivery changed from the trace cache to MS ROM." },
	{ "p4-uop-queue-writes", "Count the number of valid uops written to the uop queue." },
	{ "p4-uop-type", "This event is used in conjunction with the front-end at-retirement mechanism to tag load and store uops." },
	{ "p4-uops-retired", "Count uops retired during a clock cycle." },
	{ "p4-wc-buffer", "Count write-combining buffer operations." },
	{ "p4-x87-assist", "Count the retirement of x87 instructions that required special handling." },
	{ "p4-x87-fp-uop", "Count x87 floating-point uops." },
	{ "p4-x87-simd-moves-uop", "Count each x87 FPU, MMX, SSE, or SSE2 uops that load data or store data or perform register-to-register moves." },
	/* counters with some modifiers */
	{ "p4-uop-queue-writes,mask=+from-tc-build,+from-tc-deliver", "Count the number of valid uops written to the uop queue." },
	{ "p4-page-walk-type,mask=+dtmiss", "Count data page walks performed by the page miss handler." },
	{ "p4-page-walk-type,mask=+itmiss", "Count instruction page walks performed by the page miss handler." },
	{ "p4-instr-retired,mask=+nbogusntag,+nbogustag", "Count all non-bogus instructions retired during a clock cycle." },
	{ "p4-branch-retired,mask=+mmnp,+mmnm", "Count branches not-taken." },
	{ "p4-branch-retired,mask=+mmtm,+mmtp", "Count branches taken." },
	{ "p4-branch-retired,mask=+mmnp,+mmtp", "Count branches predicted." },
	{ "p4-branch-retired,mask=+mmnm,+mmtm", "Count branches mis-predicted." },
	{ "p4-bsq-cache-reference,mask=+rd-2ndl-miss", "Count 2nd level cache misses." },
	{ "p4-bsq-cache-reference,mask=+rd-2ndl-miss,+rd-2ndl-hits,+rd-2ndl-hite,+rd-2ndl-hitm", "Count 2nd level cache accesses." },
	{ "p4-bsq-cache-reference,mask=+rd-2ndl-hits,+rd-2ndl-hite,+rd-2ndl-hitm", "Count 2nd level cache hits." },
	{ "p4-bsq-cache-reference,mask=+rd-3rdl-miss", "Count 3rd level cache misses." },
	{ "p4-bsq-cache-reference,mask=+rd-3rdl-miss,+rd-3rdl-hits,+rd-3rdl-hite,+rd-3rdl-hitm", "Count 3rd level cache accesses." },
	{ "p4-bsq-cache-reference,mask=+rd-3rdl-hits,+rd-3rdl-hite,+rd-3rdl-hitm", "Count 3rd level cache hits." },
	{ NULL, NULL }
};

