#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <errno.h>
#include <stddef.h>
#include <dlfcn.h>

#ifndef _GNU_SOURCE
	#define _GNU_SOURCE
#endif
#include <sched.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h" /* defines papi_malloc(), etc. */

#include "topdown.h"

// The following macro follows if a string function has an error. It should
// never happen; but it is necessary to prevent compiler warnings. We print
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR                                                               \
	{                                                                                     \
		fprintf(stderr, "%s:%i unexpected string function error.\n", __FILE__, __LINE__); \
		exit(-1);                                                                         \
	}

papi_vector_t _topdown_vector;

static _topdown_native_event_entry_t *topdown_native_events = NULL;
static int num_events = 0;

static int librseq_loaded = 0;

#define INTEL_CORE_TYPE_EFFICIENT	0x20	/* also known as 'ATOM' */
#define INTEL_CORE_TYPE_PERFORMANCE	0x40	/* also known as 'CORE' */
#define INTEL_CORE_TYPE_HOMOGENEOUS	-1		/* core type is non-issue */
static int required_core_type = INTEL_CORE_TYPE_HOMOGENEOUS;

/**************************/
/* x86 specific functions */
/**************************/

/* forward declarations */
void assert_affinity(int core_type);
static inline __attribute__((always_inline))
unsigned long long rdpmc_rseq_protected(unsigned int counter, int allowed_core_type);

/* rdpmc instruction wrapper */
static inline unsigned long long _rdpmc(unsigned int counter) {

	unsigned int low, high;
	/* if we need protection... */
	if (required_core_type != INTEL_CORE_TYPE_HOMOGENEOUS) {
		/* if librseq is available, protect with librseq */
		if (librseq_loaded)
			return rdpmc_rseq_protected(counter, required_core_type);

		/* otherwise, just hope we aren't moved to an unsupported core */
		/* between assert_affinity() and the inline asm */
		assert_affinity(required_core_type);
	}

	__asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));

	return (unsigned long long)low | ((unsigned long long)high) <<32;
}


typedef struct {
	unsigned int eax;
	unsigned int ebx;
	unsigned int ecx;
	unsigned int edx;
} cpuid_reg_t;

void cpuid2( cpuid_reg_t *reg, unsigned int func, unsigned int subfunc )
{
	__asm__ ("cpuid;"
			 : "=a" (reg->eax), "=b" (reg->ebx), "=c" (reg->ecx), "=d" (reg->edx)
			 : "a"  (func), "c" (subfunc));
}

/**************************************/
/* Hybrid processor support functions */
/**************************************/

/* ensure the core this process is running on is of the correct type */
int active_core_type_is(int core_type)
{
	cpuid_reg_t reg;

	/* check that CPUID leaf 0x1A is supported */
	cpuid2(&reg, 0, 0);
	if (reg.eax < 0x1a) return PAPI_ENOSUPP;
	cpuid2(&reg, 0x1a, 0);
	if (reg.eax == 0) return PAPI_ENOSUPP;

	return ((reg.eax >> 24) & 0xff) == core_type;
}

/* helper to allow printing core type in errors */
void core_type_to_name(int core_type, char *out)
{
	int err;

	switch (core_type) {
		case INTEL_CORE_TYPE_EFFICIENT:
			err = snprintf(out, PAPI_MIN_STR_LEN, "e-core (Atom)");
			if (err > PAPI_MAX_STR_LEN)
				HANDLE_STRING_ERROR;
			break;

		case INTEL_CORE_TYPE_PERFORMANCE:
			err = snprintf(out, PAPI_MIN_STR_LEN, "p-core (Core)");
			if (err > PAPI_MAX_STR_LEN)
				HANDLE_STRING_ERROR;
			break;

		default:
			err = snprintf(out, PAPI_MIN_STR_LEN, "not applicable (N/A)");
			if (err > PAPI_MAX_STR_LEN)
				HANDLE_STRING_ERROR;
			break;
	}
}

/* exit if the core affinity is disallowed in order to avoid segfaulting */
void handle_affinity_error(int allowed_type)
{
	char allowed_name[PAPI_MIN_STR_LEN];

	core_type_to_name(allowed_type, allowed_name);
	fprintf(stderr, 
		"Error: Process was moved to an unsupported core type. To use the PAPI topdown component, process affinity must be limited to cores of type '%s' on this architecture.\n", 
		allowed_name);

	exit(127);
}

/* assert that the current process affinity is to an allowed core type */
void assert_affinity(int core_type) {
	/* ensure the process is still on a valid core to avoid segfaulting */
	if (!active_core_type_is(core_type)) {
		handle_affinity_error(core_type);
	}
}

/**********************************************/
/* Restartable sequence heterogeneous support */
/**********************************************/

/* dlsym access to librseq symbols */
static ptrdiff_t *rseq_offset_ptr;
static int (*rseq_available_ptr)(unsigned int query);

/* local wrappers for dlsym function pointers */
static int librseq_rseq_available(unsigned int query) { return (*rseq_available_ptr)(query); }

int link_librseq()
{
	void* lib = dlopen("librseq.so", RTLD_NOW);
	if (!lib) {
		return PAPI_ENOSUPP;
	}

	rseq_available_ptr = dlsym(lib, "rseq_available");
    if (rseq_available_ptr == NULL) {
		return PAPI_ENOSUPP;
    }
	rseq_offset_ptr = dlsym(lib, "rseq_offset");
    if (rseq_offset_ptr == NULL) {
		return PAPI_ENOSUPP;
    }

	if (!rseq_available_ptr(0)) {
		return PAPI_ENOSUPP;
	}

    return 0;
} 

/* This function assumes some properties of the system have been verified. */
/* 1. Must be an Intel x86 processor */
/* 2. Processor must be hybrid/heterogeneous (e-core/p-core) */
/* 3. perf_event_open() + mmap() have been used to enable userspace rdpmc */
static inline __attribute__((always_inline))
unsigned long long rdpmc_rseq_protected(unsigned int counter, int allowed_core_type)
{
	unsigned int low = -1;
	unsigned int high = -1;
	int core_check;

restart_sequence:
	core_check = 0;
	__asm__ __volatile__ goto (
		/* set up critical section of restartable sequence */
		".pushsection __rseq_cs, \"aw\"\n\t" ".balign 32\n\t" "3:\n\t" ".long 0x0\n\t" ".long 0x0\n\t" ".quad 1f\n\t" ".quad (2f) - (1f)\n\t" ".quad 4f\n\t" ".long 0x0\n\t" ".long 0x0\n\t" ".quad 1f\n\t" ".quad (2f) - (1f)\n\t" ".quad 4f\n\t" ".popsection\n\t" ".pushsection __rseq_cs_ptr_array, \"aw\"\n\t" ".quad 3b\n\t" ".popsection\n\t"
		
		/* start rseq by storing table entry pointer into rseq_cs. */
		"leaq 3b(%%rip), %%rax\n\t" 
		"movq %%rax, %%fs:8(%[rseq_offset])\n\t" 
		"1:\n\t"

		/* check if core type is valid */
		"mov $0x1A, %%eax\n\t"
		"mov $0x00, %%ecx\n\t"
		"cpuid\n\t"
		"mov %%eax, %[core_check]\n\t"
		"test %[core_type], %%eax\n\t"
		"jz 4f\n\t" /* abort if core type is invalid */

		/* make the rdpmc call */
		"movl %[counter], %%ecx\n\t"
		"rdpmc\n\t"
		/* retrieve results of rdpmc */
		"mov %%edx, %[high]\n\t"
		"mov %%eax, %[low]\n\t"
		"2:\n\t"
		/* define abort section */
		".pushsection __rseq_failure, \"ax\"\n\t" ".byte 0x0f, 0xb9, 0x3d\n\t" ".long " "0x53053053" "\n\t" "4" ":\n\t" "" "jmp %l[" "abort" "]\n\t" ".popsection\n\t"
		:
		: [core_check]	"m"  (core_check),
		  [low]			"m"	 (low),
		  [high]		"m"  (high),
		  [rseq_offset]	"r" (*rseq_offset_ptr),
		  [counter]		"r" (counter),
		  [core_type]	"r" (allowed_core_type << 24) /* shift mask into place */
		: "memory", "cc", "rax", "eax", "ecx", "edx"
		: abort
	);
	return (unsigned long long)low | ((unsigned long long)high) << 32;

abort:
	/* we may abort because the core type was found to be invalid, or */
	/* we might abort because the restartable sequence was preempted */
	/* therefore we have to check why the abort happened here */
	if ((((core_check >> 24) & 0xff) != allowed_core_type) && core_check != 0) {
		/* sequence reached the core check, and the core type was disallowed !*/
		handle_affinity_error(allowed_core_type);
		return PAPI_EBUG; /* should never return, handle_affinity_error exits */
	}
	
	/* if the critical section aborted, but not because the core type is */ 
	/* invalid, then give it another shot */
	/* while theoretically possible, this has never been observed to restart */
	/* more than once before either succeeding or failing the check */
	goto restart_sequence;
}

/********************************/
/* Internal component functions */
/********************************/

/* In case headers aren't new enough to have __NR_perf_event_open */
#ifndef __NR_perf_event_open
#define __NR_perf_event_open 298 /* __x86_64__ is the only arch we support */
#endif

__attribute__((weak)) int perf_event_open(struct perf_event_attr *attr, pid_t pid,
										  int cpu, int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/* read PERF_METRICS */
static inline unsigned long long read_metrics(void)
{
	return _rdpmc(TOPDOWN_PERF_METRICS | TOPDOWN_METRIC_COUNTER_TOPDOWN_L1_L2);
}

/* extract the metric defined by event i from the value */
float extract_metric(int i, unsigned long long val)
{
	return (double)(((val) >> (i * 8)) & 0xff) / 0xff;
}

/***********************************************/
/* Required PAPI component interface functions */
/***********************************************/

static int
_topdown_init_component(int cidx)
{
	unsigned long long val;
	int err, i;
	int retval = PAPI_OK;
	int supports_l2;

	char *strCpy;
	char typeStr[PAPI_MIN_STR_LEN];

	const PAPI_hw_info_t *hw_info;

	/* Check for processor support */
	hw_info = &(_papi_hwi_system_info.hw_info);
	switch (hw_info->vendor)
	{
	case PAPI_VENDOR_INTEL:
		break;
	default:
		err = snprintf(_topdown_vector.cmp_info.disabled_reason,
					   PAPI_MAX_STR_LEN, "Not a supported CPU vendor");
		_topdown_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = 0;
		if (err > PAPI_MAX_STR_LEN)
			HANDLE_STRING_ERROR;
		retval = PAPI_ENOSUPP;
		goto fn_fail;
	}

	/* Ideally, we should check the IA32_PERF_CAPABILITIES MSR for */
	/* PERF_METRICS support. However, since doing this requires a */
	/* sysadmin to go through a lot of hassle, it may be better to
	/* just hardcode supported platforms instead */

	if (hw_info->vendor == PAPI_VENDOR_INTEL)
	{
		if (hw_info->cpuid_family != 6)
		{
			/* Not a family 6 machine */
			strCpy = strncpy(_topdown_vector.cmp_info.disabled_reason,
							 "CPU family not supported", PAPI_MAX_STR_LEN);
			_topdown_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = 0;
			if (strCpy == NULL)
				HANDLE_STRING_ERROR;
			retval = PAPI_ENOIMPL;
			goto fn_fail;
		}

		/* Detect topdown support */
		switch (hw_info->cpuid_model)
		{
		/* The model id can be found in Table 2-1 of the */
		/* IA-32 Architectures Software Developer’s Manual */

		/* homogeneous machines that do not support l2 TMA */
		case 0x6a:	/* IceLake 3rd gen Xeon */
		case 0x6c:	/* IceLake 3rd gen Xeon */
		case 0x7d:	/* IceLake 10th gen Core */
		case 0x7e:	/* IceLake 10th gen Core */
		case 0x8c:	/* TigerLake 11th gen Core */
		case 0x8d:	/* TigerLake 11th gen Core */
		case 0xa7:	/* RocketLake 11th gen Core */
			required_core_type = INTEL_CORE_TYPE_HOMOGENEOUS;
			supports_l2 = 0;
			break;

		/* homogeneous machines that support l2 TMA */
		case 0x8f:	/* SapphireRapids 4th gen Xeon */
		case 0xcf:	/* EmeraldRapids 5th gen Xeon */
			required_core_type = INTEL_CORE_TYPE_HOMOGENEOUS;
			supports_l2 = 1;
			break;

		/* hybrid machines that support l2 TMA and are locked to the P-core */
		case 0xaa:	/* MeteorLake Core Ultra 7 hybrid */
		case 0xad:	/* GraniteRapids 6th gen Xeon P-core */
		case 0xae:	/* GraniteRapids 6th gen Xeon P-core */
		case 0x97:	/* AlderLake 12th gen Core hybrid */
		case 0x9a:	/* AlderLake 12th gen Core hybrid */
		case 0xb7:	/* RaptorLake-S/HX 13th gen Core hybrid */
		case 0xba:	/* RaptorLake 13th gen Core hybrid */
		case 0xbd:	/* LunarLake Series 2 Core Ultra hybrid */
		case 0xbf:	/* RaptorLake 13th gen Core hybrid */
			required_core_type = INTEL_CORE_TYPE_PERFORMANCE;
			supports_l2 = 1;
			
			/* if we are on a heterogeneous processor, try and load librseq */
			if (link_librseq() == PAPI_OK) {
        		librseq_loaded = 1;

				/* indicate in desc that librseq was found and is being used */
				err = snprintf(_topdown_vector.cmp_info.description, PAPI_MAX_STR_LEN,
				TOPDOWN_COMPONENT_DESCRIPTION " (librseq in use)");
				_topdown_vector.cmp_info.description[PAPI_MAX_STR_LEN - 1] = 0;
			}

			break;

		default: /* not a supported model */
			strCpy = strncpy(_topdown_vector.cmp_info.disabled_reason,
							 "CPU model not supported", PAPI_MAX_STR_LEN);
			_topdown_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = 0;
			if (strCpy == NULL)
				HANDLE_STRING_ERROR;
			retval = PAPI_ENOIMPL;
			goto fn_fail;
		}
	}

	/* if there is a core type requirement for this platform, check it */
	if (!active_core_type_is(required_core_type) && required_core_type != INTEL_CORE_TYPE_HOMOGENEOUS) {
		core_type_to_name(required_core_type, typeStr);
		err = snprintf(_topdown_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
			"The PERF_EVENT MSR does not exist on this core. Limit process affinity to cores of type '%s' only.", typeStr);
		_topdown_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = 0;
		if (err > PAPI_MAX_STR_LEN)
			HANDLE_STRING_ERROR;
		retval = PAPI_ECMP;
		goto fn_fail;
	}

	/* allocate the events table */
	topdown_native_events = (_topdown_native_event_entry_t *)
		papi_calloc(TOPDOWN_MAX_COUNTERS, sizeof(_topdown_native_event_entry_t));
	if (topdown_native_events == NULL)
	{
		err = snprintf(_topdown_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
				"%s:%i topdown_native_events papi_calloc for %lu bytes failed.",
				__FILE__, __LINE__, TOPDOWN_MAX_COUNTERS * sizeof(_topdown_native_event_entry_t));
		_topdown_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = 0;
		if (err > PAPI_MAX_STR_LEN)
			HANDLE_STRING_ERROR;
		retval = PAPI_ENOMEM;
		goto fn_fail;
	}

	/* fill out the events table */
	i = 0;

	/* level 1 events */
	strcpy(topdown_native_events[i].name, "TOPDOWN_RETIRING_PERC");
	strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were retiring instructions");
	strcpy(topdown_native_events[i].units, "%");
	topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
	topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_RETIRING;
	topdown_native_events[i].selector = i + 1;

	i++;
	strcpy(topdown_native_events[i].name, "TOPDOWN_BAD_SPEC_PERC");
	strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were stalled due to bad speculation");
	strcpy(topdown_native_events[i].units, "%");
	topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
	topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_BAD_SPEC;
	topdown_native_events[i].selector = i + 1;

	i++;
	strcpy(topdown_native_events[i].name, "TOPDOWN_FE_BOUND_PERC");
	strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were waiting on the frontend");
	strcpy(topdown_native_events[i].units, "%");
	topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
	topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_FE_BOUND;
	topdown_native_events[i].selector = i + 1;

	i++;
	strcpy(topdown_native_events[i].name, "TOPDOWN_BE_BOUND_PERC");
	strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were waiting on the backend");
	strcpy(topdown_native_events[i].units, "%");
	topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
	topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_BE_BOUND;
	topdown_native_events[i].selector = i + 1;

	if (supports_l2) {
		/* level 2 events */
		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_HEAVY_OPS_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were retiring heavy operations");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_HEAVY_OPS;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_BR_MISPREDICT_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were wasted due to branch misses");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_BR_MISPREDICT;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_FETCH_LAT_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were stalled due to no uops being issued");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_FETCH_LAT;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_MEM_BOUND_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were stalled due to demand load/store instructions");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = TOPDOWN_METRIC_IDX_MEM_BOUND;
		topdown_native_events[i].selector = i + 1;

		/* derived level 2 events */
		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_LIGHT_OPS_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were retiring light operations");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = -1;
		topdown_native_events[i].derived_parent_idx = TOPDOWN_METRIC_IDX_RETIRING;
		topdown_native_events[i].derived_sibling_idx = TOPDOWN_METRIC_IDX_HEAVY_OPS;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_MACHINE_CLEARS_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were wasted due to pipeline resets");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = -1;
		topdown_native_events[i].derived_parent_idx = TOPDOWN_METRIC_IDX_BAD_SPEC;
		topdown_native_events[i].derived_sibling_idx = TOPDOWN_METRIC_IDX_BR_MISPREDICT;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_FETCH_BAND_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were wasted due to less uops being issued than there are slots");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = -1;
		topdown_native_events[i].derived_parent_idx = TOPDOWN_METRIC_IDX_FE_BOUND;
		topdown_native_events[i].derived_sibling_idx = TOPDOWN_METRIC_IDX_FETCH_LAT;
		topdown_native_events[i].selector = i + 1;

		i++;
		strcpy(topdown_native_events[i].name, "TOPDOWN_CORE_BOUND_PERC");
		strcpy(topdown_native_events[i].description, "The percentage of pipeline slots that were stalled due to insufficient non-memory core resources");
		strcpy(topdown_native_events[i].units, "%");
		topdown_native_events[i].return_type = PAPI_DATATYPE_FP64;
		topdown_native_events[i].metric_idx = -1;
		topdown_native_events[i].derived_parent_idx = TOPDOWN_METRIC_IDX_BE_BOUND;
		topdown_native_events[i].derived_sibling_idx = TOPDOWN_METRIC_IDX_MEM_BOUND;
		topdown_native_events[i].selector = i + 1;
	}

	num_events = i + 1;

	/* Export the total number of events available */
	_topdown_vector.cmp_info.num_native_events = num_events;
	_topdown_vector.cmp_info.num_cntrs = num_events;
	_topdown_vector.cmp_info.num_mpx_cntrs = num_events;

	/* Export the component id */
	_topdown_vector.cmp_info.CmpIdx = cidx;

fn_exit:
	_papi_hwd[cidx]->cmp_info.disabled = retval;
	return retval;
fn_fail:
	goto fn_exit;
}

static int
_topdown_init_thread(hwd_context_t *ctx)
{
	(void)ctx;
	return PAPI_OK;
}

static int
_topdown_init_control_state(hwd_control_state_t *ctl)
{
	_topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

	int retval = PAPI_OK;
	struct perf_event_attr slots, metrics;
	int slots_fd = -1;
	int metrics_fd = -1;
	void *slots_p, *metrics_p;

	// Set up slots
	memset(&slots, 0, sizeof(slots));
	slots.type = PERF_TYPE_RAW;
	slots.size = sizeof(struct perf_event_attr);
	slots.config = 0x0400ull;
	slots.exclude_kernel = 1;

	// Open slots
	slots_fd = perf_event_open(&slots, 0, -1, -1, 0);
	if (slots_fd < 0)
	{
		retval = PAPI_ENOMEM;
		goto fn_fail;
	}

	// Memory mapping the fd to permit _rdpmc calls from userspace
	slots_p = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, slots_fd, 0);
	if (slots_p == (void *) -1L)
	{
		retval = PAPI_ENOMEM;
		goto fn_fail;
	}

	// Set up metrics
	memset(&metrics, 0, sizeof(metrics));
	metrics.type = PERF_TYPE_RAW;
	metrics.size = sizeof(struct perf_event_attr);
	metrics.config = 0x8000;
	metrics.exclude_kernel = 1;

	// Open metrics with slots as the group leader
	metrics_fd = perf_event_open(&metrics, 0, -1, slots_fd, 0);
	if (metrics_fd < 0)
	{
		retval = PAPI_ENOMEM;
		goto fn_fail;
	}

	// Memory mapping the fd to permit _rdpmc calls from userspace
	metrics_p = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, metrics_fd, 0);
	if (metrics_p == (void *) -1L)
	{
		retval = PAPI_ENOMEM;
		goto fn_fail;
	}

	// We set up with no errors, so fill out the control state
	control->slots_fd = slots_fd;
	control->slots_p;
	control->metrics_fd = metrics_fd;
	control->metrics_p;

fn_exit:
	return retval;

fn_fail:
	// We need to close & free whatever we opened and allocated
	if (slots_p != NULL)
		munmap(slots_p, getpagesize());
	if (metrics_p != NULL)
		munmap(metrics_p, getpagesize());
	if (slots_fd >= 0)
		close(slots_fd);
	if (metrics_fd >= 0)
		close(metrics_fd);
	goto fn_exit;
}

static int
_topdown_update_control_state(hwd_control_state_t *ctl,
							  NativeInfo_t *native,
							  int count,
							  hwd_context_t *ctx)
{
	(void)ctx;
	_topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

	int i;
	for (i = 0; i < TOPDOWN_MAX_COUNTERS; i++)
	{
		control->being_measured[i] = 0;
	}

	for (i = 0; i < count; i++)
	{
		int index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
		native[i].ni_position = topdown_native_events[index].selector - 1;
		control->being_measured[index] = 1;
	}

	return PAPI_OK;
}

/**@class _wrapper_perf_event_ioc_reset
 * @brief Reset the PERF_METRICS counter and slots to maintain precision
 *        as perf the recommendation of section 21.3.9.3 of the IA-32
 *        Architectures Software Developer's Manual. Resetting means we
 *        do not need to record 'before' metrics/slot values, as they are
 *        always effectively 0. Despite the reset meaning we don't need to
 *        record the slots value at all, the SDM states that SLOTS and the
 *        PERF_METRICS MSR should be reset together.
*/
static
int _wrapper_perf_event_ioc_reset(_topdown_control_state_t *control)
{
        // This ioctl call does not need to be protected by assert_affinity()
        int retval = ioctl(control->slots_fd, PERF_EVENT_IOC_RESET, 0);
        if (retval != 0) {
                SUBDBG("Resetting control->slots_fd failed.\n");
                return PAPI_ESYS;
        }

        // This ioctl call does not need to be protected by assert_affinity()
        retval = ioctl(control->metrics_fd, PERF_EVENT_IOC_RESET, 0);
        if (retval != 0) {
                SUBDBG("Resetting control->metrics_fd failed.\n");
                return PAPI_ESYS;
        }

        return PAPI_OK;
}

/**@class _wrapper_perf_event_ioc_enable
 * @brief Enable the event's.
 *
 *        NOTE: As control->slots_fd is the group leader for control->metrics_fd
 *        you could use PERF_IOC_FLAG_GROUP in the ioctl argument to enable all events
 *        in the group i.e. control->metrics_fd. However, from Linux 2.6.31 to Linux 3.4
 *        PERF_IOC_FLAG_GROUP ioctl argument was broken.
*/
static
int _wrapper_perf_event_ioc_enable(_topdown_control_state_t *control)
{
        int retval = ioctl(control->slots_fd, PERF_EVENT_IOC_ENABLE, 0);
        if (retval != PAPI_OK) {
		SUBDBG("Enabling control->slots_fd failed.\n");
                return retval;
        }

        retval = ioctl(control->metrics_fd, PERF_EVENT_IOC_ENABLE, 0);
        if (retval != PAPI_OK) {
		SUBDBG("Enabling control->metrics_fd failed.\n");
                return retval;
        }

        return PAPI_OK;
}

static int
_topdown_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
	(void)ctx;
	_topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

        // Reset counters to maintain precision.
        int retval = _wrapper_perf_event_ioc_reset(control);
        if (retval != PAPI_OK) {
                return retval;
        }

	retval = _wrapper_perf_event_ioc_enable(control);
	if (retval != PAPI_OK) {
		return retval;
	}

	return PAPI_OK;
}

static int
_topdown_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
	       long long **events, int flags)
{
        (void)flags;
        (void)ctx;
        _topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

        // Extract the values
        unsigned long long metrics_after = read_metrics();
        int i;
        for (i = 0; i < TOPDOWN_MAX_COUNTERS; i++)
        {
                if (control->being_measured[i])
                {
                        double perc;
                        // Handle case where the metric is not derived
                        if (topdown_native_events[i].metric_idx >= 0)
                        {
                                perc = extract_metric(topdown_native_events[i].metric_idx,
                                        metrics_after) * 100.0;
                        }
                        else // Handle case where the metric is derived
                        {
                                // Metric perc = parent perc - sibling perc
                                perc = extract_metric(
                                        topdown_native_events[i].derived_parent_idx,
                                        metrics_after) * 100.0
                                        - extract_metric(
                                        topdown_native_events[i].derived_sibling_idx,
                                        metrics_after) * 100.0;
                        }

                        // Sometimes the percentage will be a very small negative value
                        // instead of 0 due to floating point error. tidy that up:
                        if (perc < 0.0) {
                                perc = 0.0;
                        }

                        // Store the raw bits of the double into the counter value
                        control->count[i] = *(long long*)&perc;
                }
        }
        // Pass back a pointer to our results
        *events = control->count;

        return PAPI_OK;
}

static int
_topdown_reset(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
        (void)ctx;
        _topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

        int retval = _wrapper_perf_event_ioc_reset(control);
        if (retval != PAPI_OK) {
                return retval;
        }

        return PAPI_OK;
}

static int
_topdown_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
	(void)ctx;
	_topdown_control_state_t *control = (_topdown_control_state_t *)ctl;

        // slots_fd is defined as the group leader in _topdown_init_control_state;
        // therefore, disabling slots_fd will disable the entire group.
	int retval = ioctl(control->slots_fd, PERF_EVENT_IOC_DISABLE, 0);
	if (retval != 0) {
		SUBDBG("Disabling the entire event group failed.\n");
		return PAPI_ESYS;
	}

	return PAPI_OK;
}

static int
_topdown_shutdown_component(void)
{
	/* Free anything we allocated */
	papi_free(topdown_native_events);

	return PAPI_OK;
}

static int
_topdown_shutdown_thread(hwd_context_t *ctx)
{
	( void ) ctx;

	return PAPI_OK;
}

static int
_topdown_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
	( void ) ctx;
	( void ) code;
	( void ) option;

	return PAPI_OK;
}

static int
_topdown_set_domain(hwd_control_state_t *cntrl, int domain)
{
	(void) cntrl;
	(void) domain;

	return PAPI_OK;
}

static int
_topdown_ntv_enum_events(unsigned int *EventCode, int modifier)
{

	int index;

	switch (modifier)
	{
	case PAPI_ENUM_FIRST:
		/* return the first event that we support */
		*EventCode = 0;
		return PAPI_OK;

	case PAPI_ENUM_EVENTS:
		index = *EventCode;
		/* Make sure we have at least 1 more event after us */
		if (index < num_events - 1)
		{
			/* This assumes a non-sparse mapping of the events */
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		}
		else
		{
			return PAPI_ENOEVNT;
		}
		break;

	default:
		return PAPI_EINVAL;
	}

	return PAPI_EINVAL;
}

static int
_topdown_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
	int index = EventCode & PAPI_NATIVE_AND_MASK;

	if (index >= 0 && index < num_events)
	{
		strncpy(name, topdown_native_events[index].name, len);
		return PAPI_OK;
	}

	return PAPI_ENOEVNT;
}

static int
_topdown_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
	int index = EventCode;

	if (index >= 0 && index < num_events)
	{
		strncpy(descr, topdown_native_events[index].description, len);
		return PAPI_OK;
	}
	return PAPI_ENOEVNT;
}

static int
_topdown_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{

	int index = EventCode;

	if ((index < 0) || (index >= num_events))
		return PAPI_ENOEVNT;

	strncpy(info->symbol, topdown_native_events[index].name, 
			sizeof(info->symbol) - 1);
	info->symbol[sizeof(info->symbol) - 1] = '\0';

	strncpy(info->long_descr, topdown_native_events[index].description, 
			sizeof(info->long_descr) - 1);
	info->long_descr[sizeof(info->long_descr) - 1] = '\0';

	strncpy(info->units, topdown_native_events[index].units, 
			sizeof(info->units) - 1);
	info->units[sizeof(info->units) - 1] = '\0';

	info->data_type = topdown_native_events[index].return_type;

	return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _topdown_vector = {
	.cmp_info = {
		.name = "topdown",
		.short_name = "topdown",
		.description = TOPDOWN_COMPONENT_DESCRIPTION,
		.version = "1.0",
		.support_version = "n/a",
		.kernel_version = "n/a",
		.default_domain = PAPI_DOM_USER,
		.available_domains = PAPI_DOM_USER,
		.default_granularity = PAPI_GRN_THR,
		.available_granularities = PAPI_GRN_THR,
		.hardware_intr_sig = PAPI_INT_SIGNAL,
	},

	/* Sizes of framework-opaque component-private structures */
	.size = {
		.context = sizeof(_topdown_context_t),
		.control_state = sizeof(_topdown_control_state_t),
		.reg_value = 1, /* unused */
		.reg_alloc = 1, /* unused */
	},

	/* Used for general PAPI interactions */
	.start = _topdown_start,
	.stop = _topdown_stop,
	.read = _topdown_read,
	.reset = _topdown_reset,
	.init_component = _topdown_init_component,
	.init_thread = _topdown_init_thread,
	.init_control_state = _topdown_init_control_state,
	.update_control_state = _topdown_update_control_state,
	.ctl = _topdown_ctl,
	.shutdown_thread = _topdown_shutdown_thread,
	.shutdown_component = _topdown_shutdown_component,
	.set_domain = _topdown_set_domain,

	/* Name Mapping Functions */
	.ntv_enum_events = _topdown_ntv_enum_events,
	.ntv_code_to_name = _topdown_ntv_code_to_name,
	.ntv_code_to_descr = _topdown_ntv_code_to_descr,
	.ntv_code_to_info = _topdown_ntv_code_to_info,
};
