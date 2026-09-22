#define TOPDOWN_COMPONENT_DESCRIPTION	"A component for accessing topdown " \
									"metrics on 10th gen+ Intel processors"

/* these MSR access defines are constant based on the assumptoin that */
/* new architectures will not change them */
#define TOPDOWN_PERF_FIXED	(1 << 30)	/* return fixed counters */
#define TOPDOWN_PERF_METRICS	(1 << 29)	/* return metric counters */

#define TOPDOWN_FIXED_COUNTER_SLOTS		        3
#define TOPDOWN_METRIC_COUNTER_TOPDOWN_L1_L2	0

/* L1 Topdown indices in the PERF_METRICS counter */
#define TOPDOWN_METRIC_IDX_RETIRING     0
#define TOPDOWN_METRIC_IDX_BAD_SPEC     1
#define TOPDOWN_METRIC_IDX_FE_BOUND     2
#define TOPDOWN_METRIC_IDX_BE_BOUND     3

/* L2 Topdown indices in the PERF_METRICS counter */
/* The L2 events not here are derived from the others */
#define TOPDOWN_METRIC_IDX_HEAVY_OPS        4
#define TOPDOWN_METRIC_IDX_BR_MISPREDICT    5
#define TOPDOWN_METRIC_IDX_FETCH_LAT        6
#define TOPDOWN_METRIC_IDX_MEM_BOUND        7

/** Holds per event information */
typedef struct topdown_native_event_entry
{
	int selector; /* signifies which counter slot is being used. indexed from 1 */

	char name[PAPI_MAX_STR_LEN];
	char description[PAPI_MAX_STR_LEN];
	char units[PAPI_MIN_STR_LEN]; /* the unit to use for this event */
	int return_type; /* the PAPI return type to use for this event */

	int metric_idx; /* index in PERF_METRICS. if -1, it's derived */
	int derived_parent_idx; /* if derived, which parent do we subtract from */
	int derived_sibling_idx; /* if derived, which metric do we subtract */

} _topdown_native_event_entry_t;

/** Holds per event-set information */
typedef struct topdown_control_state
{
#define TOPDOWN_MAX_COUNTERS    16
	int being_measured[TOPDOWN_MAX_COUNTERS];
	long long count[TOPDOWN_MAX_COUNTERS];

	int slots_fd; /* file descriptor for the slots fixed counter */
	void *slots_p; /* we need this in ctl so it can be freed */
	unsigned long long slots_before;
	int metrics_fd; /* file descriptor for the PERF_METRICS counter */
	void *metrics_p; /* we need this in ctl so it can be freed */
	unsigned long long metrics_before;
} _topdown_control_state_t;

/* these MSR access defines are constant based on the assumptoin that */
/* new architectures will not change them */
#define TOPDOWN_PERF_FIXED	(1 << 30)	/* return fixed counters */
#define TOPDOWN_PERF_METRICS	(1 << 29)	/* return metric counters */

#define TOPDOWN_FIXED_COUNTER_SLOTS		        3
#define TOPDOWN_METRIC_COUNTER_TOPDOWN_L1_L2	0

/* L1 Topdown indices in the PERF_METRICS counter */
#define TOPDOWN_METRIC_IDX_RETIRING     0
#define TOPDOWN_METRIC_IDX_BAD_SPEC     1
#define TOPDOWN_METRIC_IDX_FE_BOUND     2
#define TOPDOWN_METRIC_IDX_BE_BOUND     3

/* L2 Topdown indices in the PERF_METRICS counter */
/* The L2 events not here are derived from the others */
#define TOPDOWN_METRIC_IDX_HEAVY_OPS        4
#define TOPDOWN_METRIC_IDX_BR_MISPREDICT    5
#define TOPDOWN_METRIC_IDX_FETCH_LAT        6
#define TOPDOWN_METRIC_IDX_MEM_BOUND        7

/* Holds per thread information; however, we do not use this structure,
   but the framework still needs its size */
typedef struct topdown_context
{
    int junk;
} _topdown_context_t;
