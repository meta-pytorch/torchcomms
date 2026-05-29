#ifndef __LINUX_NVML_H__
#define __LINUX_NVML_H__

#include "nvml.h"

#define FEATURE_CLOCK_INFO          1
#define FEATURE_ECC_LOCAL_ERRORS    2
#define FEATURE_FAN_SPEED           4
#define FEATURE_MAX_CLOCK           8
#define FEATURE_MEMORY_INFO         16
#define FEATURE_PERF_STATES         32
#define FEATURE_POWER               64
#define FEATURE_TEMP                128
#define FEATURE_ECC_TOTAL_ERRORS    256
#define FEATURE_UTILIZATION         512
#define FEATURE_POWER_MANAGEMENT 1024
#define FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN 2048
#define FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX 4096

#define HAS_FEATURE( features, query ) ( features & query )

#define MEMINFO_TOTAL_MEMORY    0
#define MEMINFO_UNALLOCED       1
#define MEMINFO_ALLOCED         2

#define LOCAL_ECC_REGFILE       0
#define LOCAL_ECC_L1            1
#define LOCAL_ECC_L2            2
#define LOCAL_ECC_MEM           3

#define GPU_UTILIZATION         0
#define MEMORY_UTILIZATION      1

/* we lookup which card we are on at read time; this is a place holder */
typedef int nvml_register_t;

struct local_ecc {
    nvmlEccBitType_t bits;
    int which_one;
};

typedef union {
    nvmlClockType_t clock; /* used in get[Max]ClockSpeed */
    struct local_ecc ecc_opts; /* local ecc errors, total ecc errors */
    int which_one; /* memory_info , utilization*/
} nvml_resource_options_t;

typedef struct nvml_native_event_entry {
    nvml_resource_options_t options;
    char name[PAPI_MAX_STR_LEN];
    char units[PAPI_MIN_STR_LEN];
    char description[PAPI_MAX_STR_LEN];
    int type;
} nvml_native_event_entry_t;

#endif
