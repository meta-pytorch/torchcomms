#ifndef __CUPTI_EVENT_AND_METRIC_H__
#define __CUPTI_EVENT_AND_METRIC_H__

#include "cupti_utils.h"
#include "papi_cupti_common.h"
#include "cupti_config.h"

#include <stdint.h>

typedef struct cuptie_control_s     *cuptie_control_t;

// Interfaces to handle initialization and shutdown //
int cuptie_init(void);
int cuptie_shutdown(void);

// Interfaces to handle native events //
int cuptie_evt_enum(uint32_t *event_code, int modifier); 
int cuptie_evt_code_to_descr(uint32_t event_code, char *descr, int len);
int cuptie_evt_name_to_code(const char *name, uint32_t *event_code);
int cuptie_evt_code_to_name(uint32_t event_code, char *name, int len);
int cuptie_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info);

// Interfaces to handling profiling //
int cuptie_ctx_create(cuptic_info_t thr_info, cuptie_control_t *pstate, uint32_t *eventIds, int num_events);
int cuptie_ctx_start(cuptie_control_t ctl);
int cuptie_ctx_read(cuptie_control_t ctl, long long **counterValues);
int cuptie_ctx_reset(cuptie_control_t ctl);
int cuptie_ctx_stop(cuptie_control_t ctl);
int cuptie_ctx_destroy(cuptie_control_t *pctl);

typedef enum
{
    EVENT_OR_METRIC_API = 0,
    EITHER_API,
    PERFWORKS_API,
} required_api_for_a_cuda_device;

typedef enum 
{
   EVENT = 0,
   METRIC,
} cupti_event_or_metric;

typedef struct gpu_record_event_and_metric_s {
    char chipName[PAPI_MIN_STR_LEN];
    CUdevice deviceHandle;
} gpu_record_event_and_metric_t;

typedef struct event_and_metric_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_HUGE_STR_LEN];
    cupti_event_or_metric api;
    cuptiu_bitmap_t device_map;
} cuptiu_event_and_metric_t;

typedef struct event_and_metric_table_s {
    unsigned int count;
    unsigned int capacity;
    CUpti_EventGroupSets *eventGroupSets;
    CUpti_MetricID metricIDs[PAPI_CUDA_MAX_COUNTERS];
    int *idsThatMakeupAUserAddedEventArray[PAPI_CUDA_MAX_COUNTERS];
    int *cumulativeValuesArray[PAPI_CUDA_MAX_COUNTERS];
    int totalNumberOfIdsThatMakeupTheUserAddedEventArray[PAPI_CUDA_MAX_COUNTERS];
    int totalNumberOfUserAddedNativeEvents;
    cupti_event_or_metric userAddedIdsCorrespondingApiArray[PAPI_CUDA_MAX_COUNTERS];
    uint64_t startTimeStampNs;
    gpu_record_event_and_metric_t *avail_gpu_info;
    cuptiu_event_and_metric_t *events;
    void *htable;
} cuptiu_event_and_metric_table_t;

#endif  /* __CUPTI_EVENT_AND_METRIC_H__ */
