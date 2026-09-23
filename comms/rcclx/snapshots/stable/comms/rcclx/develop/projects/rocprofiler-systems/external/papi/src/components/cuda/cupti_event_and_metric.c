#include <dlfcn.h>
#include <stdbool.h>

#include "papi.h"
#include "papi_debug.h"
#include "cupti_event_and_metric.h"
#include "papi_cupti_common.h"
#include "htable.h"

#include "cupti_events.h"
#include "cupti_metrics.h"
#include "cupti_config.h"

#include <cupti_activity.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * Event identifier encoding format:
 * +---------------------------------+-------+----+------------+
 * |         unused                  |  dev  | ql |   nameid   |
 * +---------------------------------+-------+----+------------+
 *
 * unused    : 34 bits 
 * device    : 7  bits ([0 - 127] devices)
 * qlmask    : 2  bits (qualifier mask)
 * nameid    : 21: bits (roughly > 2 million event names)
 */
#define EVENTS_WIDTH (sizeof(uint32_t) * 8)
#define DEVICE_WIDTH ( 7)
#define QLMASK_WIDTH ( 2) 
#define NAMEID_WIDTH (21)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - DEVICE_WIDTH)
#define QLMASK_SHIFT (DEVICE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define DEVICE_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define DEVICE_FLAG  (0x1)

typedef struct {
    int device;
    int flags;
    int nameid;
} event_info_t;

typedef struct cuptie_gpu_state_s {
    int                    deviceIdx;
    cuptiu_event_and_metric_table_t  *added_events;
} cuptie_gpu_state_t;

struct cuptie_control_s {
    cuptie_gpu_state_t *gpu_ctl;
    long long           *counters;
    int                 read_count;
    int                 running;
    cuptic_info_t       info;
};

// Number of devices that are currently on the machine, this includes CCs >= 7.0 and CCs <= 7.0
static int numDevicesOnMachine;

// Main table that holds event and metrics
static cuptiu_event_and_metric_table_t *cuptiu_table_p;

// CUPTI Profiler API function pointers //
CUptiResult ( *cuptiProfilerInitializeEventAndMetricPtr ) (CUpti_Profiler_Initialize_Params* params);
CUptiResult ( *cuptiProfilerDeInitializeEventAndMetricPtr ) (CUpti_Profiler_DeInitialize_Params* params);

// Event API function pointers //
// Enumeration
CUptiResult (*cuptiDeviceGetNumEventDomainsPtr) (CUdevice device, uint32_t *numDomains);
CUptiResult (*cuptiDeviceEnumEventDomainsPtr) (CUdevice device, size_t *arraySizeBytes, CUpti_EventDomainID *domainArray);
CUptiResult (*cuptiEventDomainGetNumEventsPtr) (CUpti_EventDomainID eventDomain, uint32_t *numEvents);
CUptiResult (*cuptiEventDomainEnumEventsPtr) (CUpti_EventDomainID eventDomain, size_t *sarraySizeBytes, CUpti_EventID *eventArray);
CUptiResult (*cuptiEventGetAttributePtr) (CUpti_EventID event, CUpti_EventAttribute attrib, size_t *valueSize, void *value);
// Managing an EventGroup
CUptiResult (*cuptiEventGetIdFromNamePtr) (CUdevice device, const char *eventName, CUpti_EventID *event);
CUptiResult (*cuptiEventGroupSetsCreatePtr) (CUcontext context, size_t eventIdArraySizeBytes, CUpti_EventID *eventIdArray, CUpti_EventGroupSets **eventGroupPasses);
CUptiResult (*cuptiEventGroupSetEnablePtr) (CUpti_EventGroupSet *eventGroupSet);
CUptiResult (*cuptiEventGroupSetAttributePtr) (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t valueSize, void *value);
CUptiResult (*cuptiEventGroupGetAttributePtr) (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiDeviceGetEventDomainAttributePtr) (CUdevice device, CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiEventGroupResetAllEventsPtr) (CUpti_EventGroup eventGroup);
CUptiResult (*cuptiEventGroupSetDisablePtr) (CUpti_EventGroupSet *eventGroupSet);
CUptiResult (*cuptiEventGroupSetsDestroyPtr) (CUpti_EventGroupSets *eventGroupSets);
// Evaluation
CUptiResult (*cuptiEventGroupReadAllEventsPtr) (CUpti_EventGroup eventGroup, CUpti_ReadEventFlags flags, size_t *eventValueBufferSizeBytes, uint64_t *eventValueBuffer, size_t *eventIdArraySizeBytes, CUpti_EventID *eventIdArray, size_t *numEventIdsRead);

// Metric API function pointers //
// Enumeration
CUptiResult (*cuptiDeviceGetNumMetricsPtr) (CUdevice device, uint32_t *numMetrics);
CUptiResult (*cuptiDeviceEnumMetricsPtr) (CUdevice device, size_t *arraySizeBytes, CUpti_MetricID *metricArray);
CUptiResult (*cuptiMetricGetAttributePtr) (CUpti_MetricID metric, CUpti_MetricAttribute attrib, size_t *valueSize, void *value);
// Managing an EventGroup
CUptiResult (*cuptiMetricGetIdFromNamePtr) (CUdevice device, const char *metricName, CUpti_MetricID *metric);
CUptiResult (*cuptiMetricCreateEventGroupSetsPtr) (CUcontext context, size_t metricIdArraySizeBytes, CUpti_MetricID *metricIdArray, CUpti_EventGroupSets **eventGroupPasses);
// Evaluation
CUptiResult (*cuptiMetricGetNumEventsPtr) (CUpti_MetricID metric, uint32_t *numEvents);
CUptiResult (*cuptiMetricEnumEventsPtr) (CUpti_MetricID metric, size_t *eventIdArraySizeBytes, CUpti_EventID *eventIdArray);
CUptiResult (*cuptiGetTimestampPtr) (uint64_t* timestamp); // Apart of the Activity API, but allows us to get the timeDuration for cuptiMetricGetValue
CUptiResult (*cuptiMetricGetValuePtr) (CUdevice device, CUpti_MetricID metric, size_t eventIdArraySizeBytes, CUpti_EventID *eventIdArray, size_t eventValueArraySizeBytes, uint64_t *eventValueArray, uint64_t timeDuration, CUpti_MetricValue *metricValue);


// Helper functions for CUPTI Profiler API
static int initialize_cupti_profiler_api(void);
static int deinitialize_cupti_profiler_api(void);

// Helper functions related to loading the function CUPTI function pointers
static int load_event_and_metric_sym(void);
static void unload_event_and_metric_sym(void);
static int load_cupti_profiler_sym(void);
static void unload_cupti_profiler_sym(void);

// Helper functions for the Event and Metric APIs //
// Enumeration
static int enumerate_events_for_event_api(cuptiu_event_and_metric_table_t *table, CUdevice device, int deviceIdx);
static int enumerate_metrics_for_metric_api(cuptiu_event_and_metric_table_t *table, CUdevice device, int deviceIdx);
static int find_same_chipname(int gpu_id);
static int assign_chipnames_for_a_device_index(void);
static int determine_required_api(int deviceIdx);
// Managing an EventGroup
static int check_if_event_or_metric_required_multiple_passes(const char *addedEventName, int cuptiApi, int deviceIdx, uint32_t *addedNativeEventID);
// Evaluation
static int convert_metric_value_to_long_long(CUpti_MetricID metricID, CUpti_MetricValue metricValue, long long *conversionOfMetricValue);

// Helper functions related to the native event interface - event and metric
static int store_event_or_metric_ntv_events(cuptiu_event_and_metric_table_t *evt_table, const char *evt_name, const char *evt_desc, int deviceIdx, int api);
static int verify_user_added_event_or_metric(uint32_t *events_id, int num_events, cuptie_control_t state, cuptic_info_t thr_info);
static int event_and_metric_id_to_info(uint32_t event_id, event_info_t *info);
static int event_and_metric_id_create(event_info_t *info, uint32_t *event_id);
static int evt_name_to_device(const char *name, int *device);
static int evt_name_to_basename(const char *name, char *base, int len);

// Helper functions related to the Cuda component hash tables - event and metric
static int init_event_and_metric_main_htable(void);
static int init_event_and_metric_table(void);
static int create_event_and_metric_table(int totalNumberOfEntries, cuptiu_event_and_metric_table_t **initializedTable);
static void destroy_event_and_metric_table(cuptiu_event_and_metric_table_t **initializedTable);
static void shutdown_event_table(void);

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section Functions related to the overall initialization of the Event and
 *           Metric workflow.
 *
 *  @{
 */

/** @class cuptie_init
  * @brief Load function pointers, initialize CUPTI profiler, and initialize hash tables.
*/
int cuptie_init(void)
{
    SUBDBG("ENTERING: Initializing the Cuda component for CUPTI Event and Metric API support.\n");
    int papi_errno;

    int maxSupportedEventAndMetricCudaToolkit = 13000;
    int cudaRuntimeVersion;
    cudaArtCheckErrors( cudaRuntimeGetVersionPtr(&cudaRuntimeVersion), return PAPI_EMISC );
    if (cudaRuntimeVersion >= 13000) {
        cuptic_err_set_last("Event and Metric API support has been dropped by NVIDIA in Cuda Toolkit 13.\n");
        return PAPI_ECMP;
    }

    // Load Event and Metric API
    papi_errno = load_event_and_metric_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to load Event and Metric API functions.\n");
        return papi_errno;
    }

    // Load CUPTI Profiler API
    papi_errno = load_cupti_profiler_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to load CUPTI Profiler API functions.\n");
        return papi_errno;
    }

    // Get the number of devices on the system
    papi_errno = cuptic_device_get_count(&numDevicesOnMachine);
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to get the number of devices on the system.\n");
        return PAPI_EMISC;
    }

    if (numDevicesOnMachine < 0) {
        cuptic_err_set_last("For the current system, no NVIDIA devices detected.\n");
        return PAPI_ECMP;
    }

    // Initialize cupti profiler api such that we can get chip names
    papi_errno = initialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failed to initialize the CUPTI Profiler API. A possible reason is a mismatched Cuda Toolkit and NVIDIA architecture."
                            " Try setting PAPI_CUDA_ROOT to an earlier Cuda Toolkit version.");
        return PAPI_EMISC;
    }

    // Initialize the main hash table
    papi_errno = init_event_and_metric_main_htable();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to allocate memory for the main hash table.\n");
        return papi_errno;
    }

    // For each device assign the chipnames
    papi_errno = assign_chipnames_for_a_device_index();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to assign chipname's for each device on the system.\n");
        return papi_errno;
    }

    // Enumerate through available devices and store the events and metrics 
    papi_errno = init_event_and_metric_table();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failure to get the event's and metric's for each device on the system.\n");
        return papi_errno;
    }

    // Initialize the Cuda driver API
    CUresult cuError = cuInitPtr(0);
    if (cuError != CUDA_SUCCESS) {
        cuptic_err_set_last("Failure to initialize the Cuda driver API.\n");
        return PAPI_EMISC;
    }

    SUBDBG("EXITING: Initialization for the CUPTI Event and Metric API completed.\n");
    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section Helper functions related to the overall initialization of 
 *           the Event and Metric workflow.
 *
 *  @{
 */

/** @class load_event_and_metric_sym
  * @brief Load Event and Metric API functions.
*/
static int load_event_and_metric_sym(void)
{
    SUBDBG("ENTERING: Loading Event and Metric API functions.\n");

    if (dl_cupti == NULL) {
        return PAPI_EMISC;
    }

    // Event API  //
    // Enumeration
    cuptiDeviceGetNumEventDomainsPtr      = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetNumEventDomains");
    cuptiDeviceEnumEventDomainsPtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceEnumEventDomains");
    cuptiEventDomainGetNumEventsPtr       = DLSYM_AND_CHECK(dl_cupti, "cuptiEventDomainGetNumEvents");
    cuptiEventDomainEnumEventsPtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiEventDomainEnumEvents");
    cuptiEventGetAttributePtr             = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGetAttribute");
    // Managing an EventGroup
    cuptiEventGetIdFromNamePtr            = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGetIdFromName");
    cuptiEventGroupSetsCreatePtr          = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetsCreate");
    cuptiEventGroupSetEnablePtr           = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetEnable");
    cuptiEventGroupSetAttributePtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetAttribute");
    cuptiEventGroupGetAttributePtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupGetAttribute");
    cuptiDeviceGetEventDomainAttributePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetEventDomainAttribute");
    cuptiEventGroupResetAllEventsPtr      = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupResetAllEvents");
    cuptiEventGroupSetDisablePtr          = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetDisable");
    cuptiEventGroupSetsDestroyPtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetsDestroy");
    // Reading event values
    cuptiEventGroupReadAllEventsPtr       = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupReadAllEvents");

    // Metric API //
    // Enumeration
    cuptiDeviceGetNumMetricsPtr           = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetNumMetrics");
    cuptiDeviceEnumMetricsPtr             = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceEnumMetrics");
    cuptiMetricGetAttributePtr            = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetAttribute");
    // Managing an EventGroup
    cuptiMetricGetIdFromNamePtr           = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetIdFromName");
    cuptiMetricCreateEventGroupSetsPtr    = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricCreateEventGroupSets");
    // Reading metric values
    cuptiMetricGetNumEventsPtr            = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetNumEvents");
    cuptiMetricEnumEventsPtr              = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricEnumEvents");
    cuptiGetTimestampPtr                  = DLSYM_AND_CHECK(dl_cupti, "cuptiGetTimestamp");
    cuptiMetricGetValuePtr                = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetValue");

    SUBDBG("EXITING: Completed loading Event and Metric API functions.\n");
    return PAPI_OK;
}

/** @class load_cupti_profiler_sym
  * @brief Load CUPTI Profiler API functions.
*/
static int load_cupti_profiler_sym(void) 
{
    SUBDBG("ENTERING: Loading CUPTI Profiler API functions.\n");

    cuptiProfilerInitializeEventAndMetricPtr            = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializeEventAndMetricPtr          = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");

    SUBDBG("EXITING: Completed loading CUPTI Profiler API functions.\n");
    return PAPI_OK;
}


/** @class initialize_cupti_profiler_api
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerInitialize.
*/
static int initialize_cupti_profiler_api(void)
{   
    SUBDBG("ENTERING: Initializing CUPTI Profiler API.\n");
    
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    profilerInitializeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerInitializeEventAndMetricPtr(&profilerInitializeParams), return PAPI_EMISC );
    
    SUBDBG("EXITING: Initialization of the CUPTI Profiler API completed.\n");
    return PAPI_OK;
}

/** @class init_event_and_metric_htable
 *  @brief Initialize the main htable used to collect metrics.
*/
static int init_event_and_metric_main_htable(void)
{
    SUBDBG("ENTERING: Initializing event and metric hash table.\n");

    int i, val = 1, base = 2;

    /* allocate (2 ^ NAMEID_WIDTH) metric names, this matches the 
       number of bits for the event encoding format */
    for (i = 0; i < NAMEID_WIDTH; i++) {
        val *= base;
    }

    /* initialize struct */
    cuptiu_table_p = (cuptiu_event_and_metric_table_t *) malloc(sizeof(cuptiu_event_and_metric_table_t));
    if (cuptiu_table_p == NULL) {
        SUBDBG("Failed to allocate memroy for cuptiu_table_p.\n");
        return PAPI_ENOMEM;
    }

    cuptiu_table_p->capacity = val;
    cuptiu_table_p->count = 0;

    cuptiu_table_p->events = (cuptiu_event_and_metric_t *) calloc(val, sizeof(cuptiu_event_and_metric_t));
    if (cuptiu_table_p->events == NULL) {
        SUBDBG("Failed to allocate memory for cuptiu_table_p->events.\n");
        return PAPI_ENOMEM;
    }

    cuptiu_table_p->avail_gpu_info = (gpu_record_event_and_metric_t *) calloc(numDevicesOnMachine, sizeof(gpu_record_event_and_metric_t));
    if (cuptiu_table_p->avail_gpu_info == NULL) {
        SUBDBG("Failed to allocate memory for cuptiu_table_p->avail_gpu_info.\n");
        return PAPI_ENOMEM;
    }

    /* initialize the main hash table for metric collection */
    int papi_errno = htable_init(&cuptiu_table_p->htable);
    if (papi_errno != PAPI_OK) {
        SUBDBG("Failed to initialize main hash table cuptiu_table_p->htable.\n");
        return PAPI_ENOMEM;
    }

    SUBDBG("EXITING: Initialization of event and metric hash table completed.\n");
    return PAPI_OK;
}

/** @class assign_chipnames_for_a_device_index
  * @brief For each device on the system get and assign the chipname.
*/
static int assign_chipnames_for_a_device_index(void)
{
    SUBDBG("ENTERING: Assigning a chipname for each device index.\n");

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        char chipName[PAPI_MIN_STR_LEN];
        int papi_errno = get_chip_name(deviceIdx, chipName);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        int strLen = snprintf(cuptiu_table_p->avail_gpu_info[deviceIdx].chipName, PAPI_MIN_STR_LEN, "%s", chipName);
        if (strLen < 0 || strLen >= PAPI_MIN_STR_LEN) {
             SUBDBG("Failed to fully write chip name.\n");
             return PAPI_EBUF;
        }
    }

    SUBDBG("EXITING: Assigning chipnames for each device index completed.\n");
    return PAPI_OK;
}

/** @class init_event_and_metric_table
  * @brief For each device with cc <= 7.0 enumerate through the events and metrics
  *        and store the names with their corresponding descriptions.
*/
static int init_event_and_metric_table(void)
{
    SUBDBG("ENTERING: Adding events and metrics from the Event and Metric APIs to the hash table.\n");

    // Loop through all of the available devices on the machine
    int deviceIdx, tableIdx = 0;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        int cuptiApi = determine_required_api(deviceIdx);
        // Cuda call failed in determine_required_api; therefore, we exit
        if (cuptiApi < 0) {
            return PAPI_EMISC;
        }
        // Skip devices that will require the Perfworks API to be profiled
        else if (cuptiApi == PERFWORKS_API) {
            continue;
        }

        int found = find_same_chipname(deviceIdx);
        // Unique device found, collect metadata
        if (found == -1) {
            // Increment table index
            if (deviceIdx > 0)
                tableIdx++;

            // Get the handle for the current compute device 
            CUdevice device;
            cudaCheckErrors( cuDeviceGetPtr(&device, deviceIdx), return PAPI_EMISC );
            // Store the handle to be used later with 
            cuptiu_table_p->avail_gpu_info[tableIdx].deviceHandle = device;
        }
        // Metadata already collected for device
        else {
            // Set tableIdx to existing device metadata was collected for
            tableIdx = found;
        }

        // For the Events API, get and store the events and their corresponding descriptions
        int papi_errno = enumerate_events_for_event_api(cuptiu_table_p, cuptiu_table_p->avail_gpu_info[tableIdx].deviceHandle, deviceIdx);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        // For the Metrics API, get and store the metrics and their corresponding descriptions
        papi_errno = enumerate_metrics_for_metric_api(cuptiu_table_p, cuptiu_table_p->avail_gpu_info[tableIdx].deviceHandle, deviceIdx);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    SUBDBG("EXITING: Successfully added events and metrics to the hash table.\n");
    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section Helper functions related to the enumeration of events found in
 *           init_event_and_metric_table.
 *
 *  @{
 */

/** @class determine_required_api
  * @brief For a device index, determine the CC and the corresponding
  *        API needed to profile.
  *
  * @param deviceIdx
  *   The device index..
*/
static int determine_required_api(int deviceIdx)
{   
    SUBDBG("ENTERING: Determing the device compute capability major.\n");
    int cc;
    int papi_errno = get_gpu_compute_capability(deviceIdx, &cc);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    
    if (cc > 70) {
        return PERFWORKS_API;
    }
    else if (cc < 70) {
        return EVENT_OR_METRIC_API;
    }
    else {
        return EITHER_API;
    }
}

/** @class find_same_chipname
  * @brief Check to see if chipnames are identical.
  * 
  * @param deviceId
  *   A device id. E.g. 0, 1, 2, etc.
*/
static int find_same_chipname(int deviceId)
{
    int i;
    for (i = 0; i < deviceId; i++) {
        if (!strcmp(cuptiu_table_p->avail_gpu_info[deviceId].chipName, cuptiu_table_p->avail_gpu_info[i].chipName)) {
            return i;
        }
    }
    return -1;
}

/** @class enumerate_events_for_event_api
  * @brief For the Events API enumerate through the available events
  *        for a device collecting the description.
  * @param *table
  *   Structure containing member variables such as name, evt_code, evt_pos,
  *   and htable.
  * @param ctx
  *   The created Cuda context for a device.
  * @param device
  *   A created Cuda device handle.
  * @param deviceIdx
  *   Device index.
*/
static int enumerate_events_for_event_api(cuptiu_event_and_metric_table_t *table, CUdevice device, int deviceIdx)
{
    SUBDBG("ENTERING: Enumerating events for the CUPTI Event API.\n");
    // Get the total number of Event Domains for a device
    uint32_t numDomains;
    cuptiCheckErrors( cuptiDeviceGetNumEventDomainsPtr(device, &numDomains),  return PAPI_EMISC );

    // Get the event domains for a device
    size_t size = numDomains * sizeof(CUpti_EventDomainID);
    CUpti_EventDomainID *domainArray = (CUpti_EventDomainID *) calloc(numDomains, sizeof(CUpti_EventDomainID));
    if (domainArray == NULL) {
        SUBDBG("Failed to allocate memory for domainArray.\n");
        return PAPI_ENOMEM;
    }
    cuptiCheckErrors( cuptiDeviceEnumEventDomainsPtr(device, &size, domainArray), return PAPI_EMISC );

    // Go over the total number of domains found for the device
    int domainIdx;
    for (domainIdx = 0; domainIdx < numDomains; domainIdx++) {
        // For each domain, get the total number of events 
        uint32_t numEvents;
        cuptiCheckErrors( cuptiEventDomainGetNumEventsPtr(domainArray[domainIdx], &numEvents), return PAPI_EMISC );

        // Allocate memory
        CUpti_EventID *eventArray = (CUpti_EventID *) calloc(numEvents, sizeof(CUpti_EventID));
        if (eventArray == NULL) {
            SUBDBG("Failed to allocate memory for eventArray.\n");
            return PAPI_ENOMEM;
        }
        size = numEvents * sizeof(CUpti_EventID);

        // For the domain, get the actual events
        cuptiCheckErrors( cuptiEventDomainEnumEventsPtr(domainArray[domainIdx], &size, eventArray), return PAPI_EMISC );

        // Go over the total number of events found in the domain for the device
        CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
        if (eventGroupPasses == NULL) {
            SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
            return PAPI_ENOMEM;
        }
        int eventIdx;
        for (eventIdx = 0; eventIdx < numEvents; eventIdx++) {
            // Name attribute
            size = PAPI_2MAX_STR_LEN * sizeof(char);
            char eventName[PAPI_2MAX_STR_LEN];
            cuptiCheckErrors( cuptiEventGetAttributePtr(eventArray[eventIdx], CUPTI_EVENT_ATTR_NAME, &size, eventName), return PAPI_EMISC );

            char reconstructedEventName[PAPI_2MAX_STR_LEN];
            int strLen = snprintf(reconstructedEventName, PAPI_2MAX_STR_LEN, "%s", eventName);
            if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
                SUBDBG("Failed to fully write event name with event: appended to front.\n");
                return PAPI_EBUF;
            }

            // Long description attribute
            size = PAPI_HUGE_STR_LEN * sizeof(char);
            char eventDesc[PAPI_HUGE_STR_LEN];
            cuptiCheckErrors( cuptiEventGetAttributePtr(eventArray[eventIdx], CUPTI_EVENT_ATTR_LONG_DESCRIPTION, &size, eventDesc), return PAPI_EMISC );

            int papi_errno = store_event_or_metric_ntv_events(table, reconstructedEventName, eventDesc, deviceIdx, EVENT);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }
        }
        free(eventGroupPasses);
        free(eventArray);
    }
    free(domainArray);

    SUBDBG("EXITING: Enumeration completed for the CUPTI Event API.\n");
    return PAPI_OK;
}

/** @class enumerate_metrics_for_metric_api
  * @brief For the Metric API enumerate through the available metrics
  *        for a device collecting the description and number of passes.
  * @param *table
  *   Structure containing member variables such as name, evt_code, evt_pos,
  *   and htable.
  * @param ctx
  *   The created Cuda context for a device.
  * @param device
  *   A created Cuda device handle.
  * @param deviceIdx
  *   Device index.
*/
static int enumerate_metrics_for_metric_api(cuptiu_event_and_metric_table_t *table, CUdevice device, int deviceIdx)
{
    SUBDBG("ENTERING: Enumerating metrics for the CUPTI Metric API.\n");
    // Get the total number of metrics for a device
    int numMetrics;
    cuptiCheckErrors( cuptiDeviceGetNumMetricsPtr(device, &numMetrics), return PAPI_EMISC );

    // Get the metrics for the device
    size_t size = numMetrics * sizeof(CUpti_MetricID);
    CUpti_MetricID *metricIdList = (CUpti_MetricID *) calloc(numMetrics, sizeof(CUpti_MetricID));
    if (metricIdList == NULL) {
        SUBDBG("Failed to allocate memory for metricIdList.\n");
        return PAPI_ENOMEM;
    }
    cuptiCheckErrors( cuptiDeviceEnumMetricsPtr(device, &size, metricIdList), return PAPI_EMISC );

    // For each metric get the name and description attribute
    CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
    if (eventGroupPasses == NULL) {
        SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
        return PAPI_ENOMEM;
    }
    int metricIdx;
    for (metricIdx = 0; metricIdx < numMetrics; metricIdx++) {
        // Name attribute
        size = PAPI_2MAX_STR_LEN * sizeof(char);
        char metricName[PAPI_2MAX_STR_LEN];
        cuptiCheckErrors( cuptiMetricGetAttributePtr(metricIdList[metricIdx], CUPTI_METRIC_ATTR_NAME, &size, metricName), return PAPI_EMISC );

        char reconstructedMetricName[PAPI_2MAX_STR_LEN];
        int strLen = snprintf(reconstructedMetricName, PAPI_2MAX_STR_LEN, "%s", metricName);
        if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
            SUBDBG("Failed to fully write event name with event: appended to front.\n");
            return PAPI_EBUF;
        }

        // Long description attribute
        size = PAPI_HUGE_STR_LEN * sizeof(char);
        char metricDesc[PAPI_HUGE_STR_LEN];
        cuptiCheckErrors( cuptiMetricGetAttributePtr(metricIdList[metricIdx], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size, metricDesc), return PAPI_EMISC );

        char fullMetricDesc[PAPI_HUGE_STR_LEN];
        strLen = snprintf(fullMetricDesc, PAPI_HUGE_STR_LEN, "%s", metricDesc);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write description for metric.\n");
            return PAPI_EBUF;
        }

        int papi_errno = store_event_or_metric_ntv_events(table, reconstructedMetricName, metricDesc, deviceIdx, METRIC);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    free(metricIdList);
    free(eventGroupPasses);

    SUBDBG("EXITING: Enumeration completed for the CUPTI Metric API.\n");
    return PAPI_OK;
}

/** @class store_event_or_metric_ntv_events
  * @brief For the devices on a machine, store an event or metric with its corresponding
  *        metadata.
  * @param evt_table
  *   Structure containing member variables such as name, evt_code, evt_pos, and htable.
  * @param *evt_name
  *   Cuda event or metric name that we want to store.
  * @param *evt_desc
  *   A Cuda event or metric's description that we want to store.
  * @param deviceIdx
  *   Device index.
  * @param api
  *   The CUPTI api that was used, either event or metric.
*/
static int store_event_or_metric_ntv_events(cuptiu_event_and_metric_table_t *evt_table, const char *evt_name, const char *evt_desc, int deviceIdx, int api)
{
    int *count = &evt_table->count;
    cuptiu_event_and_metric_t *events = evt_table->events;

    if (evt_name == NULL) {
        return PAPI_EINVAL;
    }

    if (evt_table->count >= evt_table->capacity) {
        return PAPI_EBUG;
    }

    cuptiu_event_and_metric_t *event;
    if (htable_find(evt_table->htable, evt_name, (void **) &event) != HTABLE_SUCCESS) {
        event = &events[*count];
        (*count)++;

        int strLen = snprintf(event->name, PAPI_2MAX_STR_LEN, "%s", evt_name);
        if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
            SUBDBG("Failed to fully write evt_name to event->name.\n");
            return PAPI_EBUF;
        }

        strLen = snprintf(event->desc, PAPI_HUGE_STR_LEN, "%s", evt_desc);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write evt_desc to event->desc.\n");
            return PAPI_EBUF;
        }

        event->api = api;

        if (htable_insert(evt_table->htable, evt_name, event) != HTABLE_SUCCESS) {
            return PAPI_ESYS;
        } 
    }
    cuptiu_dev_set(&event->device_map, deviceIdx);

    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/


/***************************************************************************//**
 *  @section  Functions specific to a PAPI profiling workflow (e.g. start - stop)
 *
 *  @{
 */

/** @class cuptie_ctx_create
  * @brief Allocate memory, convert user added events to event and metric IDs, and
  *        verify multiple passes is not required.
  *
  * @param thr_info
  *   Information for the Cuda contexts on the calling cpu threads.
  * @param *pstate
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
  * @param *eventIds
  *   User added Cuda native event ids.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
*/
int cuptie_ctx_create(cuptic_info_t thr_info, cuptie_control_t *pstate, uint32_t *eventIds, int num_events)
{
    SUBDBG("ENTERING: Creating a profiling context for the request cuda native events.\n");

    cuptie_control_t state = (cuptie_control_t) calloc(1, sizeof(struct cuptie_control_s));
    if (state == NULL) {
        SUBDBG("Failed to allocate memory for state.\n");
        return PAPI_ENOMEM; 
    }

    state->gpu_ctl = (cuptie_gpu_state_t *) calloc(numDevicesOnMachine, sizeof(cuptie_gpu_state_t));
    if (state->gpu_ctl == NULL) {
        SUBDBG("Failed to allocate memory for state->gpu_ctl.\n");
        return PAPI_ENOMEM;
    }

    long long *counters = (long long *) malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        SUBDBG("Failed to allocate memory for counters.\n");
        return PAPI_ENOMEM;
    }

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        state->gpu_ctl[deviceIdx].deviceIdx = deviceIdx;
    }

    event_info_t native_event_info;
    int papi_errno = event_and_metric_id_to_info(eventIds[num_events - 1], &native_event_info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Check for a context that has been created by the user on the calling cpu thread
    CUcontext currentUserContext;
    cudaCheckErrors( cuCtxGetCurrentPtr(&currentUserContext), return PAPI_EMISC);
    // No user created context on the calling cpu thread; therefore, we create one based off the
    // appended device qualifier i.e. event:device=#
    if (currentUserContext == NULL) {
        // Only create a context if one has not already been stored for the appended device qualifier
        if (thr_info[native_event_info.device].ctx == NULL) {
            unsigned int contextFlags = 0;
            CUcontext internalContext; 
            cudaArtCheckErrors( cudaSetDevicePtr(native_event_info.device), return PAPI_EMISC );
            cudaArtCheckErrors( cudaFreePtr(NULL), return PAPI_EMISC );  
            cudaCheckErrors( cuCtxGetCurrentPtr(&internalContext), return PAPI_EMISC); 
            thr_info[native_event_info.device].ctx = internalContext;
            // Pop the context off so verify_user_added_event_or_metric functions properly
            cudaCheckErrors( cuCtxPopCurrentPtr(&internalContext), return PAPI_EMISC );
        }
    }
    // User created context found on the calling cpu thread
    else {
         CUdevice deviceIdxForContext;
         cudaCheckErrors( cuCtxGetDevicePtr(&deviceIdxForContext), return PAPI_EMISC );

        // No previous context stored for the appended device qualifier
        if (thr_info[deviceIdxForContext].ctx == NULL) {
           if (deviceIdxForContext != native_event_info.device) {
               SUBDBG("The Cuda context associated with device %d does not match appended device qualifier %d.\n", deviceIdxForContext, native_event_info.device);
               return PAPI_ECOMBO;
           }
           thr_info[deviceIdxForContext].ctx = currentUserContext;
        }
        // Previous context store for the appended device qualifier 
        else {
           if (thr_info[deviceIdxForContext].ctx != currentUserContext) {
               SUBDBG("Multiple contexts found for device %d. Keeping the first context context found.\n", native_event_info.device);
           }    
        }
        // Pop the context off so verify_user_added_event_or_metric functions properly
        cudaCheckErrors( cuCtxPopCurrentPtr(&currentUserContext), return PAPI_EMISC ); 
    }

    papi_errno = verify_user_added_event_or_metric(eventIds, num_events, state, thr_info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;    
    }

    // For the case we found a cuda context from the user on the calling cpu thread
    // and popped it off, we must push it back. As we do not want their context
    // management function calls to fail.
    if (currentUserContext != NULL) {
        cudaCheckErrors( cuCtxPushCurrentPtr(currentUserContext), return PAPI_EMISC );
    }

    state->info = thr_info;
    state->counters = counters;
    *pstate = state;

    SUBDBG("EXITING: Creation of a profiling context completed.\n");
    return PAPI_OK;
}

/** @class cuptie_ctx_start
  * @brief Take the user added event or metrics and create event group sets.
  *        Then enable the event group sets for profiling.
  *
  * @param state
  *  Structure containing read_count, state of the eventset, etc. 
*/
int cuptie_ctx_start(cuptie_control_t state)
{
    SUBDBG("ENTERING: Setting up profiling for the Event and Metric APIs.\n");

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[deviceIdx]);
        if (gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents == 0) {
            continue;
        }

        cudaCheckErrors( cuCtxSetCurrentPtr(state->info[deviceIdx].ctx), return PAPI_EMISC );

        // Calculate the total number of user added events
        int numTotalEventIdEntries = 0;
        int i;
        for (i = 0; i < gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents; i++) {
            numTotalEventIdEntries += gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[i];
        }

        CUpti_EventID *eventIdsArray = (CUpti_EventID *) calloc(numTotalEventIdEntries, sizeof(CUpti_EventID));
        if (eventIdsArray == NULL) {
            SUBDBG("Failed to allocate memory for eventIdsArray in cuptie_ctx_start.\n");
            return PAPI_ENOMEM;
        }

        // Convert 2D array of Event IDs into a 1D array of Event IDs
        int index = 0;
        int recordIdx;
        for (recordIdx = 0; recordIdx < gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents; recordIdx++) {
            int subIdsIdx;
            for (subIdsIdx = 0; subIdsIdx < gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx]; subIdsIdx++) {
                eventIdsArray[index++] = gpu_ctl->added_events->idsThatMakeupAUserAddedEventArray[recordIdx][subIdsIdx];
            }
        }

        CUpti_EventGroupSets *eventGroupSets;
        size_t numEventsAddedSize = sizeof(CUpti_EventID) * numTotalEventIdEntries;
        cuptiCheckErrors( cuptiEventGroupSetsCreatePtr(state->info[deviceIdx].ctx, numEventsAddedSize, eventIdsArray, &eventGroupSets), return PAPI_EMISC );

        // There should only ever be a single set, as sets > 1 require multiple passes
        CUpti_EventGroupSet *sets = &(eventGroupSets->sets[0]);
        for (i = 0; i < sets->numEventGroups; i++) {
            uint32_t trash = 1;
            CUpti_EventGroupAttribute eventGroupAttr = CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES;
            cuptiCheckErrors( cuptiEventGroupSetAttributePtr(sets->eventGroups[i], eventGroupAttr, sizeof(trash), &trash), return PAPI_EMISC );

            uint32_t numEvents;
            size_t numGroupEventsSize = sizeof(numEvents);
            cuptiCheckErrors( cuptiEventGroupGetAttributePtr(sets->eventGroups[i], CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numGroupEventsSize, &numEvents), return PAPI_EMISC);
        }

        cuptiCheckErrors( cuptiEventGroupSetEnablePtr(sets), return PAPI_EMISC );

        gpu_ctl->added_events->eventGroupSets = eventGroupSets;

        free(eventIdsArray);

        cuptiCheckErrors( cuCtxPopCurrentPtr(&state->info[deviceIdx].ctx), return PAPI_EMISC );
    }

    SUBDBG("EXITING: Profiling setup completed.\n");
    return PAPI_OK;
}

/** @class cuptie_ctx_read
  * @brief For each event group set calculate the event or metric values.
  *
  * @param state
  *  Structure containing read_count, state of the eventset, etc. 
  * @param **counterValues
  *  Variable to store the counter values for the calculated event or metric
  *  values.
*/
int cuptie_ctx_read(cuptie_control_t state, long long **counterValues)
{
    SUBDBG("ENTERING: Reading values for the Event and Metric APIs.\n");

    int numCountersRead = 0;
    long long *readCounterValues = state->counters;

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[deviceIdx]);
        if (gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents == 0) {
            continue;
        }

        // Get the read time stamp
        uint64_t readTimeStampNs = 0;
        cuptiCheckErrors( cuptiGetTimestampPtr(&readTimeStampNs), return PAPI_EMISC );
        uint64_t duration = readTimeStampNs - gpu_ctl->added_events->startTimeStampNs;
        gpu_ctl->added_events->startTimeStampNs = readTimeStampNs;

        cudaCheckErrors( cuCtxSetCurrentPtr(state->info[deviceIdx].ctx), return PAPI_EMISC );

        cudaCheckErrors( cuCtxSynchronizePtr(), return PAPI_EMISC );

        CUdevice device;
        cudaCheckErrors( cuDeviceGetPtr(&device, deviceIdx), return PAPI_EMISC );

        // Irrespective of if we are working with Events or Metrics, this section below needs to be done
        int groupSetIdx;
        int numEventGroups = gpu_ctl->added_events->eventGroupSets->sets[0].numEventGroups;
        for (groupSetIdx = 0; groupSetIdx < numEventGroups; groupSetIdx++) { // Go over all of the available event groups for the device
            CUpti_EventGroup eventGroup = gpu_ctl->added_events->eventGroupSets->sets[0].eventGroups[groupSetIdx];

            // Get the total number of events in the event group
            int numGroupEvents = 0;
            size_t numGroupEventsSize = sizeof(numGroupEvents);
            CUpti_EventGroupAttribute groupAttrib = CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS;
            cuptiCheckErrors( cuptiEventGroupGetAttributePtr(eventGroup, groupAttrib, &numGroupEventsSize, &numGroupEvents), return PAPI_EMISC );

            // Get the event group domain, this is needed for getting the total number of instances
            CUpti_EventDomainID groupDomain;
            size_t groupDomainSize = sizeof(groupDomain);
            groupAttrib = CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID;
            cuptiCheckErrors( cuptiEventGroupGetAttributePtr(eventGroup, groupAttrib, &groupDomainSize, &groupDomain), return PAPI_EMISC );
        
            // Get the total number of instances for the group domain
            uint32_t numTotalInstances;
            size_t numTotalInstancesSize = sizeof(numTotalInstances);
            CUpti_EventDomainAttribute domainAttrib = CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT;
            cuptiCheckErrors( cuptiDeviceGetEventDomainAttributePtr(device, groupDomain, domainAttrib, &numTotalInstancesSize, &numTotalInstances), return PAPI_EMISC );

            uint32_t numInstances;
            size_t numInstancesSize = sizeof(numInstances);
            domainAttrib = CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT;
            cuptiCheckErrors( cuptiDeviceGetEventDomainAttributePtr(device, groupDomain, domainAttrib, &numInstancesSize, &numInstances), return PAPI_EMISC );
  
            size_t sizeOfEventValueBufferInBytes = sizeof(uint64_t) * numGroupEvents * numInstances;
            uint64_t *eventValueBuffer = (uint64_t *) malloc(sizeOfEventValueBufferInBytes);
            if (eventValueBuffer == NULL) {
                SUBDBG("Failed to allocate memory for eventValueBuffer.\n");
                return PAPI_ENOMEM;
            }

            size_t sizeOfEventIdArrayInBytes = numGroupEvents * sizeof(CUpti_EventID);
            CUpti_EventID *eventIdArray = (CUpti_EventID *) malloc(sizeOfEventIdArrayInBytes);
            if (eventIdArray == NULL) {
                SUBDBG("Failed to allocate memory for eventIdArray.\n");
                return PAPI_ENOMEM;
            }

            size_t numEventIdsRead;
            CUpti_ReadEventFlags readEventFlags = CUPTI_EVENT_READ_FLAG_NONE;
            // Will read us back the event values, with the eventIdArray holding the ids of the events in the same order
            // as the values returned
            cuptiCheckErrors(cuptiEventGroupReadAllEventsPtr(eventGroup,
                                                             readEventFlags,
                                                             &sizeOfEventValueBufferInBytes,
                                                             eventValueBuffer,
                                                             &sizeOfEventIdArrayInBytes,
                                                             eventIdArray,
                                                             &numEventIdsRead), return PAPI_EMISC);

            // For the total number of event ids that have been read, accumulate the values
            uint64_t *accumulateEventVals = (uint64_t *) calloc(numEventIdsRead, sizeof(uint64_t));
            if (accumulateEventVals == NULL) {
                SUBDBG("Failed to allocate memory for aggregateCounterVals.\n");
                return PAPI_ENOMEM;
            }

            int eventIdx; 
            for (eventIdx = 0; eventIdx < numEventIdsRead; eventIdx++) {
                int instanceIdx;
                for (instanceIdx = 0; instanceIdx < numInstances; instanceIdx++) { 
                    accumulateEventVals[eventIdx] += eventValueBuffer[eventIdx + (numGroupEvents * instanceIdx)];
                }
            }

            int recordIdx;
            for (recordIdx = 0; recordIdx < gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents; recordIdx++) {
                if (gpu_ctl->added_events->userAddedIdsCorrespondingApiArray[recordIdx] == EVENT) {
                    int subIdsIdx;
                    for (subIdsIdx = 0; subIdsIdx < gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx]; subIdsIdx++) {
                        for (eventIdx = 0; eventIdx < numEventIdsRead; eventIdx++) {
                            if (gpu_ctl->added_events->idsThatMakeupAUserAddedEventArray[recordIdx][subIdsIdx] == eventIdArray[eventIdx]) {
                                gpu_ctl->added_events->cumulativeValuesArray[recordIdx][subIdsIdx] += accumulateEventVals[eventIdx];
                                readCounterValues[recordIdx] = gpu_ctl->added_events->cumulativeValuesArray[recordIdx][subIdsIdx];
                                numCountersRead++;
                            }
                        }
                    }
                }
                else if (gpu_ctl->added_events->userAddedIdsCorrespondingApiArray[recordIdx] == METRIC) {
                    // NOTE: A metric is made up of one or more events. At the time of writing this (09/24/2025),
                    // I have only ever seen a metric require a maximum of two events. If the metric does require
                    // two events, then those two events are placed into a single event group by themselves
                    // (this has been shown from my own personal testing). However, if a user were to add two metrics
                    // with both metrics only requiring a single event, then these two events could be placed into the same eventGroup.
                    size_t sizeOfNormalizedEventValuesArrayInBytes = gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx] * sizeof(uint64_t);
                    uint64_t *normalizedEventValuesArray = (uint64_t *) calloc(gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx], sizeof(uint64_t));

                    int allMatchingEventsFound = 0;

                    int *eventsThatMakeupAMetricArray = (int *) calloc(gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx], sizeof(int));
                    int subIdsIdx;
                    for (subIdsIdx = 0; subIdsIdx < gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx]; subIdsIdx++) {
                        for (eventIdx = 0; eventIdx < numEventIdsRead; eventIdx++) {
                            // NOTE: Two metric's can have the same event id which can lead to the actual values being identical, for example
                            // issue_slots and inst_issued. Due to this we just look for matching event id's and then store the value.
                            if (gpu_ctl->added_events->idsThatMakeupAUserAddedEventArray[recordIdx][subIdsIdx] == eventIdArray[eventIdx]) {
                                // Normalize the values
                                gpu_ctl->added_events->cumulativeValuesArray[recordIdx][subIdsIdx] += (accumulateEventVals[eventIdx] * numTotalInstances) / numInstances;
                                eventsThatMakeupAMetricArray[subIdsIdx] = eventIdArray[eventIdx];
                                normalizedEventValuesArray[subIdsIdx] = gpu_ctl->added_events->cumulativeValuesArray[recordIdx][subIdsIdx];
                                allMatchingEventsFound++;

                                // Once we have collected all of the events that makeup the metric we can then get the metric value
                                if (allMatchingEventsFound == gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx]) {
                                    goto get_metric_value;
                                }
                            }
                        }
                    }
                    // For the current record index and it's sub indexes we did not find a match; therefore,
                    // we will free the memory and continue onto the next interation
                    goto free_memory_and_continue;

                    // Calculate the metric value as we have all the event's that makeup the current metric
                    get_metric_value: ;
                        CUpti_MetricValue metricValue;
                        cuptiCheckErrors( cuptiMetricGetValuePtr(device,
                                                                 gpu_ctl->added_events->metricIDs[recordIdx],
                                                                 gpu_ctl->added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[recordIdx] * sizeof(CUpti_EventID),
                                                                 eventsThatMakeupAMetricArray,
                                                                 sizeOfNormalizedEventValuesArrayInBytes,
                                                                 normalizedEventValuesArray,
                                                                 duration,
                                                                 &metricValue), return PAPI_EMISC);

                        long long metricValueLongLong;
                        int papi_errno = convert_metric_value_to_long_long(gpu_ctl->added_events->metricIDs[recordIdx], metricValue, &metricValueLongLong);
                        if (papi_errno != PAPI_OK) {
                            return papi_errno;
                        }
                        readCounterValues[recordIdx] = metricValueLongLong;
                        numCountersRead++;

                    free_memory_and_continue:
                        free(normalizedEventValuesArray);
                        free(eventsThatMakeupAMetricArray);
                }
            }
            free(eventValueBuffer);
            free(eventIdArray);
            free(accumulateEventVals);
        }
        cudaCheckErrors( cuCtxPopCurrentPtr(&state->info[deviceIdx].ctx), return PAPI_EMISC );
    }
    state->read_count = numCountersRead;
    *counterValues = readCounterValues;

    SUBDBG("EXITING: Reading values completed.\n");
    return PAPI_OK;
}

/** @class cuptie_ctx_stop
  * @brief Disable an event group set.
  *
  * @param state
  *  Structure containing read_count, state of the eventset, etc. 
*/
int cuptie_ctx_stop(cuptie_control_t state)
{
    SUBDBG("ENTERING: Disabling and destroying the event group sets created. Collection of events will be stopped.\n");

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[deviceIdx]);
        if (gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents == 0) {
            continue;
        }

        cudaCheckErrors( cuCtxSetCurrentPtr(state->info[deviceIdx].ctx), return PAPI_EMISC );

        CUpti_EventGroupSets *eventGroupSets = gpu_ctl->added_events->eventGroupSets;
        // There should only ever be a single set, as sets > 1 require multiple passes
        CUpti_EventGroupSet *eventGroupSet = &(eventGroupSets->sets[0]);
        cuptiCheckErrors( cuptiEventGroupSetDisablePtr(eventGroupSet), return PAPI_EMISC );
        cuptiCheckErrors( cuptiEventGroupSetsDestroyPtr(eventGroupSets), return PAPI_EMISC );

        cudaCheckErrors( cuCtxPopCurrentPtr(&state->info[deviceIdx].ctx), return PAPI_EMISC );
    }

    SUBDBG("EXITING: Disabling event group sets completed.\n");
    return PAPI_OK;
}

/** @class cuptie_ctx_reset
  * @brief Reset the stored counter values and read count to zero. As well
  *        as for an event group reset all events.
  *
  * @param state
  *  Structure containing read_count, state of the eventset, etc. 
*/
int cuptie_ctx_reset(cuptie_control_t state)
{
    SUBDBG("ENTERING: Resetting counter values.\n");

    int counterIdx;
    for (counterIdx = 0; counterIdx < state->read_count; counterIdx++) {
        state->counters[counterIdx] = 0;
    }
    state->read_count = 0;

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[deviceIdx]);
        if (gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents == 0) {
            continue;
        }

        cudaCheckErrors( cuCtxSetCurrentPtr(state->info[deviceIdx].ctx), return PAPI_EMISC );

        // There should only ever be a single set, as sets > 1 require multiple passes
        int numEventGroups = gpu_ctl->added_events->eventGroupSets->sets[0].numEventGroups;
        CUpti_EventGroupSet set = gpu_ctl->added_events->eventGroupSets->sets[0];

        int eventGroupIdx;
        for (eventGroupIdx = 0; eventGroupIdx < numEventGroups; eventGroupIdx++) {
            CUpti_EventGroup eventGroup = set.eventGroups[eventGroupIdx];
            cuptiCheckErrors( cuptiEventGroupResetAllEventsPtr(eventGroup), return PAPI_EMISC );
        }

        cudaCheckErrors( cuCtxPopCurrentPtr(&state->info[deviceIdx].ctx), return PAPI_EMISC );
    }

    SUBDBG("EXITING: Resetting counter values completed.\n");
    return PAPI_OK;
}

/** @class cuptie_ctx_destroy
  * @brief Free memory and destroy the event and metric tables.
  *
  * @param state
  *  Structure containing read_count, state of the eventset, etc. 
*/
int cuptie_ctx_destroy(cuptie_control_t *pstate)
{
    SUBDBG("ENTERING: Destroying event table.\n");

    int deviceIdx;
    for (deviceIdx = 0; deviceIdx < numDevicesOnMachine; deviceIdx++) {
        cuptie_gpu_state_t *gpu_ctl =  &((*pstate)->gpu_ctl[deviceIdx]);
        if (gpu_ctl->added_events->totalNumberOfUserAddedNativeEvents == 0) {
            continue;
        }

        destroy_event_and_metric_table(&(gpu_ctl->added_events));
    }

    free((*pstate)->counters);
    free((*pstate)->gpu_ctl);
    free((*pstate));
    *pstate = NULL;

    SUBDBG("EXITING: Destroying event table completed.\n");
    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section  Helper functions specific to a PAPI profiling workflow 
 *            (e.g. start - stop).
 *
 *  @{
 */

/** @class check_if_event_or_metric_requires_multiple_passes
  * @brief Take the event or metric that is added by the user and check
  *        to make sure it does not require multiple passes.
  * @param *addedEventName
  *   The Cuda native event added by the user (Event or Metric).
  * @param cuptiApi
  *   The API the Cuda native event is associated with (Event or Metric).
  * @param deviceIdx
  *   The device index for the added Cuda native event.
  * @param *addedNativeEventID
  *   If the event does not require multiple passes store it such that it can be added
  *   to an EventGroup later.
*/
static int check_if_event_or_metric_required_multiple_passes(const char *addedEventName, int cuptiApi, int deviceIdx, uint32_t *addedNativeEventID)
{
    SUBDBG("ENTERING: Checking if the user added event does not require multiple passes.\n");

    CUcontext pctx;
    cudaCheckErrors( cuCtxGetCurrentPtr(&pctx), return PAPI_EMISC );

    CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
    if (eventGroupPasses == NULL) {
        SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
        return PAPI_ENOMEM;
    }

    CUpti_EventID eventId;
    CUpti_MetricID metricId;
    int papi_errno = PAPI_OK;
    switch(cuptiApi)
    {
        case EVENT:
            cuptiCheckErrors( cuptiEventGetIdFromNamePtr(deviceIdx, addedEventName, &eventId), return PAPI_EMISC );
            cuptiCheckErrors( cuptiEventGroupSetsCreatePtr(pctx, sizeof(CUpti_EventID), &eventId, &eventGroupPasses), return PAPI_EMISC );
            break;
        case METRIC:
            cuptiCheckErrors( cuptiMetricGetIdFromNamePtr(deviceIdx, addedEventName, &metricId), return PAPI_EMISC );
            cuptiCheckErrors( cuptiMetricCreateEventGroupSetsPtr(pctx, sizeof(CUpti_MetricID), &metricId, &eventGroupPasses), return PAPI_EMISC );
            break;
        default:
            SUBDBG("API option is not supported. Internal error.\n");
            papi_errno = PAPI_EBUG;
            break;
    }

    if (eventGroupPasses->numSets > 1) {
        SUBDBG("Multiple pass event/metric detected. PAPI does not support events/metrics that require multiple passes.\n");
        papi_errno = PAPI_EMULPASS;
    }

    // Store the found Event or Metric ID 
    *addedNativeEventID = (cuptiApi == EVENT) ? eventId : metricId;

    // Cleanup
    free(eventGroupPasses);

    SUBDBG("EXITING: Check for multiple passes completed.\n");
    return papi_errno;
}

/** @class verify_user_added_event_or_metric
  * @brief For user added events or metrics, verify they exist and do not require
  *        multiple passes. If both are true, store metadata.
  * @param *events_id
  *   Cuda native event id's.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
*/
static int verify_user_added_event_or_metric(uint32_t *events_id, int num_events, cuptie_control_t state, cuptic_info_t thr_info)
{
    SUBDBG("ENTERING: Verifying user added events exist and do not require multiple passes.\n");

    int i, papi_errno = PAPI_OK;
    for (i = 0; i < numDevicesOnMachine; i++) {
        papi_errno = create_event_and_metric_table(num_events, &(state->gpu_ctl[i].added_events));
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    int totalNumberOfUserAddedEvents = 0;
    for (i = 0; i < num_events; i++) {
        event_info_t native_event_info;
        papi_errno = event_and_metric_id_to_info(events_id[i], &native_event_info);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        // Set the appropriate context for the native events appended qualifier 
        cudaCheckErrors( cuCtxSetCurrentPtr(thr_info[native_event_info.device].ctx), return PAPI_EMISC );

        // Verify the user added event exists
        void *p;
        if (htable_find(cuptiu_table_p->htable, cuptiu_table_p->events[native_event_info.nameid].name, (void **) &p) != HTABLE_SUCCESS) {
            SUBDBG("The added event %s does not exist.\n", cuptiu_table_p->events[native_event_info.nameid].name);
            return PAPI_ENOEVNT;
        }

        uint32_t addedNativeEventID;
        // Verify that the user added event does not require multiple passes
        int papi_errno = check_if_event_or_metric_required_multiple_passes(cuptiu_table_p->events[native_event_info.nameid].name,
                                                                          cuptiu_table_p->events[native_event_info.nameid].api,
                                                                          native_event_info.device, &addedNativeEventID);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        // If the user added event belongs to the Event API
        if (cuptiu_table_p->events[native_event_info.nameid].api == EVENT) {
            state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents] = (int *) calloc(1, sizeof(int));
            if (state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents] == NULL) {
                SUBDBG("Failed to allocate memory for index position %d.\n", totalNumberOfUserAddedEvents);
                return PAPI_ENOMEM;
            }

            state->gpu_ctl[native_event_info.device].added_events->cumulativeValuesArray[totalNumberOfUserAddedEvents] = (int *) calloc(1, sizeof(int));
            if (state->gpu_ctl[native_event_info.device].added_events->cumulativeValuesArray[totalNumberOfUserAddedEvents] == NULL) {
                SUBDBG("Failed to allocate memory for index position %d.\n", totalNumberOfUserAddedEvents);
                return PAPI_ENOMEM;
            }

            // Event IDs only ever have a single ID associated with them unlike with Metric IDs that may have 1 or more; therefore, 0 is harded coded
            // along with 1
            state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents][0] = addedNativeEventID;
            state->gpu_ctl[native_event_info.device].added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[totalNumberOfUserAddedEvents] = 1;
            state->gpu_ctl[native_event_info.device].added_events->userAddedIdsCorrespondingApiArray[totalNumberOfUserAddedEvents] = EVENT;

            // -1 is used as placeholder here as if it not place here then we would need to keep track of a second index instead of just using
            // totalNumberOfUserAddedEvents
            state->gpu_ctl[native_event_info.device].added_events->metricIDs[totalNumberOfUserAddedEvents] = -1;
        }
        // If the user added event belongs to the Metric API
        else {
            // NOTE: The Metric API can be thought of as a higher level api as metrics consist of events from the Event API.
            // Due to this, you can convert a single metric to the event or events that it is made up of. To allow users to add
            // both event and metrics at the same time we are going to convert a metric to the events that it is made up of.
            // Because from testing, creating and enabling a cuptiMetricCreateEventGroupSets and cuptiEventGroupSetsCreate for the same context
            // will result in a failure.
            uint32_t numEvents;
            cuptiCheckErrors( cuptiMetricGetNumEventsPtr(addedNativeEventID, &numEvents), return PAPI_EMISC);

            // Allocate memory for the total number of events that the metric is made up of
            size_t sizeOfEventIdsArrayInBytes = numEvents * sizeof(CUpti_EventID);
            CUpti_EventID *eventIdsArray = (CUpti_EventID *) malloc(sizeOfEventIdsArrayInBytes);
            if (eventIdsArray == NULL) {
                SUBDBG("Failed to allocate memory for eventIdsArray.\n");
                return PAPI_ENOMEM;
            }

            state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents] = (int *) calloc(numEvents, sizeof(int));
            if (state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents] == NULL) {
                SUBDBG("Failed to allocate memory for index position %d.\n", totalNumberOfUserAddedEvents);
                return PAPI_ENOMEM;
            }

            state->gpu_ctl[native_event_info.device].added_events->cumulativeValuesArray[totalNumberOfUserAddedEvents] = (int *) calloc(numEvents, sizeof(int));
            if (state->gpu_ctl[native_event_info.device].added_events->cumulativeValuesArray[totalNumberOfUserAddedEvents] == NULL) {
                SUBDBG("Failed to allocate memory for index position %d.\n", totalNumberOfUserAddedEvents);
                return PAPI_ENOMEM;
            }

            // For the metric id, get the event or events that it is made up of
            cuptiCheckErrors( cuptiMetricEnumEventsPtr(addedNativeEventID, &sizeOfEventIdsArrayInBytes, eventIdsArray), return PAPI_EMISC);

            // Store the Event Ids that makeup the metric
            int eventIdx;
            for (eventIdx = 0; eventIdx < numEvents; eventIdx++) {
                state->gpu_ctl[native_event_info.device].added_events->idsThatMakeupAUserAddedEventArray[totalNumberOfUserAddedEvents][eventIdx] = eventIdsArray[eventIdx];
            }

            state->gpu_ctl[native_event_info.device].added_events->totalNumberOfIdsThatMakeupTheUserAddedEventArray[totalNumberOfUserAddedEvents] = numEvents;
            state->gpu_ctl[native_event_info.device].added_events->userAddedIdsCorrespondingApiArray[totalNumberOfUserAddedEvents] = METRIC;

            // Store Metric ID as it will be needed within cuptie_ctx_read
            state->gpu_ctl[native_event_info.device].added_events->metricIDs[totalNumberOfUserAddedEvents] = addedNativeEventID;

            // Free allocated memory
            free(eventIdsArray);
        }
        totalNumberOfUserAddedEvents++;
        state->gpu_ctl[native_event_info.device].added_events->totalNumberOfUserAddedNativeEvents = totalNumberOfUserAddedEvents;

        // Pop off the set context
        cudaCheckErrors( cuCtxPopCurrentPtr(&thr_info[native_event_info.device].ctx), return PAPI_EMISC );
    }

    SUBDBG("EXITING: Checking user added a valid event completed.\n");
    return PAPI_OK;
}

/** @class convert_metric_value_to_long_long
  * @brief For a metric ID, convert the value first to its 
  *        metric value kind and then convert it to long long.
  * @param metricID
  *   A CUPTI metric ID.
  * @param metricValue
  *   The metric value corresponding to the metric ID.
  * @param *conversionOfMetricValue
  *   The metric value converted to a long long.
*/
static int convert_metric_value_to_long_long(CUpti_MetricID metricID, CUpti_MetricValue metricValue, long long *conversionOfMetricValue)
{
    SUBDBG("Converting metric value to long long array.\n");

    CUpti_MetricValueKind metricValueKind;
    size_t metricValueKindSize = sizeof(metricValueKind);
    CUpti_MetricAttribute metricAttr = CUPTI_METRIC_ATTR_VALUE_KIND;
    cuptiCheckErrors( cuptiMetricGetAttributePtr(metricID, metricAttr, &metricValueKindSize, &metricValueKind), return PAPI_EMISC );
    switch(metricValueKind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            *conversionOfMetricValue = (long long) metricValue.metricValueDouble;
            return PAPI_OK;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            *conversionOfMetricValue = (long long) metricValue.metricValueUint64;
            return PAPI_OK;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            *conversionOfMetricValue = (long long) metricValue.metricValuePercent * 100;
            return PAPI_OK;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            *conversionOfMetricValue = (long long) metricValue.metricValueThroughput;
            return PAPI_OK;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            *conversionOfMetricValue = (long long) metricValue.metricValueUtilizationLevel;
            return PAPI_OK;
        default:
            SUBDBG("Provided CUpti_MetricValueKind does not exist in switch statement. Email developers.\n");
            return PAPI_EBUG;
    }
}

/** @class create_event_and_metric_table
  * @brief Create and initialize a hash table and structure for the user added events
  *        to be placed.
  * @param totalNumberOfEntries
  *   Total number of user added events we need space for.
  * @param **initializedTable
  *   Variable to store the initialized hash table and structure.
*/
static int create_event_and_metric_table(int totalNumberOfEntries, cuptiu_event_and_metric_table_t **initializedTable)
{
    cuptiu_event_and_metric_table_t *eventTable = (cuptiu_event_and_metric_table_t *) malloc(sizeof(cuptiu_event_and_metric_table_t));
    if (eventTable == NULL) {
        SUBDBG("Failed to allocate memory for evt_table.\n");
        goto fn_fail;
    }

    eventTable->capacity = totalNumberOfEntries;
    eventTable->startTimeStampNs = 0;
    eventTable->totalNumberOfUserAddedNativeEvents = 0;

    int htable_errno = htable_init(&(eventTable->htable));
    if (htable_errno != HTABLE_SUCCESS) {
        SUBDBG("Failed to initialize hash table.\n");
        destroy_event_and_metric_table(&eventTable);
        goto fn_fail;
    }

    *initializedTable = eventTable;

    return PAPI_OK;

fn_fail:
    *initializedTable = NULL;
    return PAPI_ENOMEM;
}

/** @class destroy_event_and_metric_table
  * @brief Destroy the initialized event and metric table.
  *
  * @param **initializedTable
  *   Variable that has the initialized hash table and structure.
*/
static void destroy_event_and_metric_table(cuptiu_event_and_metric_table_t **initializedTable)
{
    cuptiu_event_and_metric_table_t *eventTable = *initializedTable;
    if (eventTable == NULL) {
        return;
    }

    int i;
    for (i = 0; i < eventTable->totalNumberOfUserAddedNativeEvents; i++) {
        free(eventTable->idsThatMakeupAUserAddedEventArray[i]);
        free(eventTable->cumulativeValuesArray[i]);
    }

    if (eventTable->htable) {
        htable_shutdown(eventTable->htable);
        eventTable->htable = NULL;
    }

    free(eventTable);
    *initializedTable = NULL;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section  Functions specific to the PAPI native event interface
 *            (e.g. name to code).
 *
 *  @{
 */

/** @class cuptie_evt_code_to_descr
  * @brief Convert a Cuda native event code to description.
  *        This is not implemented as it should never be called
  *        due to evt_code_to_info being implemented.
  *
  * @param event_code
  *   Cuda native event code.
  * @param *descr
  *   Stores the corresponding description. 
  * @param len
  *   Maximum length allowed for the description.
*/
int cuptie_evt_code_to_descr(uint32_t event_code, char *descr, int len)
{
    // code_to_descr should never be called as evt_code_to_info is implemented 
    SUBDBG("EXITING: Code to description not supported.\n");
    return PAPI_ENOIMPL;
}

/** @class cuptie_evt_enum
  * @brief Enumerate Cuda native events.
  * 
  * @param *event_code
  *   Cuda native event code. 
  * @param modifier
  *   Modifies the search logic. Three modifiers are used PAPI_ENUM_FIRST,
  *   PAPI_ENUM_EVENTS, and PAPI_NTV_ENUM_UMASKS.
*/
int cuptie_evt_enum(uint32_t *event_code, int modifier)
{

    int papi_errno = PAPI_OK;
    event_info_t info;
    SUBDBG("ENTER: event_code: %u, modifier: %d\n", *event_code, modifier);

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if(cuptiu_table_p->count == 0) {
                papi_errno = PAPI_ENOEVNT;
                break;
            }
            info.device = 0;
            info.flags = 0;
            info.nameid = 0;
            papi_errno = event_and_metric_id_create(&info, event_code);
            break;
        case PAPI_ENUM_EVENTS:
            papi_errno = event_and_metric_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }
            if (cuptiu_table_p->count > info.nameid + 1) {
                info.device = 0;
                info.flags = 0;
                info.nameid++;
                papi_errno = event_and_metric_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_ENOEVNT;
            break;
        case PAPI_NTV_ENUM_UMASKS:
            papi_errno = event_and_metric_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }
            if (info.flags == 0){
                info.device = 0;
                info.flags = DEVICE_FLAG;
                papi_errno = event_and_metric_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_ENOEVNT;
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
}

/** @class cuptip_evt_code_to_info
  * @brief Takes a Cuda native event code and collects info such as Cuda native 
  *        event name, Cuda native event description, and number of devices. 
  * @param event_code
  *   Cuda native event code. 
  * @param *info
  *   Structure for member variables such as symbol, short description, and 
  *   long desctiption. 
*/
int cuptie_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info)
{   
    SUBDBG("ENTERING: Converting Cuda native event code to info.\n");
    
    event_info_t inf;
    int papi_errno = event_and_metric_id_to_info(event_code, &inf);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    
    int strLen;
    switch (inf.flags) {
        case 0:
        {
            // Store details for the Cuda event or metric
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].name );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write Cuda event or metric name into info->symbol.\n");
                return PAPI_EBUF;
            }
            
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write Cuda event or metric description into info->long_descr.\n");
                return PAPI_EBUF;
            }
            SUBDBG("EXITING: Completed converting native event code to info.\n");
            return PAPI_OK;
        }
        case DEVICE_FLAG:
        {
            int init_metric_device_idx;
            char devices[PAPI_MAX_STR_LEN] = { 0 };
            int i; 
            for (i = 0; i < numDevicesOnMachine; ++i) {
                if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                    // Store the first device found to use with :device=#, as on a heterogeneous
                    // system events may not appear on each device
                    if (devices[0] == '\0') {
                        init_metric_device_idx = i;
                    }
                    
                    strLen = snprintf(devices + strlen(devices), PAPI_HUGE_STR_LEN - strlen(devices), "%i,", i);
                    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN - strlen(devices)) {
                        SUBDBG("Failed to fully write device qualifier into devices.\n");
                        return PAPI_EBUF;
                    }
                }
            }
            *(devices + strlen(devices) - 1) = 0;
            
            // Store details for the Cuda event or metric
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:device=%i", cuptiu_table_p->events[inf.nameid].name, init_metric_device_idx );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write Cuda event or metric name into info->symbol.\n");
                return PAPI_EBUF;
            }
            
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                               cuptiu_table_p->events[inf.nameid].desc, devices );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write Cuda event or metric description into info->long_descr.\n");
                return PAPI_EBUF;
            }
            SUBDBG("EXITING: Completed converting native event code to info.\n");
            return PAPI_OK;
        }
        default:
            SUBDBG("EXITING: Switch statement does not have appropriate case.");
            return PAPI_EBUG;
    }
}

/** @class cuptie_evt_name_to_code
  * @brief Convert a Cuda native event name to code.
  *
  * @param *name
  *   The Cuda native event name.
  * @param *event_code
  *   Stores the corresponding native event code.
*/
int cuptie_evt_name_to_code(const char *name, uint32_t *event_code)
{
    SUBDBG("ENTERING: Converting %s to an event code.\n", name);

    int device;
    int papi_errno = evt_name_to_device(name, &device);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    char base[PAPI_MAX_STR_LEN] = { 0 };
    papi_errno = evt_name_to_basename(name, base, PAPI_MAX_STR_LEN);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuptiu_event_and_metric_t *event;
    int htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
        return papi_errno;
    }

    // Flags = DEVICE_FLAG will need to be updated if more qualifiers are added,
    // see implementation in rocm (roc_profiler.c)
    int flags = (device >= 0) ? DEVICE_FLAG : 0;
    if (flags == 0){
        return PAPI_EINVAL;
    }

    int nameid = (int) (event - cuptiu_table_p->events);
    event_info_t info = { device, flags, nameid };
    papi_errno = event_and_metric_id_create(&info, event_code);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = event_and_metric_id_to_info(*event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Section handles if the Cuda component is partially disabled
    int *enabledCudaDeviceIds, cudaCmpPartial;
    size_t cudaEnabledDevicesCnt;
    cuptic_partial(&cudaCmpPartial, &enabledCudaDeviceIds, &cudaEnabledDevicesCnt);
    if (cudaCmpPartial) {
        papi_errno = PAPI_PARTIAL;

        int i; 
        for (i = 0; i < cudaEnabledDevicesCnt; i++) {
            if (device == enabledCudaDeviceIds[i]) {
                papi_errno = PAPI_OK;
                break;
            }
        }
    } 

    SUBDBG("EXITING: Converting a Cuda native event name to code completed.\n");
    return papi_errno;
}

/** @class cuptie_evt_code_to_name
  * @brief Convert a Cuda native event code to name.
  *
  * @param event_code
  *   The Cuda native event code we want to convert to a name.
  * @param *name
  *   Stores the corresponding event name.
  * @param len
  *   Maximum length allowed for the name.
*/
int cuptie_evt_code_to_name(uint32_t event_code, char *name, int len)
{
    event_info_t info;
    int papi_errno = event_and_metric_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int strLen;
    switch (info.flags) {
        case (DEVICE_FLAG):
            strLen = snprintf(name, len, "%s:device=%i", cuptiu_table_p->events[info.nameid].name, info.device);
            if (strLen < 0 || strLen > len) {
                SUBDBG("Failed to fully write the Cuda native event name.\n");
                return PAPI_EBUF;
            }
            break;
        default:
            strLen = snprintf(name, len, "%s", cuptiu_table_p->events[info.nameid].name);
            if (strLen < 0 || strLen > len) {
                SUBDBG("Failed to fully write the Cuda native event name.\n");
                return PAPI_EBUF;
            }
            break;
   }

   return papi_errno;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section  Helper functions specific to the PAPI native event interface
 *            (e.g. name to code).
 *
 *  @{
 */

/** @class event_and_metric_id_create
  * @brief Create event ID. Function is needed for cuptip_event_enum.
  *
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
  * @param *event_id
  *   Created event id.
*/
static int event_and_metric_id_create(event_info_t *info, uint32_t *event_id)
{
    *event_id  = (uint32_t)(info->device   << DEVICE_SHIFT);
    *event_id |= (uint32_t)(info->flags    << QLMASK_SHIFT);
    *event_id |= (uint32_t)(info->nameid   << NAMEID_SHIFT);

    return PAPI_OK;
}

/** @class event_and_metric_id_to_info
  * @brief Convert event id to info. Function is needed for cuptip_event_enum.
  *
  * @param event_id
  *   An event id.
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
*/
static int event_and_metric_id_to_info(uint32_t event_id, event_info_t *info)
{
    info->device   = (uint32_t)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->flags    = (uint32_t)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid   = (uint32_t)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->device >= numDevicesOnMachine) {
        return PAPI_ENOEVNT;
    }    

    if (0 == (info->flags & DEVICE_FLAG) && info->device > 0) {
        return PAPI_ENOEVNT;
    }    

    if (info->nameid >= cuptiu_table_p->count) {
        return PAPI_ENOEVNT;
    }

    return PAPI_OK;
}

/** @class evt_name_to_device
  * @brief Return the device number for a user provided Cuda native event.
  *        This can be done with a device qualifier present (:device=#) or
  *        we internally find the first device the native event exists for.
  * @param *name
  *   Cuda native event name with a device qualifier appended.
  * @param *device
  *   Device number.
*/
static int evt_name_to_device(const char *name, int *device)
{
    char *p = strstr(name, ":device=");
    // User did provide :device=# qualifier
    if (p) {
        *device = (int) strtol(p + strlen(":device="), NULL, 10);
    }
    // User did not provide :device=# qualifier
    else {
        int i, htable_errno;
        cuptiu_event_and_metric_t *event;

        htable_errno = htable_find(cuptiu_table_p->htable, name, (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_EINVAL;
        }

        // Search for the first device the event exists for.
        for (i = 0; i < numDevicesOnMachine; ++i) {
            if (cuptiu_dev_check(event->device_map, i)) {
                *device = i;
                break;
            }
        }
    }
    return PAPI_OK;
}

/** @class evt_name_to_basename
  * @brief Convert a Cuda native event name with a device qualifer appended to 
  *        it, back to the base Cuda native event name provided by NVIDIA.
  * @param *name
  *   Cuda native event name with a device qualifier appended.
  * @param *base
  *   Base Cuda native event name (excludes device qualifier).
  * @param len
  *   Maximum alloted characters for base Cuda native event name. 
*/
static int evt_name_to_basename(const char *name, char *base, int len)
{
    char *p = strstr(name, ":");

    if (p) {
        if (len < (int)(p - name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t)(p - name));
    } else {
        if (len < (int) strlen(name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t) len);
    }
    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section Functions related to the overall deinitialization of the Event and
 *           Metric workflow.
 *
 *  @{
 */

/** @class cuptie_shutdown
  * @brief Deinitialize cupti profiler api and unload function pointers.
*/
int cuptie_shutdown(void)
{
    SUBDBG("ENTERING: Shutting down.\n");

    int papi_errno = deinitialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        return PAPI_OK;
    }

    shutdown_event_table();

    unload_event_and_metric_sym();
    unload_cupti_profiler_sym();

    SUBDBG("EXITING: Shutdown completed.\n");
    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/


/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @section Helper functions related to the overall deinitialization of 
 *           the Event and Metric workflow.
 *
 *  @{
 */

/** @class deinitialize_cupti_profiler_api
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerDeInitialize.
*/
static int deinitialize_cupti_profiler_api(void)     
{
    SUBDBG("ENTERING: Deinitializing CUPTI Profiler API.\n");

    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    profilerDeInitializeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerDeInitializeEventAndMetricPtr(&profilerDeInitializeParams), return PAPI_EMISC );

    SUBDBG("EXITING: Deinitialization of CUPTI Profiler API completed.\n");
    return PAPI_OK;
}

/** @class shutdown_event_and_metric_table
  * @brief Free event and metric table allocated memory.
*/
static void shutdown_event_table(void)
{   
    cuptiu_table_p->count = 0;
    
    free(cuptiu_table_p->avail_gpu_info);
    cuptiu_table_p->avail_gpu_info = NULL;
    
    free(cuptiu_table_p->events);
    cuptiu_table_p->events = NULL;
    
    free(cuptiu_table_p);
    cuptiu_table_p = NULL;
}

/** @class unload_event_and_metric_sym
  * @brief Unload Event and Metric API functions.
*/
static void unload_event_and_metric_sym(void)
{
    SUBDBG("ENTERING: Unloading Event and Metric API functions.\n");

    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }

    // Event API  //
    // Enumeration
    cuptiDeviceGetNumEventDomainsPtr      = NULL;
    cuptiDeviceEnumEventDomainsPtr        = NULL;
    cuptiEventDomainGetNumEventsPtr       = NULL;
    cuptiEventDomainEnumEventsPtr         = NULL;
    cuptiEventGetAttributePtr             = NULL;
    // Managing an EventGroup
    cuptiEventGetIdFromNamePtr            = NULL;
    cuptiEventGroupSetsCreatePtr          = NULL;
    cuptiEventGroupSetEnablePtr           = NULL;
    cuptiEventGroupSetAttributePtr        = NULL;
    cuptiEventGroupGetAttributePtr        = NULL;
    cuptiDeviceGetEventDomainAttributePtr = NULL;
    cuptiEventGroupResetAllEventsPtr      = NULL;
    cuptiEventGroupSetDisablePtr          = NULL;
    cuptiEventGroupSetsDestroyPtr         = NULL;
    // Reading event values
    cuptiEventGroupReadAllEventsPtr       = NULL;

    // Metric API //
    // Enumeration
    cuptiDeviceGetNumMetricsPtr           = NULL;
    cuptiDeviceEnumMetricsPtr             = NULL;
    cuptiMetricGetAttributePtr            = NULL;
    // Managing an EventGroup
    cuptiMetricGetIdFromNamePtr           = NULL;
    cuptiMetricCreateEventGroupSetsPtr    = NULL;
    // Reading metric values
    cuptiMetricGetNumEventsPtr            = NULL;
    cuptiMetricEnumEventsPtr              = NULL;
    cuptiGetTimestampPtr                  = NULL;
    cuptiMetricGetValuePtr                = NULL;

    SUBDBG("EXITING: Completed unloading Event and Metric API functions.\n");
    return;
}

/** @class unload_cupti_profiler_sym
  * @brief Unload CUPTI Profiler API functions.
*/
static void unload_cupti_profiler_sym(void) 
{
    SUBDBG("ENTERING: Unloading CUPTI Profiler API functions.\n");

    cuptiProfilerInitializeEventAndMetricPtr            = NULL;
    cuptiProfilerDeInitializeEventAndMetricPtr          = NULL;

    SUBDBG("EXITING: Completed unloading CUPTI Profiler API functions.\n");
    return;
}

/**
 *  @}
 ******************************************************************************/
