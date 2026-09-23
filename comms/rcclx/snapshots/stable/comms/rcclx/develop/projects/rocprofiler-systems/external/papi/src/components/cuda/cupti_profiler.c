/**
 * @file    cupti_profiler.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 * @author  Dong Jun WOun  dwoun@vols.utk.edu
 */

#include <dlfcn.h>
#include <stdbool.h>
#include <papi.h>
#include "papi_memory.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include "papi_cupti_common.h"
#include "cupti_profiler.h"
#include "cupti_config.h"
#include "lcuda_debug.h"
#include "htable.h"

/**
 * Event identifier encoding format:
 * +--------+------+-------+----+------------+
 * | unused | stat |  dev  | ql |   nameid   |
 * +--------+------+-------+----+------------+
 *
 * unused    : 2  bits 
 * stat      : 3  bit  ([0 -   8] stats)
 * device    : 7  bits ([0 - 127] devices)
 * qlmask    : 2  bits (qualifier mask)
 * nameid    : 18: bits (roughly > 262 Thousand event names)
 */
#define EVENTS_WIDTH (sizeof(uint32_t) * 8)
#define STAT_WIDTH   ( 3)
#define DEVICE_WIDTH ( 7)
#define QLMASK_WIDTH ( 2) 
#define NAMEID_WIDTH (18)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH - STAT_WIDTH)
#define STAT_SHIFT   (EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH - DEVICE_WIDTH)
#define QLMASK_SHIFT (DEVICE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define STAT_MASK    ((0xFFFFFFFF >> (EVENTS_WIDTH - STAT_WIDTH)) << STAT_SHIFT)
#define DEVICE_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define STAT_FLAG    (0x2)
#define DEVICE_FLAG  (0x1)

#define NUM_STATS_QUALS 7
char stats[NUM_STATS_QUALS][PAPI_MIN_STR_LEN] = {"avg", "sum", "min", "max", "max_rate", "pct", "ratio"};

typedef struct {
    int stat;
    int device;
    int flags;
    int nameid;
} event_info_t;

typedef struct byte_array_s {
    int      size;
    uint8_t *data;
} byte_array_t;

typedef struct cuptip_gpu_state_s {
    int                    dev_id;
    cuptiu_event_table_t  *added_events;
    int                   numberOfRawMetricRequests;
    NVPA_RawMetricRequest *rawMetricRequests;
    byte_array_t          counterDataPrefixImage;
    byte_array_t          configImage;
    byte_array_t          counterDataImage;
    byte_array_t          counterDataScratchBuffer;
    byte_array_t          counterAvailabilityImage;
} cuptip_gpu_state_t;

struct cuptip_control_s {
    cuptip_gpu_state_t *gpu_ctl;
    long long           *counters;
    int                 read_count;
    int                 running;
    cuptic_info_t       info;
};

static void *dl_nvpw;
static int numDevicesOnMachine;
static cuptiu_event_table_t *cuptiu_table_p;

// Cupti Profiler API function pointers //
CUptiResult ( *cuptiProfilerInitializePtr ) (CUpti_Profiler_Initialize_Params* params);
CUptiResult ( *cuptiProfilerDeInitializePtr ) (CUpti_Profiler_DeInitialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializePtr ) (CUpti_Profiler_CounterDataImage_Initialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializeScratchBufferPtr ) (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params);
CUptiResult ( *cuptiProfilerBeginSessionPtr ) (CUpti_Profiler_BeginSession_Params* params);
CUptiResult ( *cuptiProfilerSetConfigPtr ) (CUpti_Profiler_SetConfig_Params* params);
CUptiResult ( *cuptiProfilerBeginPassPtr ) (CUpti_Profiler_BeginPass_Params* params);
CUptiResult ( *cuptiProfilerEnableProfilingPtr ) (CUpti_Profiler_EnableProfiling_Params* params);
CUptiResult ( *cuptiProfilerPushRangePtr ) (CUpti_Profiler_PushRange_Params* params);
CUptiResult ( *cuptiProfilerPopRangePtr ) (CUpti_Profiler_PopRange_Params* params);
CUptiResult ( *cuptiProfilerDisableProfilingPtr ) (CUpti_Profiler_DisableProfiling_Params* params);
CUptiResult ( *cuptiProfilerEndPassPtr ) (CUpti_Profiler_EndPass_Params* params);
CUptiResult ( *cuptiProfilerFlushCounterDataPtr ) (CUpti_Profiler_FlushCounterData_Params* params);
CUptiResult ( *cuptiProfilerUnsetConfigPtr ) (CUpti_Profiler_UnsetConfig_Params* params);
CUptiResult ( *cuptiProfilerEndSessionPtr ) (CUpti_Profiler_EndSession_Params* params);
CUptiResult ( *cuptiProfilerGetCounterAvailabilityPtr ) (CUpti_Profiler_GetCounterAvailability_Params* params);
CUptiResult ( *cuptiFinalizePtr ) (void);

// Function wrappers for the Cupti Profiler API //
static int initialize_cupti_profiler_api(void);
static int deinitialize_cupti_profiler_api(void);
static int enable_profiling(void);
static int begin_pass(void);
static int end_pass(void);
static int push_range(const char *pRangeName);
static int pop_range(void);
static int flush_data(void);
static int disable_profiling(void);
static int unset_config(void);
static int end_session(void);

// Perfworks API function pointers //
// Initialize
NVPA_Status ( *NVPW_InitializeHostPtr ) (NVPW_InitializeHost_Params* params);
// Enumeration
NVPA_Status ( *NVPW_MetricsEvaluator_GetMetricNamesPtr ) (NVPW_MetricsEvaluator_GetMetricNames_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetSupportedSubmetricsPtr ) (NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetCounterPropertiesPtr ) (NVPW_MetricsEvaluator_GetCounterProperties_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetRatioMetricPropertiesPtr ) (NVPW_MetricsEvaluator_GetRatioMetricProperties_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetThroughputMetricPropertiesPtr ) (NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetMetricDimUnitsPtr ) (NVPW_MetricsEvaluator_GetMetricDimUnits_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_DimUnitToStringPtr ) (NVPW_MetricsEvaluator_DimUnitToString_Params* pParams);
// Configuration
NVPA_Status ( *NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequestPtr ) (NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_GetMetricRawDependenciesPtr ) (NVPW_MetricsEvaluator_GetMetricRawDependencies_Params* pParams);
NVPA_Status ( *NVPW_CUDA_RawMetricsConfig_Create_V2Ptr ) (NVPW_CUDA_RawMetricsConfig_Create_V2_Params* pParams);
NVPA_Status ( *NVPW_RawMetricsConfig_GenerateConfigImagePtr ) (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetConfigImagePtr ) (NVPW_RawMetricsConfig_GetConfigImage_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_CreatePtr ) (NVPW_CounterDataBuilder_Create_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_AddMetricsPtr ) (NVPW_CounterDataBuilder_AddMetrics_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_GetCounterDataPrefixPtr ) (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params);
NVPA_Status ( *NVPW_CUDA_CounterDataBuilder_CreatePtr ) (NVPW_CUDA_CounterDataBuilder_Create_Params* pParams);
NVPA_Status ( *NVPW_RawMetricsConfig_SetCounterAvailabilityPtr ) (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params);
// Evaluation
NVPA_Status ( *NVPW_MetricsEvaluator_SetDeviceAttributesPtr ) (NVPW_MetricsEvaluator_SetDeviceAttributes_Params* pParams);
NVPA_Status ( *NVPW_MetricsEvaluator_EvaluateToGpuValuesPtr ) (NVPW_MetricsEvaluator_EvaluateToGpuValues_Params* pParams);
// Used in both enumeration and evaluation
NVPA_Status ( *NVPW_CUDA_MetricsEvaluator_InitializePtr ) (NVPW_CUDA_MetricsEvaluator_Initialize_Params* pParams);
NVPA_Status ( *NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr ) (NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params* pParams);
NVPA_Status ( *NVPW_RawMetricsConfig_GetNumPassesPtr ) (NVPW_RawMetricsConfig_GetNumPasses_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_BeginPassGroupPtr ) (NVPW_RawMetricsConfig_BeginPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_EndPassGroupPtr ) (NVPW_RawMetricsConfig_EndPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_AddMetricsPtr ) (NVPW_RawMetricsConfig_AddMetrics_Params* params);
// Destroy
NVPA_Status ( *NVPW_RawMetricsConfig_DestroyPtr ) (NVPW_RawMetricsConfig_Destroy_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_DestroyPtr ) (NVPW_CounterDataBuilder_Destroy_Params* params);
NVPA_Status ( *NVPW_MetricsEvaluator_DestroyPtr ) (NVPW_MetricsEvaluator_Destroy_Params* pParams);
// Misc.
NVPA_Status ( *NVPW_GetSupportedChipNamesPtr ) (NVPW_GetSupportedChipNames_Params* params);

// Helper functions for the MetricsEvaluator API //
// Initialize
static int initialize_perfworks_api(void);
// Enumeration
static int enumerate_metrics_for_unique_devices(const char *pChipName, int *totalNumMetrics, char ***arrayOfMetricNames);
static int get_rollup_metrics(NVPW_RollupOp rollupMetric, char **strRollupMetric);
static int get_supported_submetrics(NVPW_Submetric subMetric, char **strSubMetric);
static int get_metric_properties(const char *pChipName, const char *metricName, char *fullMetricDescription);
static int get_number_of_passes_for_info(const char *pChipName, NVPW_MetricsEvaluator *pMetricsEvaluator, NVPW_MetricEvalRequest *metricEvalRequest, int *numOfPasses);
// Configuration
static int get_metric_eval_request(NVPW_MetricsEvaluator *metricEvaluator, const char *metricName, NVPW_MetricEvalRequest *pMetricEvalRequest);
static int create_raw_metric_requests(NVPW_MetricsEvaluator *pMetricsEvaluator, NVPW_MetricEvalRequest *metricEvalRequest, NVPA_RawMetricRequest **rawMetricRequests, int *rawMetricRequestsCount);
// Metric Evaluation
static int get_number_of_passes_for_eventsets(const char *pChipName, const char *metricName, int *numOfPasses);
static int get_evaluated_metric_values(NVPW_MetricsEvaluator *pMetricsEvaluator, cuptip_gpu_state_t *gpu_ctl, long long *evaluatedMetricValues);
// Destroy MetricsEvaluator
static int destroy_metrics_evaluator(NVPW_MetricsEvaluator *pMetricsEvaluator);

// Helper functions for profiling //
static int start_profiling_session(byte_array_t counterDataImage, byte_array_t counterDataScratchBufferSize, byte_array_t configImage);
static int end_profiling_session(void);
static int get_config_image(const char *chipName, const uint8_t *pCounterAvailabilityImageData, NVPA_RawMetricRequest *rawMetricRequests, int rmr_count, byte_array_t *configImage);
static int get_counter_data_prefix_image(const char *chipName, NVPA_RawMetricRequest *rawMetricRequests, int rmr_count, byte_array_t *counterDataPrefixImage);
static int get_counter_data_image(byte_array_t counterDataPrefixImage, byte_array_t *counterDataScratchBuffer, byte_array_t *counterDataImage);
static int get_event_collection_method(const char *evt_name);
static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl);
static void free_and_reset_configuration_images(cuptip_gpu_state_t *gpu_ctl);

// Functions related to Cuda component hash tables
static int init_main_htable(void);
static int init_event_table(void);
static void shutdown_event_table(void);
static void shutdown_event_stats_table(void);

// Functions related to NVIDIA device chips
static int assign_chipnames_for_a_device_index(void);
static int find_same_chipname(int dev_id);

// Functions related to the native event interface
static int get_ntv_events(cuptiu_event_table_t *evt_table, const char *evt_name, int dev_id);
static int verify_user_added_events(uint32_t *events_id, int num_events, cuptip_control_t state);
static int evt_id_to_info(uint32_t event_id, event_info_t *info);
static int evt_id_create(event_info_t *info, uint32_t *event_id);
static int evt_code_to_name(uint32_t event_code, char *name, int len);
static int evt_name_to_basename(const char *name, char *base, int len);
static int evt_name_to_device(const char *name, int *device, const char *base);
static int evt_name_to_stat(const char *name, int *stat, const char *base);
static int cuda_verify_qualifiers_are_not_repeated(const char *eventName);
static int cuda_verify_qualifiers_are_valid(const char *qualifiers);
static int cuda_verify_qualifiers_do_not_contain_junk(int flag, char *qualifierName, int equalitySignPosition, int *qualifierValue);

// Functions related to the stats qualifier
static int restructure_event_name(const char *input, char *output, char *base, char *stat);
static int is_stat(const char *token);

// Functions related to a partially disabled Cuda component
static int determine_dev_cc_major(int dev_id);

// Load and unload function pointers
static int load_cupti_perf_sym(void);
static int unload_cupti_perf_sym(void);
static int load_nvpw_sym(void);
static int unload_nvpw_sym(void);

/** @class load_cupti_perf_sym
  * @brief Load cupti functions and assign to function pointers.
*/
static int load_cupti_perf_sym(void)
{
    COMPDBG("Entering.\n");
    if (dl_cupti == NULL) {
        ERRDBG("libcupti.so should already be loaded.\n");
        return PAPI_EMISC;
    }

    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndSession");
    cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerGetCounterAvailability");
    cuptiFinalizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiFinalize");

    return PAPI_OK;
}

/** @class unload_cupti_perf_sym
  * @brief Unload cupti function pointers.
*/
static int unload_cupti_perf_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiProfilerInitializePtr                                 = NULL;
    cuptiProfilerDeInitializePtr                               = NULL;
    cuptiProfilerCounterDataImageCalculateSizePtr              = NULL;
    cuptiProfilerCounterDataImageInitializePtr                 = NULL;
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = NULL;
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr    = NULL;
    cuptiProfilerBeginSessionPtr                               = NULL;
    cuptiProfilerSetConfigPtr                                  = NULL;
    cuptiProfilerBeginPassPtr                                  = NULL;
    cuptiProfilerEnableProfilingPtr                            = NULL;
    cuptiProfilerPushRangePtr                                  = NULL;
    cuptiProfilerPopRangePtr                                   = NULL;
    cuptiProfilerDisableProfilingPtr                           = NULL;
    cuptiProfilerEndPassPtr                                    = NULL;
    cuptiProfilerFlushCounterDataPtr                           = NULL;
    cuptiProfilerUnsetConfigPtr                                = NULL;
    cuptiProfilerEndSessionPtr                                 = NULL;
    cuptiProfilerGetCounterAvailabilityPtr                     = NULL;
    cuptiFinalizePtr                                           = NULL;
    return PAPI_OK;
}

/**@class load_nvpw_sym
 * @brief Search for a variation of the shared object libnvperf_host.
 *        Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_PERFWORKS, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect a variation of the shared object libnvperf_host from
 *    PAPI_CUDA_PERFWORKS or it is not set, we will search the path defined with PAPI_CUDA_ROOT;
 *    as this is supposed to always be set.
 * 3. If we fail to collect a variation of the shared object libnvperf_host from steps 1 and 2,
 *    then we will search the linux default directories listed by /etc/ld.so.conf. As a note,
 *    updating the LD_LIBRARY_PATH is advised for this option.
 * 4. We use dlopen to search for a variation of the shared object libnvperf_host.
 *    If this fails, then we failed to find a variation of the shared object libnvperf_host.
 */
static int load_nvpw_sym(void)
{
    int soNamesToSearchCount = 3;
    const char *soNamesToSearchFor[] = {"libnvperf_host.so", "libnvperf_host.so.1", "libnvperf_host"};

    // If a user set PAPI_CUDA_PERFWORKS with a path, then search it for the shared object (takes precedent over PAPI_CUDA_ROOT)
    char *papi_cuda_perfworks = getenv("PAPI_CUDA_PERFWORKS");
    if (papi_cuda_perfworks) {
        dl_nvpw = search_and_load_shared_objects(papi_cuda_perfworks, NULL, soNamesToSearchFor, soNamesToSearchCount);
    }

    char *soMainName = "libnvperf_host";
    // If a user set PAPI_CUDA_ROOT with a path and we did not already find the shared object, then search it for the shared object
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_nvpw) {
          dl_nvpw = search_and_load_shared_objects(papi_cuda_root, soMainName, soNamesToSearchFor, soNamesToSearchCount);
    }

    // Last ditch effort to find a variation of libnvperf_host, see dlopen manpages for how search occurs
    if (!dl_nvpw) {
        dl_nvpw = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
        if (!dl_nvpw) {
            ERRDBG("Loading libnvperf_host.so failed.\n");
            goto fn_fail;
        }
    }

    // Initialize
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_InitializeHost");
    // Enumeration
    NVPW_MetricsEvaluator_GetMetricNamesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetMetricNames");
    NVPW_MetricsEvaluator_GetSupportedSubmetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetSupportedSubmetrics");
    NVPW_MetricsEvaluator_GetCounterPropertiesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetCounterProperties");
    NVPW_MetricsEvaluator_GetRatioMetricPropertiesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetRatioMetricProperties");
    NVPW_MetricsEvaluator_GetThroughputMetricPropertiesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetThroughputMetricProperties");
    NVPW_MetricsEvaluator_GetMetricDimUnitsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetMetricDimUnits");
    NVPW_MetricsEvaluator_DimUnitToStringPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_DimUnitToString"); 
    // Configuration
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequestPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest");
    NVPW_MetricsEvaluator_GetMetricRawDependenciesPtr =  DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_GetMetricRawDependencies");
    NVPW_CUDA_RawMetricsConfig_Create_V2Ptr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_RawMetricsConfig_Create_V2");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CUDA_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_CounterDataBuilder_Create");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_SetCounterAvailability"); 
    // Evaluation
    NVPW_MetricsEvaluator_SetDeviceAttributesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_SetDeviceAttributes");
    NVPW_MetricsEvaluator_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_EvaluateToGpuValues");
    // Used in both enumeration and evaluation
    NVPW_CUDA_MetricsEvaluator_InitializePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_MetricsEvaluator_Initialize");
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr  = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_AddMetrics");
    // Destroy
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_Destroy");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Destroy");
    NVPW_MetricsEvaluator_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsEvaluator_Destroy");
    // Misc.
    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_GetSupportedChipNames");

    Dl_info info;
    dladdr(NVPW_GetSupportedChipNamesPtr, &info);
    LOGDBG("NVPW library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class unload_nvpw_sym
  * @brief Unload nvperf function pointers.
*/
static int unload_nvpw_sym(void)
{
    if (dl_nvpw) {
        dlclose(dl_nvpw);
        dl_nvpw = NULL;
    }

    // Initialize
    NVPW_InitializeHostPtr                                        = NULL;
    // Enumeration
    NVPW_MetricsEvaluator_GetMetricNamesPtr                       = NULL;
    NVPW_MetricsEvaluator_GetSupportedSubmetricsPtr               = NULL;
    NVPW_MetricsEvaluator_GetCounterPropertiesPtr                 = NULL;
    NVPW_MetricsEvaluator_GetRatioMetricPropertiesPtr             = NULL;
    NVPW_MetricsEvaluator_GetThroughputMetricPropertiesPtr        = NULL;
    NVPW_MetricsEvaluator_GetMetricDimUnitsPtr                    = NULL;
    NVPW_MetricsEvaluator_DimUnitToStringPtr                      = NULL;
    // Configuration
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequestPtr = NULL;
    NVPW_MetricsEvaluator_GetMetricRawDependenciesPtr             = NULL;
    NVPW_CUDA_RawMetricsConfig_Create_V2Ptr                       = NULL;
    NVPW_RawMetricsConfig_GenerateConfigImagePtr                  = NULL;
    NVPW_RawMetricsConfig_GetConfigImagePtr                       = NULL;
    NVPW_CounterDataBuilder_CreatePtr                             = NULL;
    NVPW_CounterDataBuilder_AddMetricsPtr                         = NULL;
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr               = NULL;
    NVPW_CUDA_CounterDataBuilder_CreatePtr                        = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr               = NULL;
    // Evaluation
    NVPW_MetricsEvaluator_SetDeviceAttributesPtr                  = NULL;
    NVPW_MetricsEvaluator_EvaluateToGpuValuesPtr                  = NULL;
    // Used in both enumeration and evaluation
    NVPW_CUDA_MetricsEvaluator_InitializePtr                      = NULL;
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr      = NULL;
    NVPW_RawMetricsConfig_GetNumPassesPtr                         = NULL;
    NVPW_RawMetricsConfig_BeginPassGroupPtr                       = NULL;
    NVPW_RawMetricsConfig_EndPassGroupPtr                         = NULL;
    NVPW_RawMetricsConfig_AddMetricsPtr                           = NULL;
    // Destroy
    NVPW_RawMetricsConfig_DestroyPtr                              = NULL;
    NVPW_CounterDataBuilder_DestroyPtr                            = NULL;
    NVPW_MetricsEvaluator_DestroyPtr                              = NULL;
    // Misc.
    NVPW_GetSupportedChipNamesPtr                                 = NULL;

    return PAPI_OK;
}

/** @class initialize_perfworks_api
  * @brief Initialize the Perfworks API.
*/
static int initialize_perfworks_api(void)
{
    COMPDBG("Entering.\n");

    NVPW_InitializeHost_Params perfInitHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    perfInitHostParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_InitializeHostPtr(&perfInitHostParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class get_counter_availability
  * @brief Query counter availability. Helps to filter unavailable raw metrics on host.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   dev_id, rawMetricRequests, numberOfRawMetricRequests, and more.
*/
static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl)
{
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.pPriv = NULL;
    getCounterAvailabilityParams.ctx = NULL; // If NULL, the current CUcontext is used
    getCounterAvailabilityParams.pCounterAvailabilityImage = NULL;
    cuptiCheckErrors( cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams), return PAPI_EMISC );

    // Allocate the necessary memory for data
    gpu_ctl->counterAvailabilityImage.size = getCounterAvailabilityParams.counterAvailabilityImageSize;
    gpu_ctl->counterAvailabilityImage.data = (uint8_t *) malloc(gpu_ctl->counterAvailabilityImage.size);
    if (gpu_ctl->counterAvailabilityImage.data == NULL) {
        ERRDBG("Failed to allocate memory for counterAvailabilityImage.data.\n");
        return PAPI_ENOMEM;
    }

    getCounterAvailabilityParams.pCounterAvailabilityImage = gpu_ctl->counterAvailabilityImage.data;
    cuptiCheckErrors( cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class free_and_reset_configuration_images
  * @brief Free and reset the configuration images created in
  *        cuptip_ctx_start.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   dev_id, rawMetricRequests, numberOfRawMetricRequests, and more.
*/
void free_and_reset_configuration_images(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    // Note that you can find the memory allocation for the below variables
    // in cuptip_ctx_start as of April 21st, 2025
    free(gpu_ctl->configImage.data);
    gpu_ctl->configImage.data = NULL;
    gpu_ctl->configImage.size = 0;

    free(gpu_ctl->counterDataPrefixImage.data);
    gpu_ctl->counterDataPrefixImage.data = NULL;
    gpu_ctl->counterDataPrefixImage.size = 0;

    free(gpu_ctl->counterDataScratchBuffer.data);
    gpu_ctl->counterDataScratchBuffer.data = NULL;
    gpu_ctl->counterDataScratchBuffer.size = 0;

    free(gpu_ctl->counterDataImage.data);
    gpu_ctl->counterDataImage.data = NULL;
    gpu_ctl->counterDataImage.size = 0; 
    
    free(gpu_ctl->counterAvailabilityImage.data);
    gpu_ctl->counterAvailabilityImage.data = NULL;
    gpu_ctl->counterAvailabilityImage.size = 0;
}

/** @class find_same_chipname
  * @brief Check to see if chipnames are identical.
  * 
  * @param dev_id
  *   A gpu id number, e.g 0, 1, 2, etc.
*/
static int find_same_chipname(int dev_id)
{
    int i;
    for (i = 0; i < dev_id; i++) {
        if (!strcmp(cuptiu_table_p->avail_gpu_info[dev_id].chipName, cuptiu_table_p->avail_gpu_info[i].chipName)) {
            return i;
        }
    }
    return -1;
}

/** @class init_main_htable
 *  @brief Initialize the main htable used to collect metrics.
*/
static int init_main_htable(void)
{
    // Allocate (2 ^ NAMEID_WIDTH) metric names, this matches the
    // number of bits for the event encoding format
    int i, val = 1, base = 2;
    for (i = 0; i < NAMEID_WIDTH; i++) {
        val *= base;
    }    
   
    cuptiu_table_p = (cuptiu_event_table_t *) malloc(sizeof(cuptiu_event_table_t));
    if (cuptiu_table_p == NULL) {
        ERRDBG("Failed to allocate memory for cuptiu_table_p.\n");
        return PAPI_ENOMEM;
    }
    cuptiu_table_p->capacity = val; 
    cuptiu_table_p->count = 0;
    cuptiu_table_p->event_stats_count = 0;

    cuptiu_table_p->events = (cuptiu_event_t *) calloc(val, sizeof(cuptiu_event_t));
    if (cuptiu_table_p->events == NULL) {
        ERRDBG("Failed to allocate memory for cuptiu_table_p->events.\n");
        return PAPI_ENOMEM;
    }

    cuptiu_table_p->event_stats = (StringVector *) calloc(val, sizeof(StringVector));
    if (cuptiu_table_p->event_stats == NULL) {
        ERRDBG("Failed to allocate memory for cuptiu_table_p->event_stats.\n");
        return PAPI_ENOMEM;
    }

    cuptiu_table_p->avail_gpu_info = (gpu_record_t *) calloc(numDevicesOnMachine, sizeof(gpu_record_t));
    if (cuptiu_table_p->avail_gpu_info == NULL) {
        ERRDBG("Failed to allocate memory for cuptiu_table_p->avail_gpu_info.\n");
        return PAPI_ENOMEM;
    }

    // Initialize the main hash table for metric collection
    htable_init(&cuptiu_table_p->htable);

    return PAPI_OK;
}

/** @class cuptip_init
  * @brief Load and initialize API's.  
*/
int cuptip_init(void)
{
    COMPDBG("Entering.\n");

    int papi_errno = load_cupti_perf_sym();
    papi_errno += load_nvpw_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Unable to load CUDA library functions.");
        return papi_errno;
    }

    // Collect the number of devices on the machine
    papi_errno = cuptic_device_get_count(&numDevicesOnMachine);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (numDevicesOnMachine <= 0) {
        cuptic_err_set_last("No GPUs found on system.");
        return PAPI_ECMP;
    }
   
    // Initialize the Cupti Profiler and Perfworks API's
    papi_errno = initialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failed to initialize the CUPTI Profiler API. A possible reason is a mismatched Cuda Toolkit and NVIDIA architecture."
                            " Try setting PAPI_CUDA_ROOT to a newer Cuda Toolkit version.");
        return PAPI_EMISC;
    }

    papi_errno = initialize_perfworks_api();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Failed to initialize the PerfWorks Metrics API. A possible reason is a mismatched Cuda Toolkit and NVIDIA architecture."
                            " Try setting PAPI_CUDA_ROOT to a newer Cuda Toolkit version.");
        return PAPI_EMISC;
    }

    papi_errno = init_main_htable();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = assign_chipnames_for_a_device_index();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Collect the available metrics on the machine
    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = cuInitPtr(0);
    if (papi_errno != CUDA_SUCCESS) {
        cuptic_err_set_last("Failed to initialize CUDA driver API.");
        return PAPI_EMISC;
    }

    return PAPI_OK;
}

/** @class verify_user_added_events
  * @brief For user added events, verify they exist and do not require
  *        multiple passes. If both are true, store metadata.
  * @param *events_id
  *   Cuda native event id's.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
*/
int verify_user_added_events(uint32_t *events_id, int num_events, cuptip_control_t state)
{
    int i, papi_errno;
    for (i = 0; i < numDevicesOnMachine; i++) {
        papi_errno = cuptiu_event_table_create_init_capacity(
                         num_events,
                         sizeof(cuptiu_event_t), &(state->gpu_ctl[i].added_events)
                     ); 
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }  

     for (i = 0; i < num_events; i++) {
        event_info_t info;
        papi_errno = evt_id_to_info(events_id[i], &info);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
 
        // Verify the user added event exists
        void *p;
        if (htable_find(cuptiu_table_p->htable, cuptiu_table_p->events[info.nameid].name, (void **) &p) != HTABLE_SUCCESS) {
            return PAPI_ENOEVNT;
        }

        char stat[PAPI_HUGE_STR_LEN]="";
        int strLen;
        if (info.stat < NUM_STATS_QUALS){
            strLen = snprintf(stat, sizeof(stat), "%s", stats[info.stat]);
            if (strLen < 0 || strLen >= sizeof(stat)) {
                SUBDBG("Failed to fully write statistic qualifier.\n");
                return PAPI_ENOMEM;
            }
        }
        const char *stat_position = strstr(cuptiu_table_p->events[info.nameid].basenameWithStatReplaced, "stat");
        if (stat_position == NULL) { 
            ERRDBG("Event does not have a 'stat' placeholder.\n"); 
            return PAPI_EBUG; 
        }
        
        // Reconstructing event name. Append the basename, stat, and sub-metric.
        size_t basename_len = stat_position - cuptiu_table_p->events[info.nameid].basenameWithStatReplaced; 
        char reconstructedEventName[PAPI_HUGE_STR_LEN]="";
        strLen = snprintf(reconstructedEventName, PAPI_HUGE_STR_LEN, "%.*s%s%s",
                   (int)basename_len,
                   cuptiu_table_p->events[info.nameid].basenameWithStatReplaced,
                   stat,
                   stat_position + 4);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to add the CUPTI metric name as reconstruction of the basename and stat exceeded the buffer size.\n");
            return PAPI_EBUF;
        }

        // Verify the user added event does not require multiple passes
        int numOfPasses;
        papi_errno = get_number_of_passes_for_eventsets(cuptiu_table_p->avail_gpu_info[info.device].chipName, reconstructedEventName, &numOfPasses);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }    
        if (numOfPasses > 1) { 
            return PAPI_EMULPASS;
        }

        // For a specific device table, get the current event index
        int idx = state->gpu_ctl[info.device].added_events->count;
        // Store metadata
        strLen = snprintf(state->gpu_ctl[info.device].added_events->cuda_evts[idx],
                         PAPI_HUGE_STR_LEN, "%s", reconstructedEventName);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write the reconstructed CUPTI metric name to array of added events.\n");
            return PAPI_EBUF;
        }
        state->gpu_ctl[info.device].added_events->cuda_devs[idx] = info.device;
        state->gpu_ctl[info.device].added_events->evt_pos[idx] = i; 
        state->gpu_ctl[info.device].added_events->count++; /* total number of events added for a specific device  */
     }

     return PAPI_OK;
}

/** @class cuptip_ctx_create
  * @brief Create a profiling context for the requested Cuda events.
  * @param thr_info
  * @param *pstate
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
  * @param *events_id
  *   Cuda native event id's.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
*/
int cuptip_ctx_create(cuptic_info_t thr_info, cuptip_control_t *pstate, uint32_t *events_id, int num_events)
{
    COMPDBG("Entering.\n");

    cuptip_control_t state = (cuptip_control_t) calloc (1, sizeof(struct cuptip_control_s));
    if (state == NULL) {
        SUBDBG("Failed to allocate memory for state.\n");
        return PAPI_ENOMEM;
    }

    state->gpu_ctl = (cuptip_gpu_state_t *) calloc(numDevicesOnMachine, sizeof(cuptip_gpu_state_t));
    if (state->gpu_ctl == NULL) {
        SUBDBG("Failed to allocate memory for state->gpu_ctl.\n"); 
        return PAPI_ENOMEM;
    }

    long long *counters = (long long *) malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        SUBDBG("Failed to allocate memory for counters.\n");
        return PAPI_ENOMEM;
    }

    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        state->gpu_ctl[dev_id].dev_id = dev_id;
    }

    event_info_t info;
    int papi_errno = evt_id_to_info(events_id[num_events - 1], &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    } 

    // Store a user created cuda context or create one
    papi_errno = cuptic_ctxarr_update_current(thr_info, info.device);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Verify user added events are available on the machine
    papi_errno = verify_user_added_events(events_id, num_events, state);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    state->info = thr_info;
    state->counters = counters;
    *pstate = state;

    return PAPI_OK;
}

/** @class cuptip_ctx_start
  * @brief Code to start counting Cuda hardware events in an event set.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
*/
int cuptip_ctx_start(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    cuptip_gpu_state_t *gpu_ctl;
    CUcontext userCtx, ctx;

    // Return the Cuda context bound to the calling CPU thread
    cudaCheckErrors( cuCtxGetCurrentPtr(&userCtx), return PAPI_EMISC );

    // Enumerate through the devices a user has added an event for
    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        // Skip devices that will require the Events API to be profiled
        int cupti_api = determine_dev_cc_major(dev_id);
        if (cupti_api != API_PERFWORKS) {
            if (cupti_api == API_LEGACY) {
                continue;
            }
            else {
                return PAPI_EMISC;
            }

        }
        gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }

        LOGDBG("Device num %d: event_count %d, rmr count %d\n", dev_id, gpu_ctl->added_events->count, gpu_ctl->numberOfRawMetricRequests);
        papi_errno = cuptic_device_acquire(state->gpu_ctl[dev_id].added_events);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Profiling same gpu from multiple event sets not allowed.\n");
            return papi_errno;
        }
        // Get the cuda context
        papi_errno = cuptic_ctxarr_get_ctx(state->info, dev_id, &ctx);
        // Bind the specified CUDA context to the calling CPU thread
        cudaCheckErrors( cuCtxSetCurrentPtr(ctx), return PAPI_EMISC );

        // Query/filter cuda native events available on host
        papi_errno = get_counter_availability(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Error getting counter availability image.\n");
            return papi_errno;
        }

        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        calculateScratchBufferSizeParam.pChipName = cuptiu_table_p->avail_gpu_info[dev_id].chipName;
        calculateScratchBufferSizeParam.pCounterAvailabilityImage = NULL;
        calculateScratchBufferSizeParam.pPriv = NULL;
        nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr(&calculateScratchBufferSizeParam), return PAPI_EMISC );

        uint8_t myScratchBuffer[calculateScratchBufferSizeParam.scratchBufferSize];
        NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
        metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize;
        metricEvaluatorInitializeParams.pScratchBuffer = myScratchBuffer;
        metricEvaluatorInitializeParams.pChipName = cuptiu_table_p->avail_gpu_info[dev_id].chipName;
        metricEvaluatorInitializeParams.pCounterAvailabilityImage = NULL;
        metricEvaluatorInitializeParams.pCounterDataImage = NULL;
        metricEvaluatorInitializeParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_InitializePtr(&metricEvaluatorInitializeParams), return PAPI_EMISC );
        NVPW_MetricsEvaluator *pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

        NVPA_RawMetricRequest *rawMetricRequests = NULL;
        int i, numOfRawMetricRequests = 0;
        for (i = 0; i < gpu_ctl->added_events->count; i++) {
                NVPW_MetricEvalRequest metricEvalRequest;
                papi_errno = get_metric_eval_request(pMetricsEvaluator, gpu_ctl->added_events->cuda_evts[i], &metricEvalRequest);
                if (papi_errno != PAPI_OK) {
                    return papi_errno;
                }

                papi_errno = create_raw_metric_requests(pMetricsEvaluator, &metricEvalRequest, &rawMetricRequests, &numOfRawMetricRequests);
                if (papi_errno != PAPI_OK) {
                    return papi_errno;
                }
        }

        gpu_ctl->rawMetricRequests = rawMetricRequests;
        gpu_ctl->numberOfRawMetricRequests = numOfRawMetricRequests;

        papi_errno = get_config_image(cuptiu_table_p->avail_gpu_info[dev_id].chipName, gpu_ctl->counterAvailabilityImage.data, gpu_ctl->rawMetricRequests, gpu_ctl->numberOfRawMetricRequests, &gpu_ctl->configImage);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        } 

        papi_errno = get_counter_data_prefix_image(cuptiu_table_p->avail_gpu_info[dev_id].chipName, gpu_ctl->rawMetricRequests, gpu_ctl->numberOfRawMetricRequests, &gpu_ctl->counterDataPrefixImage);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = get_counter_data_image(gpu_ctl->counterDataPrefixImage, &gpu_ctl->counterDataScratchBuffer, &gpu_ctl->counterDataImage);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = start_profiling_session(gpu_ctl->counterDataImage, gpu_ctl->counterDataScratchBuffer, gpu_ctl->configImage);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = begin_pass();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = enable_profiling();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        char rangeName[PAPI_MIN_STR_LEN];
        int strLen = snprintf(rangeName, PAPI_MIN_STR_LEN, "PAPI_Range_%d", gpu_ctl->dev_id);
        if (strLen < 0 || strLen >= PAPI_MIN_STR_LEN) {
            ERRDBG("Failed to fully write range name.\n");
            return PAPI_EBUF;
        } 

        papi_errno = push_range(rangeName);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = destroy_metrics_evaluator(pMetricsEvaluator);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }    
    }
    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), return PAPI_EMISC );

    return PAPI_OK;
}


/** @class cuptip_ctx_read
  * @brief Query an array of numeric values corresponding
  *        to each user added event.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t.
  * @param **counters
  *   An array which holds numeric values for the corresponding
  *   user added event. 
*/
int cuptip_ctx_read(cuptip_control_t state, long long **counters)
{
    COMPDBG("Entering.\n");
    long long *counter_vals = state->counters;

    CUcontext userCtx = NULL, ctx = NULL;
    cudaArtCheckErrors( cuCtxGetCurrentPtr(&userCtx), return PAPI_EMISC );

    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        // Skip devices that will require the Events API to be profiled
        int cupti_api = determine_dev_cc_major(dev_id);
        if (cupti_api != API_PERFWORKS) {
            if (cupti_api == API_LEGACY) {
                continue;
            }
            else {
                return PAPI_EMISC;
            }

        }
        cuptip_gpu_state_t *gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }

        cudaArtCheckErrors( cuptic_ctxarr_get_ctx(state->info, dev_id, &ctx), return PAPI_EMISC );

        cudaArtCheckErrors( cuCtxSetCurrentPtr(ctx), return PAPI_EMISC );
       
        int papi_errno = pop_range();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = end_pass();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = flush_data();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        calculateScratchBufferSizeParam.pChipName = cuptiu_table_p->avail_gpu_info[dev_id].chipName;
        calculateScratchBufferSizeParam.pCounterAvailabilityImage = NULL;
        calculateScratchBufferSizeParam.pPriv = NULL;
        nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr(&calculateScratchBufferSizeParam), return PAPI_EMISC );

        uint8_t myScratchBuffer[calculateScratchBufferSizeParam.scratchBufferSize];
        NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
        metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize;
        metricEvaluatorInitializeParams.pScratchBuffer = myScratchBuffer;
        metricEvaluatorInitializeParams.pChipName = cuptiu_table_p->avail_gpu_info[dev_id].chipName;
        metricEvaluatorInitializeParams.pCounterAvailabilityImage = NULL;
        metricEvaluatorInitializeParams.pCounterDataImage = NULL;
        metricEvaluatorInitializeParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_InitializePtr(&metricEvaluatorInitializeParams), return PAPI_EMISC );
        NVPW_MetricsEvaluator *pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

        long long *metricValues = (long long *) calloc(gpu_ctl->added_events->count, sizeof(long long));
        if (metricValues == NULL) {
            SUBDBG("Failed to allocate memory for metricValues.\n");
            return PAPI_ENOMEM;
        }
        papi_errno = get_evaluated_metric_values(pMetricsEvaluator, gpu_ctl, metricValues);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        int i;
        for (i = 0; i < gpu_ctl->added_events->count; i++) {
            int evt_pos = gpu_ctl->added_events->evt_pos[i];
            if (state->read_count == 0) {
                counter_vals[evt_pos] = metricValues[i];
            }
            else {
                int method = get_event_collection_method(gpu_ctl->added_events->cuda_evts[i]);
                switch (method) {
                    case CUDA_SUM:
                        counter_vals[evt_pos] += metricValues[i];
                        break;
                    case CUDA_MIN:
                        counter_vals[evt_pos] = counter_vals[evt_pos] < metricValues[i] ? counter_vals[evt_pos] : metricValues[i];
                        break;
                    case CUDA_MAX:
                        counter_vals[evt_pos] = counter_vals[evt_pos] > metricValues[i] ? counter_vals[evt_pos] : metricValues[i];
                        break;
                    case CUDA_AVG:
                          // (size * average + value) / (size + 1) 
                          //  size - current number of values in the average
                          //  average - current average
                          //  value - number to add to the average
                         counter_vals[evt_pos] = (state->read_count * counter_vals[i] + metricValues[i]) / (state->read_count + 1);
                         break;
                    default:
                        counter_vals[evt_pos] = metricValues[i];
                        break;
                }
            }
        }
        free(metricValues);
        *counters = counter_vals;

        papi_errno = begin_pass();
        if (papi_errno != PAPI_OK) {
            return papi_errno;

        }

        char rangeName[PAPI_MIN_STR_LEN];
        int strLen = snprintf(rangeName, PAPI_MIN_STR_LEN, "PAPI_Range_%d", gpu_ctl->dev_id);
        if (strLen < 0 || strLen >= PAPI_MIN_STR_LEN) {
            ERRDBG("Failed to fully write range name.\n");
            return PAPI_EBUF;
        }

        papi_errno = push_range(rangeName);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        papi_errno = destroy_metrics_evaluator(pMetricsEvaluator);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

    }
    state->read_count++;

    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), return PAPI_EMISC);

    return PAPI_OK;
}

/** @class cuptip_ctx_reset
  * @brief Code to reset Cuda hardware counter values.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and
  *   cuptip_gpu_state_t.
*/
int cuptip_ctx_reset(cuptip_control_t state)
{
    COMPDBG("Entering.\n");

    int i;
    for (i = 0; i < state->read_count; i++) {
        state->counters[i] = 0;
    }

    state->read_count = 0;

    return PAPI_OK;
}

/** @class cuptip_ctx_stop
  * @brief Code to stop counting PAPI eventset containing Cuda hardware events.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and
  *   cuptip_gpu_state_t.
*/
int cuptip_ctx_stop(cuptip_control_t state)
{
    COMPDBG("Entering.\n");

    CUcontext userCtx = NULL;
    cudaCheckErrors( cuCtxGetCurrentPtr(&userCtx), return PAPI_EMISC );

    int dev_id;
    for (dev_id=0; dev_id < numDevicesOnMachine; dev_id++) {
        // Skip devices that will require the Events API to be profiled
        int cupti_api = determine_dev_cc_major(dev_id);
        if (cupti_api != API_PERFWORKS) {
            if (cupti_api == API_LEGACY) {
                continue;
            }
            else {
                return PAPI_EMISC;
            }

        }        
        cuptip_gpu_state_t *gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }

        CUcontext ctx = NULL;
        int papi_errno = cuptic_ctxarr_get_ctx(state->info, dev_id, &ctx);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        cudaCheckErrors( cuCtxSetCurrentPtr(ctx), return PAPI_EMISC );

        papi_errno = end_profiling_session();
        if (papi_errno != PAPI_OK) {
            SUBDBG("Failed to end profiling session.\n");
            return papi_errno;
        }

        papi_errno = cuptic_device_release(state->gpu_ctl[dev_id].added_events);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        COMPDBG("Stopped and ended profiling session for device %d\n", gpu_ctl->dev_id);
    }

    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class cuptip_ctx_destroy
  * @brief Free allocated memory in start - stop workflow and
  *        reset config images.
  * @param *pstate
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t.
*/
int cuptip_ctx_destroy(cuptip_control_t *pstate)
{
    COMPDBG("Entering.\n");
    cuptip_control_t state = *pstate;
    int i;
    for (i = 0; i < numDevicesOnMachine; i++) {
        free_and_reset_configuration_images( &(state->gpu_ctl[i]) );
        cuptiu_event_table_destroy( &(state->gpu_ctl[i].added_events) );

        // Free the created rawMetricRequests from cuptip_ctx_start
        int j;
        for (j = 0; j < state->gpu_ctl[i].numberOfRawMetricRequests; j++) {
            free((void *) state->gpu_ctl[i].rawMetricRequests[j].pMetricName);
        }
        free(state->gpu_ctl[i].rawMetricRequests);
    }

    // Free the allocated memory from cuptip_ctx_create
    free(state->counters);
    free(state->gpu_ctl);
    free(state);
    *pstate = NULL;

    return PAPI_OK;
}


/** @class get_event_collection_method 
  * @brief Determine the collection method of the event. Can be avg, max, min, or sum..
  * @param *evt_name
  *   Cuda native event name. E.g. dram__bytes.avg 
*/
int get_event_collection_method(const char *evt_name)
{
    if (strstr(evt_name, ".avg") != NULL) {
        return CUDA_AVG;
    }
    else if (strstr(evt_name, ".max") != NULL) {
        return CUDA_MAX;
    }
    else if (strstr(evt_name, ".min") != NULL) {
        return CUDA_MIN;
    }
    else if (strstr(evt_name, ".sum") != NULL) {
        return CUDA_SUM;
    }
    else {
        return CUDA_DEFAULT;
    } 
}

/** @class cuptip_shutdown
  * @brief Free memory and unload function pointers. 
*/
int cuptip_shutdown(void)
{
    COMPDBG("Entering.\n");

    shutdown_event_stats_table();
    shutdown_event_table();

    int papi_errno = deinitialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = unload_nvpw_sym();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = unload_cupti_perf_sym();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}

/** @class evt_id_create
  * @brief Create event ID. Function is needed for cuptip_event_enum.
  *
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
  * @param *event_id
  *   Created event id.
*/
int evt_id_create(event_info_t *info, uint32_t *event_id)
{
    *event_id  = (uint32_t)(info->stat     << STAT_SHIFT);
    *event_id |= (uint32_t)(info->device   << DEVICE_SHIFT);
    *event_id |= (uint32_t)(info->flags    << QLMASK_SHIFT);
    *event_id |= (uint32_t)(info->nameid   << NAMEID_SHIFT);
    return PAPI_OK;
}

/** @class evt_id_to_info
  * @brief Convert event id to info. Function is needed for cuptip_event_enum.
  *
  * @param event_id
  *   An event id.
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
*/
int evt_id_to_info(uint32_t event_id, event_info_t *info)
{
    info->stat     = (uint32_t)((event_id & STAT_MASK) >> STAT_SHIFT);
    info->device   = (uint32_t)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->flags    = (uint32_t)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid   = (uint32_t)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->stat >= (1 << STAT_WIDTH)) {
        return PAPI_ENOEVNT;
    }

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

/** @class init_event_table
  * @brief For a device get and store the metric names.
*/
int init_event_table(void) 
{
    int dev_id, deviceRecord = 0; 
    // Loop through all available devices on the current system
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        // Skip devices that will require the Events API to be profiled
        int cupti_api = determine_dev_cc_major(dev_id);
        if (cupti_api != API_PERFWORKS) {
            if (cupti_api == API_LEGACY) {
                continue;
            }
            else {
                return PAPI_EMISC;
            }

        }
        
        int papi_errno;
        int found = find_same_chipname(dev_id);
        // Unique device found, collect the constructed metric names
        if (found == -1) {
            // Increment device record
            if (dev_id > 0)
                deviceRecord++;

            papi_errno = enumerate_metrics_for_unique_devices( cuptiu_table_p->avail_gpu_info[deviceRecord].chipName,
                                                               &cuptiu_table_p->avail_gpu_info[deviceRecord].totalMetricCount,
                                                               &cuptiu_table_p->avail_gpu_info[deviceRecord].metricNames );
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }
        }
        // Device metadata already collected, set device record
        else {
            deviceRecord = found;
        }

        int i;
        for (i = 0; i < cuptiu_table_p->avail_gpu_info[deviceRecord].totalMetricCount; i++) {
            papi_errno = get_ntv_events(cuptiu_table_p, cuptiu_table_p->avail_gpu_info[deviceRecord].metricNames[i], dev_id);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }
        }

    }

    // Free memory allocated in enumerate_metrics_for_unique_devices and reset totalMetricCount to 0
    int recordIdx;
    for (recordIdx = 0; recordIdx < (deviceRecord + 1); recordIdx++) {
        int metricIdx;
        for (metricIdx = 0; metricIdx < cuptiu_table_p->avail_gpu_info[recordIdx].totalMetricCount; metricIdx++) {
            free(cuptiu_table_p->avail_gpu_info[recordIdx].metricNames[metricIdx]);
        }
        free(cuptiu_table_p->avail_gpu_info[recordIdx].metricNames);
        cuptiu_table_p->avail_gpu_info[recordIdx].totalMetricCount = 0;
    }

    return PAPI_OK;
}

/** @class is_stat
  * @brief Helper function to determine if a token represents a statistical operation.
  *
  * @param token
  *   A string from the event name. Ex. "dram__bytes" "avg"
*/
int is_stat(const char *token) {
    int i;
    for (i = 0; i < NUM_STATS_QUALS; i++) {
        if (strcmp(token, stats[i]) == 0)
            return 1;
    }
    return 0;
}

/** @restructure_event_name
  * @brief Helper function to restructure the event name
  *
  * @param input
  *   Event name string
  * @param output
  *   Event name string (stat string replaced w/ "stat")
  * @param base
  *   Event name string base(w/o stat)
  * @param stat
  *   Event stat string
*/
int restructure_event_name(const char *input, char *output, char *base, char *stat) {
    char input_copy[PAPI_HUGE_STR_LEN];
    int strLen = snprintf(input_copy, PAPI_HUGE_STR_LEN, "%s", input);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        ERRDBG("String larger than PAPI_HUGE_STR_LEN");
        return PAPI_EBUF;
    }


    input_copy[sizeof(input_copy) - 1] = '\0';

    char *parts[10] = {0};
    char *token;
    char delimiter[] = ".";
    int segment_count = 0;
    int stat_index = -1;
    
    // Initialize output strings
    output[0] = '\0';
    base[0] = '\0';
    stat[0] = '\0';

    // Split the string by periods
    token = strtok(input_copy, delimiter);
    while (token != NULL) {
        parts[segment_count] = token;
        if (is_stat(token) == 1) {
            stat_index = segment_count;
        }
        segment_count++;
        token = strtok(NULL, delimiter);
    }

    // Copy the stat
    strLen = snprintf(stat, PAPI_HUGE_STR_LEN, "%s", parts[stat_index]);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        ERRDBG("String larger than PAPI_HUGE_STR_LEN");
        return PAPI_EBUF;
    }


    // Build base name (everything except the stat)
    int i;
    for (i = 0; i < segment_count; i++) {
        if (i != stat_index) {
            if (base[0] != '\0') {
              strcat(base, ".");
              strcat(output, ".");
            }
            strcat(base, parts[i]);
            strcat(output, parts[i]);
        } else {
            if (output[0] != '\0') strcat(output, ".");
            strcat(output, "stat");
        }
    }    
    return PAPI_OK;
}

/** @class get_ntv_events
  * @brief Store Cuda native events and their corresponding device(s).
  *
  * @param *evt_table
  *   Structure containing member variables such as name, evt_code, evt_pos,
      and htable.
  * @param *evt_name
  *   Cuda native event name.
*/
static int get_ntv_events(cuptiu_event_table_t *evt_table, const char *evt_name, int dev_id) 
{
    int papi_errno, strLen;
    char name_restruct[PAPI_HUGE_STR_LEN]="", name_no_stat[PAPI_HUGE_STR_LEN]="", stat[PAPI_HUGE_STR_LEN]="";
    int *count = &evt_table->count;
    int *event_stats_count = &evt_table->event_stats_count;
    cuptiu_event_t *events = evt_table->events;
    StringVector *event_stats = evt_table->event_stats;   
    
    // Check to see if evt_name argument has been provided
    if (evt_name == NULL) {
        return PAPI_EINVAL;
    }

    // Check to see if capacity has been correctly allocated
    if (*count >= evt_table->capacity) {
        return PAPI_EBUG;
    }

    papi_errno = restructure_event_name(evt_name, name_restruct, name_no_stat, stat);
    if (papi_errno != PAPI_OK){
            return papi_errno;
    }

    cuptiu_event_t *event;
    StringVector *stat_vec;
    
    if ( htable_find(evt_table->htable, name_no_stat, (void **) &event) != HTABLE_SUCCESS ) {
        event = &events[*count];
        // Increment event count
        (*count)++;

        strLen = snprintf(event->name, PAPI_HUGE_STR_LEN, "%s", name_no_stat);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            ERRDBG("Failed to fully write CUPTI metric name with no 'stat'.\n");
            return PAPI_EBUF;
        }

        strLen = snprintf(event->basenameWithStatReplaced, PAPI_HUGE_STR_LEN, "%s", name_restruct);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            ERRDBG("Failed to fully write CUPTI metric name with the stat value replaced with 'stat'.\n");
            return PAPI_EBUF;
        }

        stat_vec = &event_stats[*event_stats_count];
        (*event_stats_count)++;
         
        event->stat = stat_vec;
        init_vector(event->stat);
        
        
        papi_errno = push_back(event->stat, stat);
        if (papi_errno != PAPI_OK){
            return papi_errno;
        }

        if ( htable_insert(evt_table->htable, name_no_stat, event) != HTABLE_SUCCESS ) {
            return PAPI_ESYS;
        }
    }
     else {
       papi_errno = push_back(event->stat, stat);
       if (papi_errno != PAPI_OK){
            return papi_errno;
       }
     }

    cuptiu_dev_set(&event->device_map, dev_id);

    return PAPI_OK;
}

/** @class shutdown_event_table
  * @brief Shutdown cuptiu_event_table_t structure that holds the cuda native 
  *        event name and the corresponding description.
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

/** @class shutdown_event_stats_table
  * @brief Shutdown StringVector structure that holds the statistic qualifiers  
  *        for event names.
*/
static void shutdown_event_stats_table(void)
{
    int i;
    for (i = 0; i < cuptiu_table_p->event_stats_count; i++) {
        free_vector(&cuptiu_table_p->event_stats[i]);
    }
    
    cuptiu_table_p->event_stats_count = 0;

    free(cuptiu_table_p->event_stats);
}

/** @class cuptip_evt_enum
  * @brief Enumerate Cuda native events.
  * 
  * @param *event_code
  *   Cuda native event code. 
  * @param modifier
  *   Modifies the search logic. Three modifiers are used PAPI_ENUM_FIRST,
  *   PAPI_ENUM_EVENTS, and PAPI_NTV_ENUM_UMASKS.
*/
int cuptip_evt_enum(uint32_t *event_code, int modifier)
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
            info.stat = 0;
            info.device = 0;
            info.flags = 0;
            info.nameid = 0;
            papi_errno = evt_id_create(&info, event_code);
            break;
        case PAPI_ENUM_EVENTS:
            papi_errno = evt_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }
            if (cuptiu_table_p->count > info.nameid + 1) {
                info.stat = 0;
                info.device = 0;
                info.flags = 0;
                info.nameid++;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_ENOEVNT;
            break;
        case PAPI_NTV_ENUM_UMASKS:
            papi_errno = evt_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }
            if (info.flags == 0){
                info.stat = 0;
                info.device = 0;
                info.flags = STAT_FLAG;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            
            if (info.flags == STAT_FLAG){
                info.stat = 0;
                info.device = 0;
                info.flags = DEVICE_FLAG;
                papi_errno = evt_id_create(&info, event_code);
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

/** @class cuptip_evt_code_to_descr
  * @brief Take a Cuda native event code and retrieve a corresponding description.
  *
  * @param event_code
  *   Cuda native event code. 
  * @param *descr
  *   Corresponding description for provided Cuda native event code.
  * @param len
  *   Maximum alloted characters for Cuda native event description. 
*/
int cuptip_evt_code_to_descr(uint32_t event_code, char *descr, int len) 
{
    event_info_t info;
    int papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }    

    int str_len = snprintf(descr, (size_t) len, "%s", cuptiu_table_p->events[event_code].desc);
    if (str_len < 0 || str_len >= len) {
        ERRDBG("String formatting exceeded max string length.\n");
        return PAPI_EBUF;  
    }    

    return papi_errno;
}

/** @class cuptip_evt_name_to_code
  * @brief Take a Cuda native event name and collect the corresponding event code.
  *
  * @param *name
  *   Cuda native event name.
  * @param *event_code
  *   Corresponding Cuda native event code for provided Cuda native event name.
*/
int cuptip_evt_name_to_code(const char *name, uint32_t *event_code)
{
    int htable_errno, device, stat, flags, nameid, papi_errno = PAPI_OK;
    cuptiu_event_t *event;
    char base[PAPI_HUGE_STR_LEN] = { 0 };
    SUBDBG("ENTER: name: %s, event_code: %p\n", name, event_code);

    papi_errno = evt_name_to_basename(name, base, PAPI_HUGE_STR_LEN);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    char *userAddedQualifiers = strstr(name, ":");
    // Detected user added qualifiers, verify their validity
    // If no qualifiers are added, we handle default cases in
    // evt_name_to_device and evt_name_to_stat respectively.
    if (userAddedQualifiers != NULL) {
        papi_errno = cuda_verify_qualifiers_are_valid(userAddedQualifiers);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }

        papi_errno = cuda_verify_qualifiers_are_not_repeated(userAddedQualifiers);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }

    // Handle device qualifier case
    papi_errno = evt_name_to_device(name, &device, base);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    
    // Handle stat qualifier case
    papi_errno = evt_name_to_stat(name, &stat, base);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }
 
    flags = (event->stat->size >= 0) ? (STAT_FLAG | DEVICE_FLAG) : DEVICE_FLAG;
    if (flags == 0){
        papi_errno = PAPI_EINVAL;
        goto fn_exit;
    }

    nameid = (int) (event - cuptiu_table_p->events);

    event_info_t info = { stat, device, flags, nameid };

    papi_errno = evt_id_create(&info, event_code);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = evt_id_to_info(*event_code, &info);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
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

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
}

/** @class cuptip_evt_code_to_name
  * @brief Returns Cuda native event name for a Cuda native event code. See 
  *        evt_code_to_name( ... ) for more details.
  * @param *event_code
  *   Cuda native event code. 
  * @param *name
  *   Cuda native event name.
  * @param len
  *   Maximum alloted characters for base Cuda native event name. 
*/
int cuptip_evt_code_to_name(uint32_t event_code, char *name, int len)
{
    return evt_code_to_name(event_code, name, len);
}

/** @class evt_code_to_name
  * @brief Helper function for cuptip_evt_code_to_name. Takes a Cuda native event
  *        code and collects the corresponding Cuda native event name. 
  * @param *event_code
  *   Cuda native event code. 
  * @param *name
  *   Cuda native event name.
  * @param len
  *   Maximum alloted characters for base Cuda native event name. 
*/
static int evt_code_to_name(uint32_t event_code, char *name, int len)
{
    event_info_t info;
    int papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int str_len;
    char stat[PAPI_MIN_STR_LEN] = ""; 
    if (info.stat < NUM_STATS_QUALS){
        str_len = snprintf(stat, PAPI_MIN_STR_LEN, "%s", stats[info.stat]);
        if (str_len < 0 || str_len >= PAPI_MIN_STR_LEN) {
            ERRDBG("Failed to fully write statistic qualifier name.\n");
            return PAPI_EBUF;
        }
    }

    switch (info.flags) {
        case (DEVICE_FLAG):
            str_len = snprintf(name, len, "%s:device=%i", cuptiu_table_p->events[info.nameid].name, info.device);
            if (str_len < 0 || str_len >= len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_EBUF;
            }
            break;
        case (STAT_FLAG):    
            str_len = snprintf(name, len, "%s:stat=%s", cuptiu_table_p->events[info.nameid].name, stat);
            if (str_len < 0 || str_len >= len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_EBUF;
            }
            break;
        case (DEVICE_FLAG | STAT_FLAG):
            str_len = snprintf(name, len, "%s:stat=%s:device=%i", cuptiu_table_p->events[info.nameid].name, stat, info.device);
            if (str_len < 0 || str_len >= len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_EBUF;
            }
            break;
        default:
            str_len = snprintf(name, len, "%s", cuptiu_table_p->events[info.nameid].name);
            if (str_len < 0 || str_len >= len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_EBUF;
            }
            break;
    }

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
int cuptip_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info)
{
    event_info_t inf;
    int papi_errno = evt_id_to_info(event_code, &inf);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    const char *stat_position = strstr(cuptiu_table_p->events[inf.nameid].basenameWithStatReplaced, "stat");
    if (stat_position == NULL) {
        return PAPI_ENOMEM;
    }
    size_t basename_len = stat_position - cuptiu_table_p->events[inf.nameid].basenameWithStatReplaced;
    char reconstructedEventName[PAPI_HUGE_STR_LEN]="";
    int strLen = snprintf(reconstructedEventName, PAPI_HUGE_STR_LEN, "%.*s%s%s",
               (int)basename_len,
               cuptiu_table_p->events[inf.nameid].basenameWithStatReplaced,
               cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[0],
               stat_position + 4);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        SUBDBG("Failed to reconstruct CUPTI metric name with the basename and stat.\n");
        return PAPI_EBUF;
    }

    int i;
    // For a Cuda event collect the description, units, and number of passes
    if (cuptiu_table_p->events[inf.nameid].desc[0] == '\0') {
        int dev_id = -1;
        for (i = 0; i < numDevicesOnMachine; ++i) {
            if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                dev_id = i;
                break;
            }
        }

        if (dev_id == -1) {
            SUBDBG("Failed to find a matching device in the device map.\n");
            return PAPI_EINVAL;
        }

        papi_errno = get_metric_properties( cuptiu_table_p->avail_gpu_info[dev_id].chipName, 
                                            reconstructedEventName,
                                            cuptiu_table_p->events[inf.nameid].desc );
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    char all_stat[PAPI_HUGE_STR_LEN]="";
    switch (inf.flags) {
        case (0):
        {
            // Store details for the Cuda event
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].name );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write metric name in case 0.\n");
                return PAPI_EBUF;
            }
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write long description in case 0.\n")
                return PAPI_EBUF;
            }
            break;
        }
        case DEVICE_FLAG:
        {
            char devices[PAPI_2MAX_STR_LEN] = { 0 };
            int init_metric_dev_id;
            for (i = 0; i < numDevicesOnMachine; ++i) {
                if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                    // For an event, store the first device found to use with :device=#, 
                    // as on a heterogenous system events may not appear on each device
                    if (devices[0] == '\0') {
                        init_metric_dev_id = i;

                    }
                    int strLen = snprintf(devices + strlen(devices), PAPI_2MAX_STR_LEN - strlen(devices), "%i,", i);
                    if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN - strlen(devices)) {
                        SUBDBG("Failed to write device %d into devices in DEVICE_FLAG case.\n", i);
                        return PAPI_EBUF;
                    }
                    
                }
            }
            *(devices + strlen(devices) - 1) = 0;

            // Store details for the Cuda event
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:device=%i", cuptiu_table_p->events[inf.nameid].name, init_metric_dev_id );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write metric name in case DEVICE_FLAG.\n");
                return PAPI_EBUF;
            }
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                      cuptiu_table_p->events[inf.nameid].desc, devices );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write long description in case DEVICE_FLAG.\n");
                return PAPI_EBUF;
            }
            break;
        }
        case STAT_FLAG:
        {
            all_stat[0]= '\0'; 
            size_t current_len = strlen(all_stat);
            for (size_t i = 0; i < cuptiu_table_p->events[inf.nameid].stat->size; i++) {
                  size_t remaining_space = PAPI_HUGE_STR_LEN - current_len - 1;  // Calculate remaining space
                
                // Ensure there's enough space for the string before concatenating
                if (remaining_space > 0) {
                    strncat(all_stat, cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[i], remaining_space);
                    current_len += strlen(cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[i]);
                } else {
                    ERRDBG("Not enough space for the all_stat string")
                    return papi_errno;
                }

                // Add a comma only if there is space and it is not the last element
                if (i < cuptiu_table_p->events[inf.nameid].stat->size - 1 && remaining_space > 2) {
                    strncat(all_stat, ", ", remaining_space - 2);
                    current_len += 2;  // Account for the added comma and space
                }
            }
        
            /* cuda native event name */
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:stat=%s", cuptiu_table_p->events[inf.nameid].name, cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[0] );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write metric name in case STAT_FLAG.\n");
                return PAPI_EBUF;
            }
            /* cuda native event long description */
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory stat qualifier [%s]",
                      cuptiu_table_p->events[inf.nameid].desc, all_stat );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("Failed to fully write long description in case STAT_FLAG.\n");
                return PAPI_EBUF;
            }
            break;
        }
        case (STAT_FLAG | DEVICE_FLAG):
        {
            int init_metric_dev_id;
            char devices[PAPI_2MAX_STR_LEN] = { 0 };
            for (i = 0; i < numDevicesOnMachine; ++i) {
                if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                    /* for an event, store the first device found to use with :device=#, 
                       as on a heterogenous system events may not appear on each device */
                    if (devices[0] == '\0') {
                        init_metric_dev_id = i;
                    }

                    strLen = snprintf(devices + strlen(devices), PAPI_2MAX_STR_LEN - strlen(devices), "%i,", i);
                    if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN - strlen(devices)) {
                        SUBDBG("Failed to write device %d into devices in STAT_FLAG | DEVICE_FLAG case.\n", i);
                        return PAPI_EBUF;
                    }
                }
            }
            *(devices + strlen(devices) - 1) = 0;
            
            all_stat[0]= '\0'; 
            size_t current_len = strlen(all_stat);
            for (size_t i = 0; i < cuptiu_table_p->events[inf.nameid].stat->size; i++) {
                  size_t remaining_space = PAPI_HUGE_STR_LEN - current_len - 1;  // Calculate remaining space
                
                // Ensure there's enough space for the string before concatenating
                if (remaining_space > 0) {
                    strncat(all_stat, cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[i], remaining_space);
                    current_len += strlen(cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[i]);
                } else {
                    ERRDBG("Not enough space for the all_stat string")
                    return papi_errno;
                }

                // Add a comma only if there is space and it is not the last element
                if (i < cuptiu_table_p->events[inf.nameid].stat->size - 1 && remaining_space > 2) {
                    strncat(all_stat, ", ", remaining_space - 2);
                    current_len += 2;  // Account for the added comma and space
                }
            }
        
            /* cuda native event name */
            strLen = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:stat=%s:device=%i", cuptiu_table_p->events[inf.nameid].name, cuptiu_table_p->events[inf.nameid].stat->arrayMetricStatistics[0], init_metric_dev_id);
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("String larger than PAPI_HUGE_STR_LEN");
                return PAPI_EBUF;
            }
            
            /* cuda native event long description */
            strLen = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory stat qualifier [%s]:Mandatory device qualifier [%s]",
                      cuptiu_table_p->events[inf.nameid].desc, all_stat, devices  );
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                ERRDBG("String larger than PAPI_HUGE_STR_LEN");
                return PAPI_EBUF;
            }
            break;
        }
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
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

/** @class cuda_verify_qualifiers_are_not_repeated
  * @brief Verify that a user has not added multiple device or stats qualifiers
  *        to an event name.
  *
  * @param *eventName
  *   User provided event name we need to verify.
*/
static int cuda_verify_qualifiers_are_not_repeated(const char *qualifiers)
{
    int numDeviceQualifiers = 0, numStatsQualifiers = 0;
    char tmpQualifiers[PAPI_HUGE_STR_LEN];
    int strLen = snprintf(tmpQualifiers, PAPI_HUGE_STR_LEN, "%s", qualifiers);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        SUBDBG("Failed to fully write qualifiers into tmpQualifiers.\n");
        return PAPI_EBUF;
    }
    char *token = strtok(tmpQualifiers, ":");
    while(token != NULL) {
        if (strncmp(token, "device", 6) == 0) {
            numDeviceQualifiers++;
        }
        else if (strncmp(token, "stat", 4) == 0){
            numStatsQualifiers++;
        }

        token = strtok(NULL, ":");
    }

    if (numDeviceQualifiers > 1 || numStatsQualifiers > 1) {
        SUBDBG("Provided Cuda event has multiple device or stats qualifiers appended.\n");
        return PAPI_ENOEVNT;
    }

    return PAPI_OK;
}

/** @class cuda_verify_qualifiers_are_valid
  * @brief Verify that a user has added valid qualifiers i.e. stat or device.
  *
  * @param *qualifiers
  *   String of user added qualifiers.
*/
static int cuda_verify_qualifiers_are_valid(const char *qualifiers)
{
    char tmpQualifiers[PAPI_HUGE_STR_LEN];
    int strLen = snprintf(tmpQualifiers, PAPI_HUGE_STR_LEN, "%s", qualifiers);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        SUBDBG("Failed to fully write qualifiers into tmpQualifiers.\n");
        return PAPI_EBUF;
    }

    char *token = strtok(tmpQualifiers, ":");
    while (token != NULL) {
        if (strncmp(token, "device", 6) != 0 &&
            strncmp(token, "stat", 4) != 0) {
            SUBDBG("The appended qualifier is not supported: %s\n", token);
            return PAPI_ENOEVNT;
        }

        token = strtok(NULL, ":");
    }

    return PAPI_OK;
}

/** @class cuda_verify_qualifiers_do_not_contain_junk
  * @brief Verify that the device and/or stats qualifier provided by the user
  *        does not contain excess characters or does not have the proper
  *        equals sign.
  *
  * @param flag
  *   Device or stats flag define. Allows us to determine the case to enter for
  *   the switch statement.
  * @param *qualifierName
  *   Name of the qualifier we need to verify. E.g. :device or :stat.
  * @param equalitySignPosition
  *   Position of where the equal sign is located in the qualifier string name.
  * @param *qualifierValue
  *   Upon verifying the provided qualifier is valid. Store either a device index
  *   or a statistic index.
*/
static int cuda_verify_qualifiers_do_not_contain_junk(int flag, char *qualifierName, int equalitySignPosition, int *qualifierValue)
{
    int pos = equalitySignPosition;
    // Verify that an equal sign was provided where it was suppose to be
    if (qualifierName[pos] != '=') {
        SUBDBG("Improper qualifier name. No equal sign found.\n");
        return PAPI_ENOEVNT;
    }

    switch(flag)
    {
        case DEVICE_FLAG:
        {
            // Verify that the next character after the equal sign is indeed a digit
            pos++;
            int isDigit = (unsigned) qualifierName[pos] - '0' < 10;
            if (!isDigit) {
                SUBDBG("Improper device qualifier name. Digit does not follow equal sign.\n");
                return PAPI_ENOEVNT;
            }

            // Verify that no junk has been appended after the device qualifier
            char *endPtr;
            *qualifierValue = (int) strtol(qualifierName + strlen(":device="), &endPtr, 10);
            // Check to make sure only qualifiers have been appended
            if (*endPtr != '\0') {
                if (strncmp(endPtr, ":stat", 5) != 0) {
                    return PAPI_ENOEVNT;
                }
            }
            return PAPI_OK;
        }
        case STAT_FLAG:
        {
            qualifierName += 6; // Move past ":stat="
            int i;
            for (i = 0; i < NUM_STATS_QUALS; i++) {
                size_t token_len = strlen(stats[i]);
                if (strncmp(qualifierName, stats[i], token_len) == 0) {
                    // Verify that no junk has been appended after the stats qualifier
                    char *no_excess_chars = qualifierName + token_len;
                    if (strlen(no_excess_chars) == 0 || strncmp(no_excess_chars, ":device", 7) == 0) {
                        *qualifierValue = i;
                        return PAPI_OK;
                    }
                }
            }
            return PAPI_ENOEVNT;
        }
        default:
            SUBDBG("Flag provided is not accounted for in switch statement.\n");
            return PAPI_EINVAL;
    }
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
static int evt_name_to_device(const char *name, int *device, const char *base)
{
    char *p = strstr(name, ":device");
    // User did provide :device=# qualifier
    if (p != NULL) {
        int equalitySignPos = 7;
        int papi_errno = cuda_verify_qualifiers_do_not_contain_junk(DEVICE_FLAG, p, equalitySignPos, device);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }
    // User did not provide :device=# qualifier
    else {
        int i, htable_errno;
        cuptiu_event_t *event;

        htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_EINVAL;
        }
        // Search for the first device the event exists for.
        for (i = 0; i < numDevicesOnMachine; ++i) {
            if (cuptiu_dev_check(event->device_map, i)) {
                *device = i;
                return PAPI_OK;
            }
        }
    }

    return PAPI_OK;
}

/** @class evt_name_to_stat
  * @brief Take a Cuda native event name with a stat qualifer appended to 
  *        it and collect the stat .
  * @param *name
  *   Cuda native event name with a stat qualifier appended.
  * @param *stat
  *   Stat collected.
*/
static int evt_name_to_stat(const char *name, int *stat, const char *base)
{
    char *p = strstr(name, ":stat");
    if (p != NULL) {
        int equalitySignPos = 5;
        int papi_errno = cuda_verify_qualifiers_do_not_contain_junk(STAT_FLAG, p, equalitySignPos, stat);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    } else {
        cuptiu_event_t *event;
        int htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_ENOEVNT;
        }
        int i;
        for (i = 0; i < NUM_STATS_QUALS; i++) {
          size_t token_len = strlen(stats[i]);
          if (strncmp(event->stat->arrayMetricStatistics[0], stats[i], token_len) == 0) {
                *stat = i;
                return PAPI_OK;
          }
        }
    }
}
/** @class assign_chipnames_for_a_device_index
  * @brief For each device found, assign a chipname.
*/

static int assign_chipnames_for_a_device_index(void)
{
    char chipName[PAPI_MIN_STR_LEN];
    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        int retval = get_chip_name(dev_id, chipName);
        if (PAPI_OK != retval ) {
            return PAPI_EMISC;
        }

        int strLen = snprintf(cuptiu_table_p->avail_gpu_info[dev_id].chipName, PAPI_MIN_STR_LEN, "%s", chipName);
        if (strLen < 0 || strLen >= PAPI_MIN_STR_LEN) {
            SUBDBG("Failed to fully write chip name.\n");
            return PAPI_EBUF;
        }    
    }    

    return PAPI_OK;
}

static int determine_dev_cc_major(int dev_id)
{
    int cc;
    int papi_errno = get_gpu_compute_capability(dev_id, &cc);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (cc >= 70) {
        return API_PERFWORKS;
    }
    // TODO: Once the Events API is added back, move this to either cupti_utils or papi_cupti_common
    //       with updated logic.
    else {
        return API_LEGACY;
    }
}

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   Metrics Evaluator
 *  @{
 */

/** @class enumerate_metrics_for_unique_devices
 *  @brief Get the total number of metrics on a device and the subsequent metric names
 *         using the Metrics Evaluator API. 
 *
 *  @param *pChipName
 *    A Cuda device chip name.
 *  @param *totalNumMetrics
 *    Count of the total number of metrics found on a device.
 *  @param ***arrayOfMetricNames
 *    Constructured metric names. With the Metrics Evaluator API, a metric name must be
 *    reconstructured using metricName.rollup.submetric.
*/
static int enumerate_metrics_for_unique_devices(const char *pChipName, int *totalNumMetrics, char ***arrayOfMetricNames)
{
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParam.pChipName = pChipName;
    calculateScratchBufferSizeParam.pCounterAvailabilityImage = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr(&calculateScratchBufferSizeParam), return PAPI_EMISC );

    uint8_t myScratchBuffer[calculateScratchBufferSizeParam.scratchBufferSize];
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize;
    metricEvaluatorInitializeParams.pScratchBuffer = myScratchBuffer;
    metricEvaluatorInitializeParams.pChipName = pChipName;
    metricEvaluatorInitializeParams.pCounterAvailabilityImage = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_InitializePtr(&metricEvaluatorInitializeParams), return PAPI_EMISC );
    NVPW_MetricsEvaluator *pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

    char **metricNames = NULL;
    int i, metricCount = 0, papi_errno;
    for (i = 0; i < NVPW_METRIC_TYPE__COUNT; ++i) {
        NVPW_MetricType metricType = (NVPW_MetricType)i;

        NVPW_MetricsEvaluator_GetMetricNames_Params getMetricNamesParams = {NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE};
        getMetricNamesParams.metricType = metricType;
        getMetricNamesParams.pMetricsEvaluator = pMetricsEvaluator;
        getMetricNamesParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_MetricsEvaluator_GetMetricNamesPtr(&getMetricNamesParams), return PAPI_EMISC );

        size_t metricIdx;
        for (metricIdx = 0; metricIdx < getMetricNamesParams.numMetrics; ++metricIdx) {
            size_t metricNameBeginIndex = getMetricNamesParams.pMetricNameBeginIndices[metricIdx];
            const char *baseMetricName = &getMetricNamesParams.pMetricNames[metricNameBeginIndex];

            // CTC metrics are not supported by the Perfworks Metrics API
            if (strstr(baseMetricName, "ctc__") || strstr(baseMetricName, "CTC")) {
                SUBDBG("CTC metric found. The Perfworks Metrics API does not support profiling these metrics. Moving to the next metric.\n");
                continue;
            }

            char fullMetricName[PAPI_HUGE_STR_LEN];
            int strLen = snprintf(fullMetricName, PAPI_HUGE_STR_LEN, "%s", baseMetricName);
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully append the base metric name.\n");
                return PAPI_EBUF;
            }

            int rollupMetricIdx;
            for (rollupMetricIdx = 0; rollupMetricIdx < NVPW_ROLLUP_OP__COUNT; ++rollupMetricIdx) {
                // Set the starting offset to be used for a metric
                int offsetForMetricName = strlen(baseMetricName);
                // Get the rollup metric if applicable
                // Rollup's are required for Counter and Throughput, but does not apply to Ratio
                char *rollupMetricName = NULL;
                if (metricType != NVPW_METRIC_TYPE_RATIO) {
                    papi_errno = get_rollup_metrics(rollupMetricIdx, &rollupMetricName);
                    if (papi_errno != 0) {
                        return papi_errno;
                    }

                    strLen = snprintf(fullMetricName + offsetForMetricName, PAPI_HUGE_STR_LEN - offsetForMetricName, "%s", rollupMetricName);
                    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN - offsetForMetricName) {
                        SUBDBG("Failed to fully append rollup metric name.\n");
                        return PAPI_EBUF;
                    }

                    // Update the offset as a rollup metric was found
                    offsetForMetricName += strlen(rollupMetricName);
                }

                // Get the list of submetrics 
                // Submetrics are required for Ratio and Throughput, optional for Counter (here we do collect for Counter as well)
                NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params supportedSubMetrics = {NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE};
                supportedSubMetrics.pMetricsEvaluator = pMetricsEvaluator;
                supportedSubMetrics.metricType = metricType;
                supportedSubMetrics.pPriv = NULL;
                nvpwCheckErrors( NVPW_MetricsEvaluator_GetSupportedSubmetricsPtr(&supportedSubMetrics), return PAPI_EMISC );

                size_t subMetricIdx;
                for (subMetricIdx = 0; subMetricIdx < supportedSubMetrics.numSupportedSubmetrics; ++subMetricIdx) {
                    char *subMetricName;
                    papi_errno = get_supported_submetrics(supportedSubMetrics.pSupportedSubmetrics[subMetricIdx], &subMetricName);
                    if (papi_errno != 0) {
                        return papi_errno;
                    }

                    if (supportedSubMetrics.pSupportedSubmetrics[subMetricIdx] != NVPW_SUBMETRIC_NONE) {
                        strLen = snprintf(fullMetricName + offsetForMetricName, PAPI_HUGE_STR_LEN - offsetForMetricName, "%s", subMetricName);
                        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN - offsetForMetricName) {
                            SUBDBG("Failed to fully append submetric names.\n");
                            return PAPI_EBUF;
                        }
                    }

                    metricNames = (char **) realloc(metricNames, (metricCount + 1) * sizeof(char *));
                    if (metricNames == NULL) {
                        SUBDBG("Failed to allocate memory for metricNames.\n");
                        return PAPI_ENOMEM;
                    }
                    metricNames[metricCount] = (char *) malloc(PAPI_HUGE_STR_LEN * sizeof(char));
                    if (metricNames[metricCount] == NULL) {
                        SUBDBG("Failed to allocate memory for the index %d in the array metricNames.\n", metricCount);
                        return PAPI_ENOMEM;
                    }

                    // Store the constructed metric name
                    strLen = snprintf(metricNames[metricCount], PAPI_HUGE_STR_LEN, "%s", fullMetricName);
                    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                        SUBDBG("Failed to fully write constructued metric name: %s\n", fullMetricName);
                        return PAPI_EBUF;
                    }
                    metricCount++;
                }
                // Avoid counting ratio metrics 4X more then should occur 
                if (metricType == NVPW_METRIC_TYPE_RATIO) {
                    break;
                }
            }
        }
    }

    papi_errno = destroy_metrics_evaluator(pMetricsEvaluator);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    *totalNumMetrics = metricCount;
    *arrayOfMetricNames = metricNames;

    return PAPI_OK;
} 

/** @class get_rollup_metrics
  * @brief Get the appropriate string for a provided member of the NVPW_RollupOp
  *        enum. Note that, rollup's are required for Counter and Throughput, but
  *        does not apply to Ratio.
  * @param rollupMetric
  *   A member of the enum NVPW_RollupOp. See nvperf_host.h for a full list.
  * @param **strRollupMetric
  *   String rollup metric to store based on the rollupMetric parameter.
*/
static int get_rollup_metrics(NVPW_RollupOp rollupMetric, char **strRollupMetric)
{
    switch(rollupMetric)
    {
        case NVPW_ROLLUP_OP_AVG:
            *strRollupMetric = ".avg";
            return PAPI_OK;
        case NVPW_ROLLUP_OP_MAX:
            *strRollupMetric = ".max";
            return PAPI_OK;
        case NVPW_ROLLUP_OP_MIN:
            *strRollupMetric = ".min";
            return PAPI_OK;
        case NVPW_ROLLUP_OP_SUM:
            *strRollupMetric = ".sum";
            return PAPI_OK;
        default:
            SUBDBG("Rollup metric was not one of avg, max, min, or sum.\n");
            *strRollupMetric = "";
            return PAPI_OK;
    } 
}

/** @class get_supported_submetrics
  * @brief Get the appropriate string for a provided member of the NVPW_Submetric
  *        enum. Note that, submetrics are required for Ratio and Throughput, optional
  *        for Counter.
  * @param subMetric
  *   A member of the enum NVPW_Submetric. See nvperf_host.h for a full list.
  * @param **strSubMetric
  *   String submetric to store based on the subMetric parameter.
*/
static int get_supported_submetrics(NVPW_Submetric subMetric, char **strSubMetric)
{
    // NOTE: The following submetrics are not supported in CUPTI 11.3 and onwards:
    //       - Burst submetrics: .peak_burst, .pct_of_peak_burst_active, .pct_of_peak_burst_active
    //                           .pct_of_peak_burst_elapsed, .pct_of_peak_burst_region,
    //                           .pct_of_peak_burst_frame.
    //       - Throughput submetrics: .pct_of_peak_burst_active, .pct_of_peak_burst_elapsed
    //                                .pct_of_peak_burst_region, .pct_of_peak_burst_frame.
    switch (subMetric)
    {
        case NVPW_SUBMETRIC_PEAK_SUSTAINED:
            *strSubMetric = ".peak_sustained";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE:
            *strSubMetric = ".peak_sustained_active";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND:
            *strSubMetric = ".peak_sustained_active.per_second";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED:
            *strSubMetric = ".peak_sustained_elapsed";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND:
            *strSubMetric = ".peak_sustained_elapsed.per_second";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME:
            *strSubMetric = ".peak_sustained_frame";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND:
            *strSubMetric = ".peak_sustained_frame.per_second";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION:
            *strSubMetric = ".peak_sustained_region";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND:
            *strSubMetric = ".peak_sustained_region.per_second";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PER_CYCLE_ACTIVE:
            *strSubMetric = ".per_cycle_active";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PER_CYCLE_ELAPSED:
            *strSubMetric = ".per_cycle_elapsed";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PER_CYCLE_IN_FRAME:
            *strSubMetric = ".per_cycle_in_frame";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PER_CYCLE_IN_REGION:
            *strSubMetric = ".per_cycle_in_region";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PER_SECOND:
            *strSubMetric = ".per_second";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE:
            *strSubMetric = ".pct_of_peak_sustained_active";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED:
            *strSubMetric = ".pct_of_peak_sustained_elapsed";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME:
            *strSubMetric = ".pct_of_peak_sustained_frame";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION:
            *strSubMetric = ".pct_of_peak_sustained_region";
            return PAPI_OK;
        case NVPW_SUBMETRIC_MAX_RATE:
            *strSubMetric = ".max_rate";
            return PAPI_OK;
        case NVPW_SUBMETRIC_PCT:
            *strSubMetric = ".pct";
             return PAPI_OK;
        case NVPW_SUBMETRIC_RATIO:
            *strSubMetric = ".ratio";
            return PAPI_OK;
        case NVPW_SUBMETRIC_NONE:
        default:
           *strSubMetric = "";
           return PAPI_OK;
    }
}

/** @class get_metric_properties
 *  @brief For a metric, get the description, units, and number
 *         of passes.
 *
 *  @param *pChipName
 *    The device chipname.
 *  @param *metricName
 *    A metric name from the Perfworks api.
 *  @param *fullMetricDescription
 *    The constructed metric description with units and number of
 *    passes.
*/
static int get_metric_properties(const char *pChipName, const char *metricName, char *fullMetricDescription)
{
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParam.pChipName = pChipName;
    calculateScratchBufferSizeParam.pCounterAvailabilityImage = NULL;
    calculateScratchBufferSizeParam.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr(&calculateScratchBufferSizeParam), return PAPI_EMISC );

    uint8_t myScratchBuffer[calculateScratchBufferSizeParam.scratchBufferSize];
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize;
    metricEvaluatorInitializeParams.pScratchBuffer = myScratchBuffer;
    metricEvaluatorInitializeParams.pChipName = pChipName;
    metricEvaluatorInitializeParams.pCounterAvailabilityImage = NULL;
    metricEvaluatorInitializeParams.pCounterDataImage = NULL;
    metricEvaluatorInitializeParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_InitializePtr(&metricEvaluatorInitializeParams), return PAPI_EMISC );
    NVPW_MetricsEvaluator *pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

    NVPW_MetricEvalRequest metricEvalRequest;
    int papi_errno = get_metric_eval_request(pMetricsEvaluator, metricName, &metricEvalRequest);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    NVPW_MetricType metricType = (NVPW_MetricType) metricEvalRequest.metricType;
    size_t metricIndex = metricEvalRequest.metricIndex;

    // For a metric, get the description
    const char *metricDescription;
    if (metricType == NVPW_METRIC_TYPE_COUNTER) {
        NVPW_MetricsEvaluator_GetCounterProperties_Params counterPropParams = {NVPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE};
        counterPropParams.pMetricsEvaluator = pMetricsEvaluator;
        counterPropParams.counterIndex = metricIndex;
        counterPropParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_MetricsEvaluator_GetCounterPropertiesPtr(&counterPropParams), return PAPI_EMISC );
        metricDescription = counterPropParams.pDescription;
    }
    else if (metricType == NVPW_METRIC_TYPE_RATIO) {
        NVPW_MetricsEvaluator_GetRatioMetricProperties_Params ratioPropParams = {NVPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE};
        ratioPropParams.pMetricsEvaluator = pMetricsEvaluator;
        ratioPropParams.ratioMetricIndex = metricIndex;
        ratioPropParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_MetricsEvaluator_GetRatioMetricPropertiesPtr(&ratioPropParams), return PAPI_EMISC );
        metricDescription = ratioPropParams.pDescription;
    }
    else if (metricType == NVPW_METRIC_TYPE_THROUGHPUT) {
        NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params throughputPropParams = {NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params_STRUCT_SIZE};
        throughputPropParams.pMetricsEvaluator = pMetricsEvaluator;
        throughputPropParams.throughputMetricIndex = metricIndex;
        throughputPropParams.pPriv = NULL;
        nvpwCheckErrors( NVPW_MetricsEvaluator_GetThroughputMetricPropertiesPtr(&throughputPropParams), return PAPI_EMISC );
        metricDescription = throughputPropParams.pDescription;
    }

    // For a metric, get the dimensional units
    NVPW_MetricsEvaluator_GetMetricDimUnits_Params dimUnitsParams = {NVPW_MetricsEvaluator_GetMetricDimUnits_Params_STRUCT_SIZE};
    dimUnitsParams.pMetricsEvaluator = pMetricsEvaluator;
    dimUnitsParams.pMetricEvalRequest = &metricEvalRequest;
    dimUnitsParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    dimUnitsParams.dimUnitFactorStructSize = NVPW_DimUnitFactor_STRUCT_SIZE;
    dimUnitsParams.pDimUnits = NULL;
    dimUnitsParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_MetricsEvaluator_GetMetricDimUnitsPtr(&dimUnitsParams), return PAPI_EMISC );

    int strLen;
    char *metricUnits = "unitless"; // It appears that some metrics have a bug which do not return a value of 1 when they should for unitless.
    if (dimUnitsParams.numDimUnits > 0) {
        NVPW_DimUnitFactor *dimUnitsFactor = (NVPW_DimUnitFactor *) malloc(dimUnitsParams.numDimUnits * sizeof(NVPW_DimUnitFactor));
        if (dimUnitsFactor == NULL) {
            SUBDBG("Failed to allocate memory for dimUnitsFactor.\n");
            return PAPI_ENOMEM;
        }
        dimUnitsParams.pDimUnits = dimUnitsFactor;
        nvpwCheckErrors( NVPW_MetricsEvaluator_GetMetricDimUnitsPtr(&dimUnitsParams), return PAPI_EMISC );

        char tmpMetricUnits[PAPI_HUGE_STR_LEN] = { 0 };
        int i, offsetMetricUnits = 0;
        for (i = 0; i < dimUnitsParams.numDimUnits; i++) {
            NVPW_MetricsEvaluator_DimUnitToString_Params dimUnitToStringParams = {NVPW_MetricsEvaluator_DimUnitToString_Params_STRUCT_SIZE};
            dimUnitToStringParams.pMetricsEvaluator = pMetricsEvaluator;
            dimUnitToStringParams.dimUnit = dimUnitsFactor[i].dimUnit;
            dimUnitToStringParams.pPriv = NULL;
            nvpwCheckErrors( NVPW_MetricsEvaluator_DimUnitToStringPtr(&dimUnitToStringParams), return PAPI_EMISC );

            char *unitsFormat = (i == 0) ? "%s" : "/%s";
            strLen = snprintf(tmpMetricUnits + offsetMetricUnits, PAPI_HUGE_STR_LEN - offsetMetricUnits, unitsFormat, dimUnitToStringParams.pPluralName);
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN - offsetMetricUnits) {
                SUBDBG("Failed to fully write dimensional units for a metric.\n");
                return PAPI_EBUF;
            }
            offsetMetricUnits = strlen(tmpMetricUnits);
        }
        free(dimUnitsFactor);
        metricUnits = tmpMetricUnits;
    }

    int numOfPasses = 0;
    papi_errno = get_number_of_passes_for_info(pChipName, pMetricsEvaluator, &metricEvalRequest, &numOfPasses);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    char *multipassSupport = "";
    if (numOfPasses > 1) {
        multipassSupport = "(multiple passes not supported)";
    }

    strLen = snprintf(fullMetricDescription, PAPI_HUGE_STR_LEN, "%s. Units=(%s). Numpass=%d%s.", metricDescription, metricUnits, numOfPasses, multipassSupport);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        SUBDBG("Failed to fully write metric description.\n");
        return PAPI_EBUF;
    }

    papi_errno = destroy_metrics_evaluator(pMetricsEvaluator);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}

/** @class get_number_of_passes_for_eventsets
 *  @brief For a metric, get the number of passes. Function is specifically
 *         designed to work with the start - stop workflow.
 *
 *  @param *pChipName
 *    The device chipname.
 *  @param *metricEvaluator
 *    A NVPW_MetricsEvaluator struct.
 *  @param *metricEvalRequest
 *    A created metric eval request for the current metric. 
 *  @param *numOfPasses
 *    The total number of passes required by the metric.
*/
static int get_number_of_passes_for_eventsets(const char *pChipName, const char *metricName, int *numOfPasses)
{
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParam.pChipName = pChipName;
    calculateScratchBufferSizeParam.pCounterAvailabilityImage = NULL;
    calculateScratchBufferSizeParam.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSizePtr(&calculateScratchBufferSizeParam), return PAPI_EMISC );

    uint8_t myScratchBuffer[calculateScratchBufferSizeParam.scratchBufferSize];
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize;
    metricEvaluatorInitializeParams.pScratchBuffer = myScratchBuffer;
    metricEvaluatorInitializeParams.pChipName = pChipName;
    metricEvaluatorInitializeParams.pCounterAvailabilityImage = NULL;
    metricEvaluatorInitializeParams.pCounterDataImage = NULL;
    metricEvaluatorInitializeParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_MetricsEvaluator_InitializePtr(&metricEvaluatorInitializeParams), return PAPI_EMISC );
    NVPW_MetricsEvaluator *pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

    NVPW_MetricEvalRequest metricEvalRequest;
    int papi_errno = get_metric_eval_request(pMetricsEvaluator, metricName, &metricEvalRequest);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    } 

    int rawMetricRequestsCount = 0;
    NVPA_RawMetricRequest *rawMetricRequests = NULL;
    papi_errno = create_raw_metric_requests(pMetricsEvaluator, &metricEvalRequest, &rawMetricRequests, &rawMetricRequestsCount);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    } 

    papi_errno = destroy_metrics_evaluator(pMetricsEvaluator);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = {NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
    rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    rawMetricsConfigCreateParams.pChipName = pChipName;
    rawMetricsConfigCreateParams.pCounterAvailabilityImage = NULL;
    rawMetricsConfigCreateParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_RawMetricsConfig_Create_V2Ptr(&rawMetricsConfigCreateParams), return PAPI_EMISC );
    // Destory pRawMetricsConfig at the end; otherwise, a memory leak will occur
    NVPA_RawMetricsConfig *pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    beginPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = rawMetricRequests;
    addMetricsParams.numMetricRequests = rawMetricRequestsCount;
    addMetricsParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    endPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = {NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE};
    rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = pRawMetricsConfig;
    rawMetricsConfigGetNumPassesParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetNumPassesPtr(&rawMetricsConfigGetNumPassesParams), return PAPI_EMISC );

    size_t numNestingLevels = 1;
    size_t numIsolatedPasses = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    size_t numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;
    *numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    rawMetricsConfigDestroyParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams), return PAPI_EMISC );

    int i;
    for (i = 0; i < rawMetricRequestsCount; i++) {
        free((void *) rawMetricRequests[i].pMetricName);
    }
    free(rawMetricRequests);

    return PAPI_OK;

}


/** @class get_number_of_passes_for_info
 *  @brief For a metric, get the number of passes. Function is specifically
 *         designed to work with the evt_code_to_info workflow.
 *
 *  @param *pChipName
 *    The device chipname.
 *  @param *metricEvaluator
 *    A NVPW_MetricsEvaluator struct.
 *  @param *metricEvalRequest
 *    A created metric eval request for the current metric. 
 *  @param *numOfPasses
 *    The total number of passes required by the metric.
*/
static int get_number_of_passes_for_info(const char *pChipName, NVPW_MetricsEvaluator *pMetricsEvaluator, NVPW_MetricEvalRequest *metricEvalRequest, int *numOfPasses)
{
    int rawMetricRequestsCount = 0; 
    NVPA_RawMetricRequest *rawMetricRequests = NULL;
    int papi_errno = create_raw_metric_requests(pMetricsEvaluator, metricEvalRequest, &rawMetricRequests, &rawMetricRequestsCount);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }  

    NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = {NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
    rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    rawMetricsConfigCreateParams.pChipName = pChipName;
    rawMetricsConfigCreateParams.pCounterAvailabilityImage = NULL;
    rawMetricsConfigCreateParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_RawMetricsConfig_Create_V2Ptr(&rawMetricsConfigCreateParams), return PAPI_EMISC );
    // Destory pRawMetricsConfig at the end; otherwise, a memory leak will occur
    NVPA_RawMetricsConfig *pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    beginPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), return PAPI_EMISC );
    
    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = rawMetricRequests;
    addMetricsParams.numMetricRequests = rawMetricRequestsCount;
    addMetricsParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    endPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = {NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE};
    rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = pRawMetricsConfig;
    rawMetricsConfigGetNumPassesParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetNumPassesPtr(&rawMetricsConfigGetNumPassesParams), return PAPI_EMISC );

    size_t numNestingLevels = 1;  
    size_t numIsolatedPasses = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    size_t numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;
    *numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    rawMetricsConfigDestroyParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams), return PAPI_EMISC );

    int i;   
    for (i = 0; i < rawMetricRequestsCount; i++) {
        free((void *) rawMetricRequests[i].pMetricName);
    }
    free(rawMetricRequests);

    return PAPI_OK;
}

/** @class get_metric_eval_request
 *  @brief A simple wrapper for the perfworks api call
 *         NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest.
 *
 *  @param *pMetricsEvaluator
 *    A NVPW_MetricsEvaluator struct.
 *  @param *metricName
 *    The name of the metric you want to convert to a metric eval request.
 *  @param *pMetricEvalRequest
 *    Variable to store the created metric eval request.
*/
static int get_metric_eval_request(NVPW_MetricsEvaluator *pMetricsEvaluator, const char *metricName, NVPW_MetricEvalRequest *pMetricEvalRequest)
{
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = pMetricsEvaluator;
    convertMetricToEvalRequest.pMetricName = metricName;
    convertMetricToEvalRequest.pMetricEvalRequest = pMetricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    convertMetricToEvalRequest.pPriv = NULL;
    nvpwCheckErrors( NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequestPtr(&convertMetricToEvalRequest), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class create_raw_metric_requests
 *  @brief Create raw metric requests for a metric.
 *
 *  @param *pMetricsEvaluator
 *    A NVPW_MetricsEvaluator struct. 
 *  @param *metricEvalRequest
 *    A metric eval request for the metric.
 *  @param **rawMetricRequests
 *    Store the raw metric requests for a metric.
 *  @param *rawMetricRequestsCount
 *    Total number of raw metric requests created.
*/
static int create_raw_metric_requests(NVPW_MetricsEvaluator *pMetricsEvaluator, NVPW_MetricEvalRequest *metricEvalRequest, NVPA_RawMetricRequest **rawMetricRequests, int *rawMetricRequestsCount)
{
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDependenciesParams = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
    getMetricRawDependenciesParams.pMetricsEvaluator = pMetricsEvaluator;
    getMetricRawDependenciesParams.pMetricEvalRequests = metricEvalRequest;
    getMetricRawDependenciesParams.numMetricEvalRequests = 1; // Set to 1 as that is the number of eval requests we will have each time
    getMetricRawDependenciesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    getMetricRawDependenciesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
    getMetricRawDependenciesParams.ppRawDependencies = NULL;
    getMetricRawDependenciesParams.ppOptionalRawDependencies = NULL;
    getMetricRawDependenciesParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_MetricsEvaluator_GetMetricRawDependenciesPtr(&getMetricRawDependenciesParams), return PAPI_EMISC );

    const char **rawDependencies;
    rawDependencies = (const char **) malloc(getMetricRawDependenciesParams.numRawDependencies * sizeof(char *));
    if (rawDependencies == NULL) {
        SUBDBG("Failed to allocate memory for variable rawDependencies.\n");
        return PAPI_ENOMEM;
    }   
    getMetricRawDependenciesParams.ppRawDependencies = rawDependencies;
    nvpwCheckErrors( NVPW_MetricsEvaluator_GetMetricRawDependenciesPtr(&getMetricRawDependenciesParams), return PAPI_EMISC );

    *rawMetricRequests = (NVPA_RawMetricRequest *) realloc(*rawMetricRequests, (getMetricRawDependenciesParams.numRawDependencies + (*rawMetricRequestsCount)) * sizeof(NVPA_RawMetricRequest));
    if (rawMetricRequests == NULL) {
        SUBDBG("Failed to allocate memory for variable tmpRawMetricRequests.\n");
        return PAPI_ENOMEM;
    }   

    int i, tmpRawMetricRequestsCount = *rawMetricRequestsCount;
    for (i = 0; i < getMetricRawDependenciesParams.numRawDependencies; i++) {
       NVPA_RawMetricRequest rawMetricRequestParams = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
       rawMetricRequestParams.pPriv = NULL;
       rawMetricRequestParams.pMetricName = strdup(rawDependencies[i]);
       rawMetricRequestParams.isolated = 1;  
       rawMetricRequestParams.keepInstances = 1;  
       (*rawMetricRequests)[(*rawMetricRequestsCount)] = rawMetricRequestParams;
       (*rawMetricRequestsCount)++;
    }   
    free(rawDependencies);

    return PAPI_OK;
}

/** @class get_evaluated_metric_values
 *  @brief For a user added metric, get the evaluated gpu value.
 *
 *  @param *pMetricsEvaluator
 *    A NVPW_MetricsEvaluator struct. 
 *  @param *gpu_ctl
 *    Structure of type cuptip_gpu_state_t which has member variables such as 
 *    dev_id, rawMetricRequests, numberOfRawMetricRequests, and more.
 *  @param *evaluatedMetricValues
 *    Total number of raw metric requests created.
*/
static int get_evaluated_metric_values(NVPW_MetricsEvaluator *pMetricsEvaluator, cuptip_gpu_state_t *gpu_ctl, long long *evaluatedMetricValues)
{
    int i;
    for (i = 0; i < gpu_ctl->added_events->count; i++) {
        NVPW_MetricEvalRequest metricEvalRequest;
        get_metric_eval_request(pMetricsEvaluator, gpu_ctl->added_events->cuda_evts[i], &metricEvalRequest);

        NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttributeParams = {NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
        setDeviceAttributeParams.pMetricsEvaluator = pMetricsEvaluator;
        setDeviceAttributeParams.pCounterDataImage = (const uint8_t *) gpu_ctl->counterDataImage.data;
        setDeviceAttributeParams.counterDataImageSize = gpu_ctl->counterDataImage.size;
        nvpwCheckErrors( NVPW_MetricsEvaluator_SetDeviceAttributesPtr(&setDeviceAttributeParams), return PAPI_EMISC );

        double metricValue;
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = {NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evaluateToGpuValuesParams.pMetricsEvaluator = pMetricsEvaluator;
        evaluateToGpuValuesParams.pMetricEvalRequests =  &metricEvalRequest;
        evaluateToGpuValuesParams.numMetricEvalRequests = 1;
        evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        evaluateToGpuValuesParams.pCounterDataImage = gpu_ctl->counterDataImage.data;
        evaluateToGpuValuesParams.counterDataImageSize = gpu_ctl->counterDataImage.size;
        evaluateToGpuValuesParams.rangeIndex = 0;
        evaluateToGpuValuesParams.isolated = 1;
        evaluateToGpuValuesParams.pMetricValues = &metricValue;
        nvpwCheckErrors( NVPW_MetricsEvaluator_EvaluateToGpuValuesPtr(&evaluateToGpuValuesParams), return PAPI_EMISC );

        evaluatedMetricValues[i] = metricValue;
    }

    return PAPI_OK;
}

/** @class destroy_metric_evaluator
  * @brief A simple wrapper for the perfworks api call
  *        NVPW_MetricsEvaluator_Destroy.
*/
static int destroy_metrics_evaluator(NVPW_MetricsEvaluator *pMetricsEvaluator)
{
    NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = {NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    metricEvaluatorDestroyParams.pMetricsEvaluator = pMetricsEvaluator;
    metricEvaluatorDestroyParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_MetricsEvaluator_DestroyPtr(&metricEvaluatorDestroyParams), return PAPI_EMISC );

    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @name Functions necessary for the configuration/profiling stage
 *  @{
 */

/** @class start_profiling_session
 *  @brief Start a profiling session.
 *
 *  @param counterDataImage
 *    Contains the size and data.
 *  @param counterDataScratchBufferSize
 *    Contains the size and data.
 *  @param configImage
 *    Contains the size and data.
*/
static int start_profiling_session(byte_array_t counterDataImage, byte_array_t counterDataScratchBufferSize, byte_array_t configImage)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.counterDataImageSize = counterDataImage.size;
    beginSessionParams.pCounterDataImage = counterDataImage.data;
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBufferSize.size;
    beginSessionParams.pCounterDataScratchBuffer = counterDataScratchBufferSize.data;
    beginSessionParams.maxLaunchesPerPass = 1;
    beginSessionParams.maxRangesPerPass = 1;
    beginSessionParams.range = CUPTI_UserRange;
    beginSessionParams.replayMode = CUPTI_UserReplay;
    beginSessionParams.pPriv = NULL;
    beginSessionParams.ctx = NULL;
    cuptiCheckErrors( cuptiProfilerBeginSessionPtr(&beginSessionParams), return PAPI_EMISC );

    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pConfig = configImage.data;
    setConfigParams.configSize = configImage.size;
    // Only set for Application Replay mode.
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    setConfigParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerSetConfigPtr(&setConfigParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class get_config_image
 *  @brief Generate the ConfigImage binary configuration image 
 *         (file format in memory).
 *
 *  @param chipName
 *    Name of the device begin used.
 *  @param *pCounterAvailabilityImageData
 *    Data from cuptiProfilerGetCounterAvailability.
 *  @param *rawMetricRequests
 *    A filled in NVPA_RawMetricRequest.
 *  @para rmr_count
 *    Number of rawMetricRequests.  
 *  @param configImage
 *    Variable to store the generated configImage.
*/
static int get_config_image(const char *chipName, const uint8_t *pCounterAvailabilityImageData, NVPA_RawMetricRequest *rawMetricRequests, int rmr_count, byte_array_t *configImage)
{
    NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParamsV2 = {NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
    rawMetricsConfigCreateParamsV2.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    rawMetricsConfigCreateParamsV2.pChipName = chipName;
    rawMetricsConfigCreateParamsV2.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_RawMetricsConfig_Create_V2Ptr(&rawMetricsConfigCreateParamsV2), return PAPI_EMISC );
    // Destory pRawMetricsConfig at the end; otherwise, a memory leak will occur
    NVPA_RawMetricsConfig *pRawMetricsConfig = rawMetricsConfigCreateParamsV2.pRawMetricsConfig;

    // Query counter availability before starting the profiling session
    if (pCounterAvailabilityImageData) {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
	setCounterAvailabilityParams.pPriv = NULL;
	setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
	setCounterAvailabilityParams.pCounterAvailabilityImage = pCounterAvailabilityImageData;
        nvpwCheckErrors( NVPW_RawMetricsConfig_SetCounterAvailabilityPtr(&setCounterAvailabilityParams), return PAPI_EMISC );
    }

    // NOTE: maxPassCount is being set to 1 as a final safety net to limit metric collection to a single pass.
    //       Metrics that require multiple passes would fail further down at AddMetrics due to this.
    //       This failure should never occur as we filter for metrics with multiple passes at get_number_of_passes,
    //       which occurs before the get_config_image call.
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    beginPassGroupParams.maxPassCount = 1;
    beginPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = rawMetricRequests;
    addMetricsParams.numMetricRequests = rmr_count;
    addMetricsParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    endPassGroupParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
    generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    generateConfigImageParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GenerateConfigImagePtr(&generateConfigImageParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
    getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    getConfigImageParams.bytesAllocated = 0;
    getConfigImageParams.pBuffer = NULL;
    getConfigImageParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), return PAPI_EMISC );

    byte_array_t *tmpConfigImage;
    tmpConfigImage = configImage;

    tmpConfigImage->size = getConfigImageParams.bytesCopied;
    tmpConfigImage->data = (uint8_t *) calloc(tmpConfigImage->size, sizeof(uint8_t));
    if (configImage->data == NULL) {
        SUBDBG("Failed to allocate memory for configImage->data.\n");
        return PAPI_ENOMEM;
    }

    getConfigImageParams.bytesAllocated = tmpConfigImage->size;
    getConfigImageParams.pBuffer = tmpConfigImage->data;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), return PAPI_EMISC );

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    rawMetricsConfigDestroyParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams), return PAPI_EMISC );

    return PAPI_OK;
}


/** @class get_counter_data_prefix_image
 *  @brief Generate the counterDataPrefix binary configuration image 
 *         (file format in memory).
 *
 *  @param chipName
 *    Name of the device begin used.
 *  @param *rawMetricRequests
 *    A filled in NVPA_RawMetricRequest.
 *  @param rmr_count
 *    Number of rawMetricRequests.  
 *  @param obtainCounterDataPrefixImage
 *    Variable to store the generated counterDataPrefix.
*/
static int get_counter_data_prefix_image(const char *chipName, NVPA_RawMetricRequest *rawMetricRequests, int rmr_count, byte_array_t *counterDataPrefixImage)
{
    NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE};
    counterDataBuilderCreateParams.pChipName = chipName;
    counterDataBuilderCreateParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CUDA_CounterDataBuilder_CreatePtr(&counterDataBuilderCreateParams), return PAPI_EMISC );

    NVPW_CounterDataBuilder_AddMetrics_Params builderAddMetricsParams = {NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
    builderAddMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    builderAddMetricsParams.pRawMetricRequests = rawMetricRequests;
    builderAddMetricsParams.numMetricRequests = rmr_count;
    builderAddMetricsParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CounterDataBuilder_AddMetricsPtr(&builderAddMetricsParams), return PAPI_EMISC );

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    getCounterDataPrefixParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), return PAPI_EMISC );

    byte_array_t *tmpCounterDataPrefixImage;
    tmpCounterDataPrefixImage = counterDataPrefixImage;
    tmpCounterDataPrefixImage->size = getCounterDataPrefixParams.bytesCopied;
    tmpCounterDataPrefixImage->data = (uint8_t *) calloc(tmpCounterDataPrefixImage->size, sizeof(uint8_t));
    if (tmpCounterDataPrefixImage->data == NULL) {
        SUBDBG("Failed to allocate memory for tmpCounterDataPrefixImage->data.\n");
        return PAPI_ENOMEM;
    }

    getCounterDataPrefixParams.bytesAllocated = tmpCounterDataPrefixImage->size;
    getCounterDataPrefixParams.pBuffer = tmpCounterDataPrefixImage->data;
    nvpwCheckErrors( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), return PAPI_EMISC );

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
    counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    counterDataBuilderDestroyParams.pPriv = NULL;
    nvpwCheckErrors( NVPW_CounterDataBuilder_DestroyPtr((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class get_counter_data_image
 *  @brief Create a counterDataImage to be used for metric evaluation. 
 *
 *  @param counterDataPrefixImage
 *    Struct containing the size and data of the counterDataPrefix
 *    binary configuration image.
 *  @param counterDataScratchBuffer
 *    Struct to store the size and data of the scratch buffer.
 *  @param counterDataImage
 *    Struct to store the size and data of the counterDataImage.
*/
static int get_counter_data_image(byte_array_t counterDataPrefixImage, byte_array_t *counterDataScratchBuffer, byte_array_t *counterDataImage)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = counterDataPrefixImage.data;
    counterDataImageOptions.counterDataPrefixSize = counterDataPrefixImage.size;
    counterDataImageOptions.maxNumRanges = 1;
    counterDataImageOptions.maxNumRangeTreeNodes = 1; // Why do we do this?
    counterDataImageOptions.maxRangeNameLength = 64; 

    // Calculate size of counterDataImage based on counterDataPrefixImage and options.
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    calculateSizeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerCounterDataImageCalculateSizePtr(&calculateSizeParams), return PAPI_EMISC );

   // Initialize counterDataImage.
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initializeParams.pPriv = NULL;

    byte_array_t *tmpCounterDataImage;
    tmpCounterDataImage = counterDataImage;

    tmpCounterDataImage->size = calculateSizeParams.counterDataImageSize;
    tmpCounterDataImage->data = (uint8_t *) calloc(tmpCounterDataImage->size, sizeof(uint8_t));
    if (counterDataImage->data  == NULL) {
        SUBDBG("Failed to allocate memory for counterDataImage->data.\n");
        return PAPI_ENOMEM;
    }

    initializeParams.pCounterDataImage = counterDataImage->data;
    cuptiCheckErrors( cuptiProfilerCounterDataImageInitializePtr(&initializeParams), return PAPI_EMISC );

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage.
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = counterDataImage->data;
    scratchBufferSizeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr(&scratchBufferSizeParams), return PAPI_EMISC );

    // Create counterDataScratchBuffer.
    byte_array_t *tmpCounterDataScratchBuffer;
    tmpCounterDataScratchBuffer = counterDataScratchBuffer;
    tmpCounterDataScratchBuffer->size = scratchBufferSizeParams.counterDataScratchBufferSize;
    tmpCounterDataScratchBuffer->data = (uint8_t *) calloc(tmpCounterDataScratchBuffer->size, sizeof(uint8_t));
    if (counterDataScratchBuffer->data == NULL) {
        SUBDBG("Failed to allocate memory for counterDataScratchBuffer->data.\n");
        return PAPI_ENOMEM;
    }   

    // Initialize counterDataScratchBuffer.
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = counterDataImage->data; //uint8_t* pCounterDataImage
    initScratchBufferParams.counterDataScratchBufferSize = counterDataScratchBuffer->size;
    initScratchBufferParams.pCounterDataScratchBuffer = counterDataScratchBuffer->data;
    initScratchBufferParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&initScratchBufferParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class end_profiling_session
 *  @brief End the started profiling session.
*/
static int end_profiling_session(void)
{
    int papi_errno = disable_profiling();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = pop_range();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = flush_data();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = unset_config();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = end_session();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @name   Wrappers for cupti profiler api calls
 *  @{
 */

/** @class initialize_cupti_profiler_api
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerInitialize.
*/
static int initialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    profilerInitializeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerInitializePtr(&profilerInitializeParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class deinitialize_cupti_profiler_api
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerDeInitialize.
*/
static int deinitialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");

    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    profilerDeInitializeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerDeInitializePtr(&profilerDeInitializeParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class enable_profiling
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerEnableProfiling.
*/
static int enable_profiling(void)
{
   CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
   enableProfilingParams.ctx = NULL; // If NULL, the current CUcontext is used
   enableProfilingParams.pPriv = NULL;
   cuptiCheckErrors( cuptiProfilerEnableProfilingPtr(&enableProfilingParams), return PAPI_EMISC );

   return PAPI_OK;
}

/** @class begin_pass
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerBeginPass.
*/
int begin_pass(void)
{
    CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    beginPassParams.ctx = NULL; // If NULL, the current CUcontext is used
    beginPassParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerBeginPassPtr(&beginPassParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class end_pass
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerEndPass.
*/
static int end_pass(void)
{
    CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    endPassParams.ctx = NULL; // If NULL, the current CUcontext is used
    endPassParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerEndPassPtr(&endPassParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class push_range
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerPushRange.
*/
static int push_range(const char *pRangeName)
{
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    pushRangeParams.pRangeName = pRangeName;
    pushRangeParams.rangeNameLength = strlen(pRangeName);
    pushRangeParams.ctx = NULL; // If NULL, the current CUcontext is used
    pushRangeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerPushRangePtr(&pushRangeParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class pop_range
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerPopRange.
*/
static int pop_range(void)
{
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    popRangeParams.ctx = NULL; // If NULL, the current CUcontext is used
    popRangeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerPopRangePtr(&popRangeParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class flush_data
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerFlushCounterData.
  *
  *        Note that Flush is required to ensure data is returned from the 
  *        device when running User Replay mode.
*/
static int flush_data(void)
{
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    flushCounterDataParams.ctx = NULL; // If NULL, the current CUcontext is used
    flushCounterDataParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class disable_profiling
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerDisableProfiling.
*/
static int disable_profiling(void)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    disableProfilingParams.ctx = NULL; // If NULL, the current CUcontext is used
    disableProfilingParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerDisableProfilingPtr(&disableProfilingParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class unset_config
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerUnsetConfig.
*/
static int unset_config(void)
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    unsetConfigParams.ctx = NULL; // If NULL, the current CUcontext is used
    unsetConfigParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerUnsetConfigPtr(&unsetConfigParams), return PAPI_EMISC );

    return PAPI_OK;
}

/** @class end_session
  * @brief A simple wrapper for the cupti profiler api call
  *        cuptiProfilerEndSession.
*/
static int end_session(void)
{
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    endSessionParams.ctx = NULL; // If NULL, the current CUcontext is used
    endSessionParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerEndSessionPtr(&endSessionParams), return PAPI_EMISC );

    return PAPI_OK;
}

/**
 *  @}
 ******************************************************************************/
