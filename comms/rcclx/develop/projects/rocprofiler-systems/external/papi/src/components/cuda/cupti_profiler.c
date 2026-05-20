/**
 * @file    cupti_profiler.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include <dlfcn.h>
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
 * +---------------------------------+-------+----+------------+
 * |         unused                  |  dev  | ql |   nameid   |
 * +---------------------------------+-------+----+------------+
 *
 * unused    : 34 bits 
 * device    : 7  bits ([0 - 127] devices)
 * qlmask    : 2  bits (qualifier mask)
 * nameid    : 21: bits (roughly > 2 million event names)
 */
#define EVENTS_WIDTH (sizeof(uint64_t) * 8)
#define DEVICE_WIDTH ( 7)
#define QLMASK_WIDTH ( 2) 
#define NAMEID_WIDTH (21)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - DEVICE_WIDTH)
#define QLMASK_SHIFT (DEVICE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define DEVICE_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define DEVICE_FLAG  (0x1)

typedef struct byte_array_s         byte_array_t;
typedef struct cuptip_gpu_state_s   cuptip_gpu_state_t;
typedef struct NVPA_MetricsContext  NVPA_MetricsContext;

typedef struct {
    int device;
    int flags;
    int nameid;
} event_info_t;


struct byte_array_s {
    int      size;
    uint8_t *data;
};

struct cuptip_gpu_state_s {
    int                    gpu_id;
    cuptiu_event_table_t  *added_events;
    int                    rmr_count;
    NVPA_RawMetricRequest *rmr;
    MCCP_t                *pmetricsContextCreateParams;
    byte_array_t           counterDataImagePrefix;
    byte_array_t           configImage;
    byte_array_t           counterDataImage;
    byte_array_t           counterDataScratchBuffer;
    byte_array_t           counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams;
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams;
};

struct cuptip_control_s {
    cuptip_gpu_state_t *gpu_ctl;
    long long           *counters;
    int                 read_count;
    int                 running;
    cuptic_info_t       info;
};

static void *dl_nvpw;
static int num_gpus;
static gpu_record_t *avail_gpu_info;

/* main event table to store metrics */
static cuptiu_event_table_t *cuptiu_table_p;

/* load and unload cuda function pointers */
static int load_cupti_perf_sym(void);
static int unload_cupti_perf_sym(void);

/* load and unload nvperf function pointers */
static int load_nvpw_sym(void);
static int unload_nvpw_sym(void);

/* utility functions to initialize API's such as cupti and perfworks */
static int initialize_cupti_profiler_api(void);
static int deinitialize_cupti_profiler_api(void);
static int initialize_perfworks_api(void);

/* utility functions to init metrics and cuda native event table */
static int init_all_metrics(void);
static int init_main_htable(void);
static int init_event_table(void);
static int shutdown_event_table(void);
static void free_all_enumerated_metrics(void);

/* functions to handle contexts */
static int nvpw_cuda_metricscontext_create(cuptip_control_t state);
static int nvpw_cuda_metricscontext_destroy(cuptip_control_t state);

/* funtions for config images */
static int metric_get_config_image(cuptip_gpu_state_t *gpu_ctl);
static int metric_get_counter_data_prefix_image(cuptip_gpu_state_t *gpu_ctl);
static int create_counter_data_image(cuptip_gpu_state_t *gpu_ctl);
static int reset_cupti_prof_config_images(cuptip_gpu_state_t *gpu_ctl);

/* functions to set up profiling and end profiling */
static int begin_profiling(cuptip_gpu_state_t *gpu_ctl);
static int end_profiling(cuptip_gpu_state_t *gpu_ctl);

/* NVIDIA chip functions */
static int get_chip_name(int dev_num, char* chipName);
static int find_same_chipname(int gpu_id);

/* functions to check if a cuda native event requires multiple passes */
static int check_multipass(cuptip_control_t state);
static int calculate_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count,
                                NVPA_RawMetricRequest *rmr, int *num_pass);

/* functions to set and get cuda native event info  or convert cuda native events  */
static int get_ntv_events(cuptiu_event_table_t *evt_table, const char *evt_name, int gpu_id);
static int verify_events(uint64_t *events_id, int num_events, cuptip_control_t state);
static int evt_id_to_info(uint64_t event_id, event_info_t *info);
static int evt_id_create(event_info_t *info, uint64_t *event_id);
static int evt_code_to_name(uint64_t event_code, char *name, int len);
static int evt_name_to_basename(const char *name, char *base, int len);
static int evt_name_to_device(const char *name, int *device);
static int retrieve_metric_descr( NVPA_MetricsContext *pMetricsContext, const char *evt_name,
                                  char *description, const char *chip_name );
static int retrieve_metric_rmr( NVPA_MetricsContext *pMetricsContext, const char *evt_name,
                                int *numDep, NVPA_RawMetricRequest **pRMR );

/* misc */
static int get_event_collection_method(const char *evt_name);
static int get_added_events_rmr(cuptip_gpu_state_t *gpu_ctl);
static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl);
static int get_measured_values(cuptip_gpu_state_t *gpu_ctl, long long *counts);

/* nvperf function pointers */
NVPA_Status ( *NVPW_GetSupportedChipNamesPtr ) (NVPW_GetSupportedChipNames_Params* params);
NVPA_Status ( *NVPW_CUDA_MetricsContext_CreatePtr ) (NVPW_CUDA_MetricsContext_Create_Params* params);
NVPA_Status ( *NVPW_MetricsContext_DestroyPtr ) (NVPW_MetricsContext_Destroy_Params * params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_BeginPtr ) (NVPW_MetricsContext_GetMetricNames_Begin_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_EndPtr ) (NVPW_MetricsContext_GetMetricNames_End_Params* params);
NVPA_Status ( *NVPW_InitializeHostPtr ) (NVPW_InitializeHost_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_BeginPtr ) (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_EndPtr ) (NVPW_MetricsContext_GetMetricProperties_End_Params* p);
NVPA_Status ( *NVPW_CUDA_RawMetricsConfig_CreatePtr ) (NVPW_CUDA_RawMetricsConfig_Create_Params*);
NVPA_Status ( *NVPW_RawMetricsConfig_DestroyPtr ) (NVPW_RawMetricsConfig_Destroy_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_BeginPassGroupPtr ) (NVPW_RawMetricsConfig_BeginPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_EndPassGroupPtr ) (NVPW_RawMetricsConfig_EndPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_AddMetricsPtr ) (NVPW_RawMetricsConfig_AddMetrics_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GenerateConfigImagePtr ) (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetConfigImagePtr ) (NVPW_RawMetricsConfig_GetConfigImage_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_CreatePtr ) (NVPW_CounterDataBuilder_Create_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_DestroyPtr ) (NVPW_CounterDataBuilder_Destroy_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_AddMetricsPtr ) (NVPW_CounterDataBuilder_AddMetrics_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_GetCounterDataPrefixPtr ) (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params);
NVPA_Status ( *NVPW_CounterData_GetNumRangesPtr ) (NVPW_CounterData_GetNumRanges_Params* params);
NVPA_Status ( *NVPW_Profiler_CounterData_GetRangeDescriptionsPtr ) (NVPW_Profiler_CounterData_GetRangeDescriptions_Params* params);
NVPA_Status ( *NVPW_MetricsContext_SetCounterDataPtr ) (NVPW_MetricsContext_SetCounterData_Params* params);
NVPA_Status ( *NVPW_MetricsContext_EvaluateToGpuValuesPtr ) (NVPW_MetricsContext_EvaluateToGpuValues_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetNumPassesPtr ) (NVPW_RawMetricsConfig_GetNumPasses_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_SetCounterAvailabilityPtr ) (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr ) (NVPW_RawMetricsConfig_IsAddMetricsPossible_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_BeginPtr ) (NVPW_MetricsContext_GetCounterNames_Begin_Params* pParams);
NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_EndPtr ) (NVPW_MetricsContext_GetCounterNames_End_Params* pParams);

/* cupti function pointers */
CUptiResult ( *cuptiDeviceGetChipNamePtr ) (CUpti_Device_GetChipName_Params* params);
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


/** @class load_cupti_perf_sym
  * @brief Load cupti functions and assign to function pointers.
*/
static int load_cupti_perf_sym(void)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    if (dl_cupti == NULL) {
        ERRDBG("libcupti.so should already be loaded.\n");
        goto fn_fail;
    }

    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
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

fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
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
    cuptiDeviceGetChipNamePtr                                  = NULL;
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
 * @brief Search for libnvperf_host.so. Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_PERFWORKS, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect libnvperf_host.so from PAPI_CUDA_PERFWORKS or it is not set,
 *    we will search the path defined with PAPI_CUDA_ROOT; as this is supposed to always be set.
 * 3. If we fail to collect libnvperf_host.so from steps 1 and 2, then we will search the linux
 *    default directories listed by /etc/ld.so.conf. As a note, updating the LD_LIBRARY_PATH is
 *    advised for this option.
 * 4. We use dlopen to search for libnvperf_host.so.
 *    If this fails, then we failed to find libnvperf_host.so.
 */
static int load_nvpw_sym(void)
{
    COMPDBG("Entering.\n");
    char dlname[] = "libnvperf_host.so";
    char lookup_path[PATH_MAX];

    /* search PAPI_CUDA_PERFWORKS for libnvperf_host.so (takes precedent over PAPI_CUDA_ROOT) */
    char *papi_cuda_perfworks = getenv("PAPI_CUDA_PERFWORKS");
    if (papi_cuda_perfworks) {
        sprintf(lookup_path, "%s/%s", papi_cuda_perfworks, dlname);
        dl_nvpw = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
    }

    const char *standard_paths[] = {
        "%s/extras/CUPTI/lib64/%s",
        "%s/lib64/%s",
        NULL,
    };

    /* search PAPI_CUDA_ROOT for libnvperf_host.so */
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_nvpw) {
        dl_nvpw = cuptic_load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
    }

    /* search linux default directories for libnvperf_host.so */
    if (linked_cudart_path && !dl_nvpw) {
        dl_nvpw = cuptic_load_dynamic_syms(linked_cudart_path, dlname, standard_paths);
    }

    /* last ditch effort to find libcupti.so */
    if (!dl_nvpw) {
        dl_nvpw = dlopen(dlname, RTLD_NOW | RTLD_GLOBAL);
        if (!dl_nvpw) {
            ERRDBG("Loading libnvperf_host.so failed.\n");
            goto fn_fail;
        }
    }

    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_GetSupportedChipNames");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_End");
    NVPW_CUDA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_SetCounterAvailability");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_IsAddMetricsPossible");
    NVPW_MetricsContext_GetCounterNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_End");

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
    NVPW_GetSupportedChipNamesPtr                     = NULL;
    NVPW_CUDA_MetricsContext_CreatePtr                = NULL;
    NVPW_MetricsContext_DestroyPtr                    = NULL;
    NVPW_MetricsContext_GetMetricNames_BeginPtr       = NULL;
    NVPW_MetricsContext_GetMetricNames_EndPtr         = NULL;
    NVPW_InitializeHostPtr                            = NULL;
    NVPW_MetricsContext_GetMetricProperties_BeginPtr  = NULL;
    NVPW_MetricsContext_GetMetricProperties_EndPtr    = NULL;
    NVPW_CUDA_RawMetricsConfig_CreatePtr              = NULL;
    NVPW_RawMetricsConfig_DestroyPtr                  = NULL;
    NVPW_RawMetricsConfig_BeginPassGroupPtr           = NULL;
    NVPW_RawMetricsConfig_EndPassGroupPtr             = NULL;
    NVPW_RawMetricsConfig_AddMetricsPtr               = NULL;
    NVPW_RawMetricsConfig_GenerateConfigImagePtr      = NULL;
    NVPW_RawMetricsConfig_GetConfigImagePtr           = NULL;
    NVPW_CounterDataBuilder_CreatePtr                 = NULL;
    NVPW_CounterDataBuilder_DestroyPtr                = NULL;
    NVPW_CounterDataBuilder_AddMetricsPtr             = NULL;
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr   = NULL;
    NVPW_CounterData_GetNumRangesPtr                  = NULL;
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = NULL;
    NVPW_MetricsContext_SetCounterDataPtr             = NULL;
    NVPW_MetricsContext_EvaluateToGpuValuesPtr        = NULL;
    NVPW_RawMetricsConfig_GetNumPassesPtr             = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr   = NULL;
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr     = NULL;
    NVPW_MetricsContext_GetCounterNames_BeginPtr      = NULL;
    NVPW_MetricsContext_GetCounterNames_EndPtr        = NULL;
    return PAPI_OK;
}

/** @class initialize_cupti_profiler_api
  * @brief Initialize the cupti profiler interface..
*/
static int initialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE, NULL };
    papi_errno = cuptiProfilerInitializePtr(&profilerInitializeParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: cuptiProfilerInitialize failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

/** @class deinitialize_cupti_profiler_api
  * @brief Deinitialize the cupti profiler interface.
*/
static int deinitialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = { CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE, NULL };
    papi_errno = cuptiProfilerDeInitializePtr(&profilerDeInitializeParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI Error %d: cuptiProfilerDeInitialize failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}


/** @class initialize_perfworks_api
  * @brief NVPW required initialization.
*/
static int initialize_perfworks_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    NVPW_InitializeHost_Params perfInitHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE, NULL };
    papi_errno = NVPW_InitializeHostPtr(&perfInitHostParams);
    if (papi_errno != NVPA_STATUS_SUCCESS) {
        ERRDBG("NVPW Error %d: NVPW_InitializeHostPtr failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int get_chip_name(int dev_num, char* chipName)
{
    int papi_errno;
    CUpti_Device_GetChipName_Params getChipName = {
        .structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .deviceIndex = 0
    };
    getChipName.deviceIndex = dev_num;
    papi_errno = cuptiDeviceGetChipNamePtr(&getChipName);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get chip name for device %d\n", papi_errno, dev_num);
        return PAPI_EMISC;
    }
    strcpy(chipName, getChipName.pChipName);
    return PAPI_OK;
}

/** @class get_added_events_rmr
  * @brief For a Cuda native event name collect raw metrics and count
  *        of raw metrics for collection. Raw Metrics are one layer of the Metric API
  *        and contains the list of raw counters and generates configuration file
  *        images. Must be done before creating a ConfigImage or 
  *        CounterDataPrefix.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int get_added_events_rmr(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    int gpu_id, num_dep, count_raw_metrics = 0, papi_errno = PAPI_OK;
    int i, j, k;
    NVPA_RawMetricRequest *all_rmr=NULL, *collect_rmr;
    cuptiu_event_t *evt_rec;

    /* for each event in the event table collect the raw metric requests */
    for (i = 0; i < gpu_ctl->added_events->count; i++) {
        papi_errno = retrieve_metric_rmr(
                         gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
                         gpu_ctl->added_events->cuda_evts[i], &num_dep, 
                         &collect_rmr
                     );
        if (papi_errno != PAPI_OK) {
            papi_errno = PAPI_ENOEVNT;
            goto fn_exit;
        }
        all_rmr = (NVPA_RawMetricRequest *) papi_realloc(all_rmr, (count_raw_metrics + num_dep) * sizeof(NVPA_RawMetricRequest));
        if (all_rmr == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        for (j = 0; j < num_dep; j++) {
            k = j + count_raw_metrics;
            all_rmr[k].structSize = collect_rmr[j].structSize;
            all_rmr[k].pPriv = NULL;
            all_rmr[k].pMetricName = strdup(collect_rmr[j].pMetricName);
            all_rmr[k].keepInstances = 1;
            all_rmr[k].isolated = 1;
            papi_free((void *) collect_rmr[j].pMetricName);
        }
        count_raw_metrics += num_dep;
        papi_free(collect_rmr);
    } 
    gpu_ctl->rmr = all_rmr;
    gpu_ctl->rmr_count = count_raw_metrics;
fn_exit:
    return papi_errno;
}

/** @class calculate_num_passes
  * @brief Calculate the numbers of passes for a Cuda native event.
  * @param state
*/
static int calculate_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count, NVPA_RawMetricRequest *rmr, int *num_pass)
{
    COMPDBG("Entering.\n");
    int numNestingLevels = 1, numIsolatedPasses, numPipelinedPasses;
    NVPA_Status nvpa_err;

    /* NOTE: maxPassCount is not set here as we want to properly show the number of passes for
             metrics that require multiple passes in papi_native_avail. */
    /* instantiate a new struct to be passed to NVPW_RawMetricsConfig_BeginPassGroup_Params */
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        // [in]
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    nvpa_err = NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    
    /* instantiate struct to be passed to NVPW_RawMetricsConfig_AddMetrics */
    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        // [in]
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .pRawMetricsConfig = pRawMetricsConfig,
        .pRawMetricRequests = rmr,
        .numMetricRequests = rmr_count,
    };
    nvpa_err = NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    /* instantiate a new struct to be passed to NVPW_RawMetricsConfig_EndPassGroup */
    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        // [in]
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    nvpa_err = NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    /* instantiate a new struct to be passed to  NVPW_RawMetricsConfig_GetNumPasses_Params*/
    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = {
        // [in]
       .structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE,
       .pPriv = NULL, // assign to NULL
       .pRawMetricsConfig = pRawMetricsConfig,
    };
    nvpa_err = NVPW_RawMetricsConfig_GetNumPassesPtr(&rawMetricsConfigGetNumPassesParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    /* calculate numpass */
    numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;
    *num_pass = numPipelinedPasses + numIsolatedPasses * numNestingLevels;
    if (*num_pass > 1) {
        ERRDBG("Metrics requested requires multiple passes to profile.\n");
        return PAPI_EMULPASS;
    }

    return PAPI_OK;
}


/** @class nvpw_cuda_metricscontext_create
  * @brief Create a pMetricsContext.
  *
  * @param state
  *     Struct that holds read count, running, cuptip_info_t, and cuptip_gpu_state_t.
*/
static int nvpw_cuda_metricscontext_create(cuptip_control_t state)
{
    int gpu_id, found, papi_errno = PAPI_OK;
    MCCP_t *pMCCP;
    NVPA_Status nvpa_err;
    /* struct that holds gpu_id, rmr_count, configImage etc.
       seee cuptip_gpu_state_s */
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            gpu_ctl->pmetricsContextCreateParams = state->gpu_ctl[found].pmetricsContextCreateParams;
            continue;
        }
        /* struct that holds metadata for call to NVPW_CUDA_MetricsContext_CreatePtr 
           this includes struct size and gpu chip name */
        pMCCP = (MCCP_t *) papi_calloc( 1, sizeof(MCCP_t) );
        /* see if struct allocated memory properly */
        if (pMCCP == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        
        /* setting metadata values */
        pMCCP->structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE;
        pMCCP->pChipName = cuptiu_table_p->avail_gpu_info[gpu_id].chip_name;

        /* create context */
        nvpa_err = NVPW_CUDA_MetricsContext_CreatePtr(pMCCP);
        if (nvpa_err != NVPA_STATUS_SUCCESS)
            goto fn_fail ;

        /* store created context in cuptip_control_t state */
        gpu_ctl->pmetricsContextCreateParams = pMCCP;
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class nvpw_cuda_metricscontext_destroy
  * @brief Destroy created context from nvpw_cuda_metricscontext_create.
  *
  * @param state
  *     Struct that holds read count, running, cuptip_info_t, and cuptip_gpu_state_t.
*/
static int nvpw_cuda_metricscontext_destroy(cuptip_control_t state)
{
    int gpu_id, found, papi_errno = PAPI_OK;
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            gpu_ctl->pmetricsContextCreateParams = NULL;
            continue;
        }
        if (gpu_ctl->pmetricsContextCreateParams->pMetricsContext) {
            NVPW_MetricsContext_Destroy_Params mCDP = {
                .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
            };
            nvpwCheckErrors( NVPW_MetricsContext_DestroyPtr(&mCDP), goto fn_fail );
            papi_free(gpu_ctl->pmetricsContextCreateParams);
            gpu_ctl->pmetricsContextCreateParams = NULL;
        }
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class check_multipass
  * @brief Check to see if the Cuda native event is multi-pass. Multi-pass Cuda
  *        native events (Numpass > 1), is not supported.
  * @param state
*/
static int check_multipass(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    int gpu_id, papi_errno, passes;
    NVPA_Status nvpa_err;
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }

        papi_errno = get_added_events_rmr(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }

        /* perfworks api: instantiate a new stuct to be passed to NVPW_CUDA_RawMetricsConfig_CreatePtr */ 
        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
            .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = cuptiu_table_p->avail_gpu_info[gpu_id].chip_name,
        };
        nvpa_err = NVPW_CUDA_RawMetricsConfig_CreatePtr(
                       &nvpw_metricsConfigCreateParams
                   );
        if (nvpa_err != NVPA_STATUS_SUCCESS) {
            goto fn_exit;
        }

        /* for an event, collect the number of passes to see if supported */
        papi_errno = calculate_num_passes( nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                                           gpu_ctl->rmr_count, gpu_ctl->rmr, &passes);
        if ( papi_errno == PAPI_EMULPASS ) {
        /* at this point we just want the number of passes (stored in passes) */
        }

        /* perfworks api: instantiate a new stuct to be passed to NVPW_CUDA_RawMetricsConfig_DestroyPtr */
        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
            .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        };
        nvpa_err = NVPW_RawMetricsConfig_DestroyPtr(
                       (NVPW_RawMetricsConfig_Destroy_Params *) 
                       &rawMetricsConfigDestroyParams
                   );
        if (nvpa_err != NVPA_STATUS_SUCCESS) {
            goto fn_fail;
        }
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class get_counter_availability
  * @brief Query counter availability. Helps to filter unavailable raw metrics on host.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl)
{
    int papi_errno;
    /* Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data */
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
        .structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pCounterAvailabilityImage = NULL,
    };
    papi_errno = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get size.\n", papi_errno);
        return PAPI_EMISC;
    }
    /* Allocate sized counterAvailabilityImage */
    gpu_ctl->counterAvailabilityImage.size = getCounterAvailabilityParams.counterAvailabilityImageSize;
    gpu_ctl->counterAvailabilityImage.data = (uint8_t *) papi_malloc(gpu_ctl->counterAvailabilityImage.size);
    if (gpu_ctl->counterAvailabilityImage.data == NULL) {
        return PAPI_ENOMEM;
    }
    /* Initialize counterAvailabilityImage */
    getCounterAvailabilityParams.pCounterAvailabilityImage = gpu_ctl->counterAvailabilityImage.data;
    papi_errno = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get bytes.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}


/** @class metric_get_config_image
  * @brief Retrieves binary ConfigImage for the Cuda native event metrics listed 
  *        for collection. The function get_added_events_rmr( ... ) must be 
  *        called before this step is possible. 
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int metric_get_config_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    int gpu_id = gpu_ctl->gpu_id;

    NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
        .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
        .pChipName = cuptiu_table_p->avail_gpu_info[gpu_id].chip_name,
    };
    nvpwCheckErrors( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), goto fn_fail );

    if( gpu_ctl->counterAvailabilityImage.data != NULL) {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {
            .structSize = NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
            .pCounterAvailabilityImage = gpu_ctl->counterAvailabilityImage.data,
        };
        nvpwCheckErrors( NVPW_RawMetricsConfig_SetCounterAvailabilityPtr(&setCounterAvailabilityParams), goto fn_fail );
    }

    /* NOTE: maxPassCount is being set to 1 as a final safety net to limit metric collection to a single pass.
             Metrics that require multiple passes would fail further down at AddMetrics due to this.
             This failure should never occur as we filter for metrics with multiple passes at check_multipass,
             which occurs before the metric_get_config_image call. */
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .maxPassCount = 1,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .pRawMetricRequests = gpu_ctl->rmr,
        .numMetricRequests = gpu_ctl->rmr_count,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), goto fn_fail );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_GenerateConfigImagePtr(&generateConfigImageParams), goto fn_fail );

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), goto fn_fail );

    gpu_ctl->configImage.size = getConfigImageParams.bytesCopied;
    gpu_ctl->configImage.data = (uint8_t *) papi_calloc(gpu_ctl->configImage.size, sizeof(uint8_t));
    if (gpu_ctl->configImage.data == NULL) {
        ERRDBG("calloc gpu_ctl->configImage.data failed!");
        return PAPI_ENOMEM;
    }

    getConfigImageParams.bytesAllocated = gpu_ctl->configImage.size;
    getConfigImageParams.pBuffer = gpu_ctl->configImage.data;
    nvpwCheckErrors( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), goto fn_fail );

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    nvpwCheckErrors( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class metric_get_counter_data_prefix_image
  * @brief Retrieves binary CounterDataPrefix for the Cuda native event metrics 
  *        listed for collection. The function get_added_events_rmr( ... ) 
  *        must be called before this step is possible. 
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int metric_get_counter_data_prefix_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    int gpu_id = gpu_ctl->gpu_id;

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        .structSize = NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pChipName = cuptiu_table_p->avail_gpu_info[gpu_id].chip_name,
    };
    nvpwCheckErrors( NVPW_CounterDataBuilder_CreatePtr(&counterDataBuilderCreateParams), goto fn_fail );

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .pRawMetricRequests = gpu_ctl->rmr,
        .numMetricRequests = gpu_ctl->rmr_count,
    };
    nvpwCheckErrors( NVPW_CounterDataBuilder_AddMetricsPtr(&addMetricsParams), goto fn_fail );

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        .structSize = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    nvpwCheckErrors( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), goto fn_fail );

    gpu_ctl->counterDataImagePrefix.size = getCounterDataPrefixParams.bytesCopied;
    gpu_ctl->counterDataImagePrefix.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataImagePrefix.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataImagePrefix.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataImagePrefix.data failed!");
        return PAPI_ENOMEM;
    }

    getCounterDataPrefixParams.bytesAllocated = gpu_ctl->counterDataImagePrefix.size;
    getCounterDataPrefixParams.pBuffer = gpu_ctl->counterDataImagePrefix.data;
    nvpwCheckErrors( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), goto fn_fail );

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        .structSize = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
    };
    nvpwCheckErrors( NVPW_CounterDataBuilder_DestroyPtr(&counterDataBuilderDestroyParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class create_counter_data_image
  * @brief Allocate space for values for each counter for each range and
  *        calculate a scratch buffer size needed for internal operations. 
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int create_counter_data_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    gpu_ctl->counterDataImageOptions = (CUpti_Profiler_CounterDataImageOptions) {
        .structSize = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataPrefix = gpu_ctl->counterDataImagePrefix.data,
        .counterDataPrefixSize = gpu_ctl->counterDataImagePrefix.size,
        .maxNumRanges = 1,
        .maxNumRangeTreeNodes = 1,
        .maxRangeNameLength = 64,
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &gpu_ctl->counterDataImageOptions,
    };
    cuptiCheckErrors( cuptiProfilerCounterDataImageCalculateSizePtr(&calculateSizeParams), goto fn_fail );

    gpu_ctl->initializeParams = (CUpti_Profiler_CounterDataImage_Initialize_Params) {
        .structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &gpu_ctl->counterDataImageOptions,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
    };

    gpu_ctl->counterDataImage.size = calculateSizeParams.counterDataImageSize;
    gpu_ctl->counterDataImage.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataImage.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataImage.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataImage.data failed!\n");
        return PAPI_ENOMEM;
    }

    gpu_ctl->initializeParams.pCounterDataImage = gpu_ctl->counterDataImage.data;
    cuptiCheckErrors( cuptiProfilerCounterDataImageInitializePtr(&gpu_ctl->initializeParams), goto fn_fail );

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = gpu_ctl->initializeParams.pCounterDataImage,
    };
    cuptiCheckErrors( cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr(&scratchBufferSizeParams), goto fn_fail );

    gpu_ctl->counterDataScratchBuffer.size = scratchBufferSizeParams.counterDataScratchBufferSize;
    gpu_ctl->counterDataScratchBuffer.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataScratchBuffer.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataScratchBuffer.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataScratchBuffer.data failed!\n");
        return PAPI_ENOMEM;
    }

    gpu_ctl->initScratchBufferParams = (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params) {
        .structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = gpu_ctl->initializeParams.pCounterDataImage,
        .counterDataScratchBufferSize = gpu_ctl->counterDataScratchBuffer.size,
        .pCounterDataScratchBuffer = gpu_ctl->counterDataScratchBuffer.data,
    };
    cuptiCheckErrors( cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&gpu_ctl->initScratchBufferParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class reset_cupti_prof_config_image
  * @brief Frees and resets variables for config image.. 
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int reset_cupti_prof_config_images(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    papi_free(gpu_ctl->counterDataImagePrefix.data);
    papi_free(gpu_ctl->configImage.data);
    papi_free(gpu_ctl->counterDataImage.data);
    papi_free(gpu_ctl->counterDataScratchBuffer.data);
    papi_free(gpu_ctl->counterAvailabilityImage.data);
    gpu_ctl->counterDataImagePrefix.data = NULL;
    gpu_ctl->configImage.data = NULL;
    gpu_ctl->counterDataImage.data = NULL;
    gpu_ctl->counterDataScratchBuffer.data = NULL;
    gpu_ctl->counterAvailabilityImage.data = NULL;
    gpu_ctl->counterDataImagePrefix.size = 0;
    gpu_ctl->configImage.size = 0;
    gpu_ctl->counterDataImage.size = 0;
    gpu_ctl->counterDataScratchBuffer.size = 0;
    gpu_ctl->counterAvailabilityImage.size = 0;
    return PAPI_OK;
}

/** @class begin_profiling
  * @brief Steps to setup profiling.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int begin_profiling(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    byte_array_t *configImage = &(gpu_ctl->configImage);
    byte_array_t *counterDataScratchBuffer = &(gpu_ctl->counterDataScratchBuffer);
    byte_array_t *counterDataImage = &(gpu_ctl->counterDataImage);

    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        .structSize = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .counterDataImageSize = counterDataImage->size,
        .pCounterDataImage = counterDataImage->data,
        .counterDataScratchBufferSize = counterDataScratchBuffer->size,
        .pCounterDataScratchBuffer = counterDataScratchBuffer->data,
        .range = CUPTI_UserRange,
        .replayMode = CUPTI_UserReplay,
        .maxRangesPerPass = 1,
        .maxLaunchesPerPass = 1,
    };
    cuptiCheckErrors( cuptiProfilerBeginSessionPtr(&beginSessionParams), goto fn_fail );

    CUpti_Profiler_SetConfig_Params setConfigParams = {
        .structSize = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pConfig = configImage->data,
        .configSize = configImage->size,
        .minNestingLevel = 1,
        .numNestingLevels = 1,
        .passIndex = 0,
        .targetNestingLevel = 1,
    };
    cuptiCheckErrors( cuptiProfilerSetConfigPtr(&setConfigParams), goto fn_fail );

    CUpti_Profiler_BeginPass_Params beginPassParams = {
        .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerBeginPassPtr(&beginPassParams), goto fn_fail );

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        .structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerEnableProfilingPtr(&enableProfilingParams), goto fn_fail );

    char rangeName[PAPI_MIN_STR_LEN];
    int gpu_id = gpu_ctl->gpu_id;
    sprintf(rangeName, "PAPI_Range_%d", gpu_id);
    CUpti_Profiler_PushRange_Params pushRangeParams = {
        .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pRangeName = (const char*) &rangeName,
        .rangeNameLength = 100,
    };
    cuptiCheckErrors( cuptiProfilerPushRangePtr(&pushRangeParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class end_profiling
  * @brief Free up the GPI resources acquired for profiling.
  * @param *gpu_ctl
  *   Structure of type cuptip_gpu_state_t which has member variables such as 
  *   gpu_id, rmr, rmr_count, and more.
*/
static int end_profiling(cuptip_gpu_state_t *gpu_ctl)
{

    COMPDBG("EndProfiling. dev = %d\n", gpu_ctl->gpu_id);
    (void) gpu_ctl;

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        .structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerDisableProfilingPtr(&disableProfilingParams), goto fn_fail );

    CUpti_Profiler_PopRange_Params popRangeParams = {
        .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerPopRangePtr(&popRangeParams), goto fn_fail );

    CUpti_Profiler_EndPass_Params endPassParams = {
        .structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerEndPassPtr(&endPassParams), goto fn_fail );

    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        .structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), goto fn_fail );

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        .structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerUnsetConfigPtr(&unsetConfigParams), goto fn_fail );

    CUpti_Profiler_EndSession_Params endSessionParams = {
        .structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    cuptiCheckErrors( cuptiProfilerEndSessionPtr(&endSessionParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}


/** @class get_measured_values
  * @brief Get the counter values for the Cuda native events
  *        added by the user.
  * @param *gpu_ctl
  *   Struct that holds member variables such as gpu id, rmr, etc.
  * @param *counts
  *   Array to hold the counter values for the associated Cuda native
  *   events. 
*/
static int get_measured_values(cuptip_gpu_state_t *gpu_ctl, long long *counts)
{
    COMPDBG("eval_metric_values. dev = %d\n", gpu_ctl->gpu_id);
    int i, papi_errno = PAPI_OK;
    int numMetrics = gpu_ctl->added_events->count;
    double *gpuValues;
    char **metricNames;

    if (!gpu_ctl->counterDataImage.size) {
        ERRDBG("Counter Data Image is empty!\n");
        return PAPI_EINVAL;
    }

    /* allocate memory */
    gpuValues = (double*) papi_malloc(numMetrics * sizeof(double));
    if (gpuValues == NULL) {
        ERRDBG("malloc gpuValues failed.\n");
        return PAPI_ENOMEM;
    }   

    /* allocate memory */
    metricNames = (char**) papi_calloc(numMetrics, sizeof(char *)); 
    if (metricNames == NULL) {
        ERRDBG("Failed to allocate memory for metricNames.\n");
        return PAPI_ENOMEM;
    }    

    for (i = 0; i < numMetrics; i++) {
        metricNames[i] = gpu_ctl->added_events->cuda_evts[i];
        LOGDBG("Setting metric name %s\n", metricNames[i]);
    }

    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        .structSize = NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
        .pCounterDataImage = gpu_ctl->counterDataImage.data,
        .rangeIndex = 0,
        .isolated = 1,
    };

    nvpwCheckErrors( NVPW_MetricsContext_SetCounterDataPtr(&setCounterDataParams), goto fn_fail );

    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
        .structSize = NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
        .numMetrics = numMetrics,
        .ppMetricNames = (const char* const*) metricNames,
        .pMetricValues = gpuValues,
    };

    nvpwCheckErrors( NVPW_MetricsContext_EvaluateToGpuValuesPtr(&evalToGpuParams), goto fn_fail );

    /* store the gpu values */
    for (i = 0; i < (int) gpu_ctl->added_events->count; i++) {
        counts[i] = gpuValues[i];
    }

    /* free memory allocations */
    papi_free(metricNames);
    papi_free(gpuValues);

fn_exit:
    return papi_errno;
fn_fail:
    return PAPI_EMISC;
}

/** @class find_same_chipname
  * @brief Check to see if chipnames are identical.
  * 
  * @param gpu_id
  *   A gpu id number, e.g 0, 1, 2, etc.
*/
static int find_same_chipname(int gpu_id)
{
    int i;
    for (i = 0; i < gpu_id; i++) {
        if (!strcmp(cuptiu_table_p->avail_gpu_info[gpu_id].chip_name, cuptiu_table_p->avail_gpu_info[i].chip_name)) {
            return i;
        }
    }
    return -1;
}

/** @class init_all_metrics
  * @brief Initialize metrics for a specific GPU.
  *        
*/
static int init_all_metrics(void)
{
    int gpu_id, papi_errno = PAPI_OK;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        papi_errno = get_chip_name(gpu_id, cuptiu_table_p->avail_gpu_info[gpu_id].chip_name);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
    int found;
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams = cuptiu_table_p->avail_gpu_info[found].pmetricsContextCreateParams;
            continue;
        }
        MCCP_t *pMCCP = (MCCP_t *) papi_calloc(1, sizeof(MCCP_t));
        if (pMCCP == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        pMCCP->structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE;
        pMCCP->pChipName = cuptiu_table_p->avail_gpu_info[gpu_id].chip_name;
        nvpwCheckErrors( NVPW_CUDA_MetricsContext_CreatePtr(pMCCP), goto fn_fail );

        cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams = pMCCP;
    }

fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class free_all_enumerated_metrics
  * @brief Free's all enumerated metrics for each gpu on the system.  
*/
static void free_all_enumerated_metrics(void)
{
    COMPDBG("Entering.\n");
    int gpu_id, found;
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams;
    if (cuptiu_table_p->avail_gpu_info == NULL) {
        return;
    }
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            cuptiu_table_p->avail_gpu_info[gpu_id].num_metrics = 0;
            cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams = NULL;
            continue;
        }
        if (cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams->pMetricsContext) {
            metricsContextDestroyParams = (NVPW_MetricsContext_Destroy_Params) {
                .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pMetricsContext = cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams->pMetricsContext,
            };
            nvpwCheckErrors(NVPW_MetricsContext_DestroyPtr(&metricsContextDestroyParams), );
        }
        papi_free(cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams);
        cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams = NULL;

    }
    papi_free(cuptiu_table_p->avail_gpu_info);
    cuptiu_table_p->avail_gpu_info = NULL;
}

/** @class init_main_htable
 *  @brief Initialize the main htable used to collect metrics.
*/
static int init_main_htable(void)
{
    int i, val = 1, base = 2, papi_errno = PAPI_OK;

    /* allocate (2 ^ NAMEID_WIDTH) metric names, this matches the 
       number of bits for the event encoding format */
    for (i = 0; i < NAMEID_WIDTH; i++) {
        val *= base;
    }    
   
    /* initialize struct */ 
    cuptiu_table_p = (cuptiu_event_table_t *) papi_malloc(sizeof(cuptiu_event_table_t));
    if (cuptiu_table_p == NULL) {
        goto fn_fail;
    }
    cuptiu_table_p->capacity = val; 
    cuptiu_table_p->count = 0;

    cuptiu_table_p->events = (cuptiu_event_t *) papi_calloc(val, sizeof(cuptiu_event_t));
    if (cuptiu_table_p->events == NULL) {
        goto fn_fail;
    }

    cuptiu_table_p->avail_gpu_info = (gpu_record_t *) papi_calloc(num_gpus, sizeof(gpu_record_t));
    if (cuptiu_table_p->avail_gpu_info == NULL) {
        goto fn_fail;
    }

    /* initialize the main hash table for metric collection */ 
    htable_init(&cuptiu_table_p->htable);

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOMEM;
    goto fn_exit;
}

/** @class cuptip_init
  * @brief Load and initialize API's.  
*/
int cuptip_init(void)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;

    papi_errno = load_cupti_perf_sym();
    papi_errno += load_nvpw_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to load CUDA library functions.");
        goto fn_fail;
    }

    /* collect number of gpu's on the system */
    papi_errno = cuptic_device_get_count(&num_gpus);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    if (num_gpus <= 0) {
        cuptic_disabled_reason_set("No GPUs found on system.");
        goto fn_fail;
    }
   
    /* initialize cupti profiler and perfworks api */
    papi_errno = initialize_cupti_profiler_api();
    papi_errno += initialize_perfworks_api();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to initialize CUPTI profiler libraries.");
        goto fn_fail;
    }

    papi_errno = init_main_htable();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = init_all_metrics();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    /* collect metrics */
    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuInitPtr(0);
    if (papi_errno != CUDA_SUCCESS) {
        cuptic_disabled_reason_set("Failed to initialize CUDA driver API.");
        goto fn_fail;
    }

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

/** @class verify_events
  * @brief Verify user added events and store metadata i.e. metric names 
  *        and device id's .
  * @param *events_id
  *   Cuda native event id's.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
*/
int verify_events(uint64_t *events_id, int num_events, 
                  cuptip_control_t state) 
{
    int papi_errno, i;
    char *metricName;
    int idx;

    for (i = 0; i < num_gpus; i++) {
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
 
        /* for a specific device table, get the current event index */
        idx = state->gpu_ctl[info.device].added_events->count; 

        metricName = state->gpu_ctl[info.device].added_events->cuda_evts[idx];
        snprintf(metricName, PAPI_MAX_STR_LEN, "%s", cuptiu_table_p->events[info.nameid].name);

        void *p;
        if (htable_find(cuptiu_table_p->htable, metricName, (void **) &p) != HTABLE_SUCCESS) {
            return PAPI_ENOEVNT;
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
int cuptip_ctx_create(cuptic_info_t thr_info, cuptip_control_t *pstate, uint64_t *events_id, int num_events)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK, gpu_id, i;
    long long *counters = NULL;
    char name[PAPI_2MAX_STR_LEN] = { 0 };

    cuptip_control_t state = (cuptip_control_t) papi_calloc (1, sizeof(struct cuptip_control_s));
    if (state == NULL) {
        return PAPI_ENOMEM;
    }

    state->gpu_ctl = (cuptip_gpu_state_t *) papi_calloc(num_gpus, sizeof(cuptip_gpu_state_t));
    if (state->gpu_ctl == NULL) {
        return PAPI_ENOMEM;
    }

    counters = papi_malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        return PAPI_ENOMEM;
    }

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        state->gpu_ctl[gpu_id].gpu_id = gpu_id;
    }

    event_info_t info;
    papi_errno = evt_id_to_info(events_id[num_events - 1], &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    } 

    /* register the user created cuda context for the current gpu if not already known */
    papi_errno = cuptic_ctxarr_update_current(thr_info, info.device);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /* create a MetricsContext */
    papi_errno = nvpw_cuda_metricscontext_create(state);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /* verify user added events are available on the machine */
    papi_errno = verify_events(events_id, num_events, state);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /* check to make sure added events do not require multiple passes */
    papi_errno = check_multipass(state);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    state->info = thr_info;
    state->counters = counters;

fn_exit:
    *pstate = state;
    return papi_errno;
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
    int gpu_id, papi_errno = PAPI_OK;
    /* create instance of cuptip_gpu_state_t */
    cuptip_gpu_state_t *gpu_ctl;
    /* create a context handle */
    CUcontext userCtx, ctx;

    // return the Cuda context bound to the calling CPU thread
    cudaCheckErrors( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );

    /* enumerate through all of the unique gpus */
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }
        LOGDBG("Device num %d: event_count %d, rmr count %d\n", gpu_id, gpu_ctl->added_events->count, gpu_ctl->rmr_count);
        papi_errno = cuptic_device_acquire(state->gpu_ctl[gpu_id].added_events);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Profiling same gpu from multiple event sets not allowed.\n");
            return papi_errno;
        }
        /* get the cuda context */
        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        /* bind the specified CUDA context to the calling CPU thread */
        cudaCheckErrors( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );

        /*  query/filter cuda native events available on host */
        papi_errno = get_counter_availability(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Error getting counter availability image.\n");
            return papi_errno;
        }

        /* CUPTI profiler host configuration */
        papi_errno = metric_get_config_image(gpu_ctl);
        papi_errno += metric_get_counter_data_prefix_image(gpu_ctl);
        papi_errno += create_counter_data_image(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Failed to create CUPTI profiler state for gpu %d\n", gpu_id);
            goto fn_fail;
        }

        papi_errno = begin_profiling(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Failed to start profiling for gpu %d\n", gpu_id);
            goto fn_fail;
        }
    }

fn_exit:
    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), goto fn_fail_misc );
    return papi_errno;
fn_fail:
    papi_errno = PAPI_ECMP;
    goto fn_exit;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class cuptip_ctx_read
  * @brief Code to read Cuda hardware counters from an event set.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t.
  * @param **counters
  *   Array that holds the counter values for the specificed Cuda native events 
  *   added by a user.  
*/
int cuptip_ctx_read(cuptip_control_t state, long long **counters)
{
    COMPDBG("Entering.\n");
    int papi_errno, gpu_id, i, j = 0, method, evt_pos;
    long long counts[30], *counter_vals = state->counters;
    cuptip_gpu_state_t *gpu_ctl = NULL;
    CUcontext userCtx = NULL, ctx = NULL;

    cudaCheckErrors( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }

        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        if (papi_errno != PAPI_OK) {
            goto fn_fail_misc;

        }

        cudaCheckErrors( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );

        CUpti_Profiler_PopRange_Params popRangeParams = {
            .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        cuptiCheckErrors( cuptiProfilerPopRangePtr(&popRangeParams), goto fn_fail_misc );

        CUpti_Profiler_EndPass_Params endPassParams = {
            .structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        cuptiCheckErrors( cuptiProfilerEndPassPtr(&endPassParams), goto fn_fail_misc );

        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
            .structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
       
        cuptiCheckErrors( cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), goto fn_fail_misc );

        papi_errno = get_measured_values(gpu_ctl, counts);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }

        for (i = 0; i < gpu_ctl->added_events->count; i++) {
            evt_pos = gpu_ctl->added_events->evt_pos[i];
            if (state->read_count == 0) {
                counter_vals[evt_pos] = counts[i];
            }
            else {
                /* determine collection method such as max, min, sum, and avg for an added Cuda native event */
                method = get_event_collection_method(gpu_ctl->added_events->cuda_evts[i]);
                switch (method) {
                    case CUDA_SUM:
                        counter_vals[evt_pos] += counts[i];
                        break;
                    case CUDA_MIN:
                        counter_vals[evt_pos] = counter_vals[evt_pos] < counts[i] ? counter_vals[evt_pos] : counts[i];
                        break;
                    case CUDA_MAX:
                        counter_vals[evt_pos] = counter_vals[evt_pos] > counts[i] ? counter_vals[evt_pos] : counts[i];
                        break;
                    case CUDA_AVG:
                         /* (size * average + value) / (size + 1) 
                            size - current number of values in the average
                            average - current average
                            value - number to add to the average
                         */
                         counter_vals[evt_pos] = (state->read_count * counter_vals[j++] + counts[i]) / (state->read_count + 1);
                         break;
                    default:
                        counter_vals[evt_pos] = counts[i];
                        break;
                }
            }
        }
        *counters = counter_vals;

        cuptiCheckErrors( cuptiProfilerCounterDataImageInitializePtr(&gpu_ctl->initializeParams), goto fn_fail_misc );
        cuptiCheckErrors( cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&gpu_ctl->initScratchBufferParams), goto fn_fail_misc );

        CUpti_Profiler_BeginPass_Params beginPassParams = {
            .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        cuptiCheckErrors( cuptiProfilerBeginPassPtr(&beginPassParams), goto fn_fail_misc );

        char rangeName[PAPI_MIN_STR_LEN];
        sprintf(rangeName, "PAPI_Range_%d", gpu_ctl->gpu_id);
        CUpti_Profiler_PushRange_Params pushRangeParams = {
            .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
            .pRangeName = (const char*) &rangeName,
            .rangeNameLength = 100,
        };
        cuptiCheckErrors( cuptiProfilerPushRangePtr(&pushRangeParams), goto fn_fail_misc );

    }
    state->read_count++;
fn_exit:
    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), );
    return papi_errno;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class cuptip_ctx_reset
  * @brief Code to reset Cuda hardware counter values.
  * @param *counters
  *   Array that holds the counter values for the specificed Cuda native events
  *   added by a user. 
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
    int gpu_id;
    int papi_errno = PAPI_OK;
    cuptip_gpu_state_t *gpu_ctl;
    CUcontext userCtx = NULL, ctx = NULL;

    cudaCheckErrors( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );

    for (gpu_id=0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->added_events->count == 0) {
            continue;
        }
        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        cudaCheckErrors( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );
        papi_errno = end_profiling(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
        papi_errno = cuptic_device_release(state->gpu_ctl[gpu_id].added_events);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

fn_exit:
    cudaCheckErrors( cuCtxSetCurrentPtr(userCtx), goto fn_fail_misc );
    return papi_errno;
fn_fail:
    goto fn_exit;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

/** @class cuptip_ctx_destroy
  * @brief Destroy created profiling context.
  * @param *pstate
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t.
*/
int cuptip_ctx_destroy(cuptip_control_t *pstate)
{
    COMPDBG("Entering.\n");
    cuptip_control_t state = *pstate;
    int i, j;
    int papi_errno = nvpw_cuda_metricscontext_destroy(state);
    for (i = 0; i < num_gpus; i++) {
        reset_cupti_prof_config_images( &(state->gpu_ctl[i]) );
        cuptiu_event_table_destroy( &(state->gpu_ctl[i].added_events) );
        for (j = 0; j < state->gpu_ctl[i].rmr_count; j++) {
            papi_free((void *) state->gpu_ctl[i].rmr[j].pMetricName);
        }
        papi_free(state->gpu_ctl[i].rmr);
    }
    papi_free(state->gpu_ctl);
    papi_free(state);
    *pstate = NULL;
    return papi_errno;
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
    shutdown_event_table();
    free_all_enumerated_metrics();
    deinitialize_cupti_profiler_api();
    unload_nvpw_sym();
    unload_cupti_perf_sym();
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
int evt_id_create(event_info_t *info, uint64_t *event_id)
{
    *event_id  = (uint64_t)(info->device   << DEVICE_SHIFT);
    *event_id |= (uint64_t)(info->flags    << QLMASK_SHIFT);
    *event_id |= (uint64_t)(info->nameid   << NAMEID_SHIFT);
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
int evt_id_to_info(uint64_t event_id, event_info_t *info)
{
    info->device   = (int)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->flags    = (int)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid   = (int)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->device >= num_gpus) {
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
  * @brief Initialize hash table and cuptiu_event_table_t structure.
*/
int init_event_table(void) 
{
    int i, dev_id, found, table_idx = 0, papi_errno = PAPI_OK;
    int listsubmetrics = 1;

    /* instatiate struct to collect the total metric count and metric names;
       instantiated here to avoid scoping issues */
    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
    
    /* loop through all available devices on the current system */
    for (dev_id = 0; dev_id < num_gpus; dev_id++) {
        found = find_same_chipname(dev_id);
        /* unique device found, collect metadata  */
        if (found == -1) {
            /* increment table index */
            if (dev_id > 0)
                table_idx++;

            /* assigning values to member variables */
            getMetricNameBeginParams.pPriv = NULL;
            getMetricNameBeginParams.pMetricsContext = cuptiu_table_p->avail_gpu_info[table_idx].pmetricsContextCreateParams->pMetricsContext;
            getMetricNameBeginParams.hidePeakSubMetrics = !listsubmetrics;
            getMetricNameBeginParams.hidePerCycleSubMetrics = !listsubmetrics;
            getMetricNameBeginParams.hidePctOfPeakSubMetrics = !listsubmetrics;

            nvpwCheckErrors( NVPW_MetricsContext_GetMetricNames_BeginPtr(&getMetricNameBeginParams), goto fn_fail ); 

            /* for each unique device found, store both the total number of metrics and metric names */
            cuptiu_table_p->avail_gpu_info[table_idx].num_metrics = getMetricNameBeginParams.numMetrics;
            cuptiu_table_p->avail_gpu_info[table_idx].metric_names = getMetricNameBeginParams.ppMetricNames;
        }
        /* device metadata already collected, set table index */
        else {
            /* set table_idx to */
            table_idx = found;
        }

        /* loop through metrics to add to overall event table */
        for (i = 0; i < cuptiu_table_p->avail_gpu_info[table_idx].num_metrics; i++) {
            papi_errno = get_ntv_events( cuptiu_table_p, cuptiu_table_p->avail_gpu_info[table_idx].metric_names[i], dev_id);
            if (papi_errno != PAPI_OK)
                goto fn_exit;
        }

    }

    /* free memory */
    for (i = 0; i < table_idx; i++) {
        NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = {
            .structSize = NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = cuptiu_table_p->avail_gpu_info[table_idx].pmetricsContextCreateParams->pMetricsContext,
        };
        nvpwCheckErrors( NVPW_MetricsContext_GetMetricNames_EndPtr((NVPW_MetricsContext_GetMetricNames_End_Params *) &getMetricNameEndParams), goto fn_fail );
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC; 
    goto fn_exit;

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
static int get_ntv_events(cuptiu_event_table_t *evt_table, const char *evt_name, int gpu_id) 
{
    int *count = &evt_table->count;
    cuptiu_event_t *events = evt_table->events;

    /* check to see if evt_name argument has been provided */
    if (evt_name == NULL) {
        return PAPI_EINVAL;
    }

    /* check to see if capacity has been correctly allocated */
    if (*count >= evt_table->capacity) {
        return PAPI_EBUG;
    }

    cuptiu_event_t *event;
    /* check to make sure event entry has not already been added */
    if ( htable_find(evt_table->htable, evt_name, (void **) &event) != HTABLE_SUCCESS ) {
        event = &events[*count];
        /* increment count */
        (*count)++;

        /* store event info */
        strcpy(event->name, evt_name);

        /* insert event info into htable */
        if ( htable_insert(evt_table->htable, evt_name, event) != HTABLE_SUCCESS ) {
            return PAPI_ESYS;
        }
    }

    cuptiu_dev_set(&event->device_map, gpu_id);

    return PAPI_OK;
}

/** @class shutdown_event_table
  * @brief Shutdown cuptiu_event_table_t structure that holds the cuda native 
  *        event name and the corresponding description.
*/
static int shutdown_event_table(void)
{
    cuptiu_table_p->count = 0;

    papi_free(cuptiu_table_p->events);

    return PAPI_OK;
}

/** @class retrieve_metric_descr
  * @brief Collect the description for the provided evt_name.
  *
  * @param *pMetricsContext
  *   Structure providing context for evt_name. 
  * @param *evt_name
  *   Cuda native event name.
  * @param *description
  *   Corresponding description for provided Cuda native event name.
  * @param gpu_id
  *   Device number, e.g. 0, 1, 2, ... ,etc.
*/
static int retrieve_metric_descr( NVPA_MetricsContext *pMetricsContext, const char *evt_name, char *description, const char *chip_name) 
{
    COMPDBG("Entering.\n");
    int num_dep, i, len, passes, papi_errno;
    const char *token_sw_evt = "sass";
    char desc[PAPI_2MAX_STR_LEN];
    NVPA_RawMetricRequest *rmr;
    NVPA_Status nvpa_err;

    /* check to make sure an argument has been passed for evt_name and description */
    if (evt_name == NULL || description == NULL) {
        return PAPI_EINVAL;
    }

    /* perfworks api: instantiate a new struct with provided event name to be passed to
       NVPW_MetricsContext_GetMetricsProperties_BeginPtr */
    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
        // [in]
        .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .pMetricsContext = pMetricsContext,
        .pMetricName = evt_name,
    };
    nvpa_err = NVPW_MetricsContext_GetMetricProperties_BeginPtr(&getMetricPropertiesBeginParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS || getMetricPropertiesBeginParams.ppRawMetricDependencies == NULL) {
        strcpy(description, "Could not get description.");
        return PAPI_EINVAL;
    }

    for (num_dep = 0; getMetricPropertiesBeginParams.ppRawMetricDependencies[num_dep] != NULL; num_dep++) {;}

    rmr = (NVPA_RawMetricRequest *) papi_calloc(num_dep, sizeof(NVPA_RawMetricRequest));
    if (rmr == NULL) {
        return PAPI_ENOMEM;
    }
    for (i = 0; i < num_dep; i++) {
        /* list of */
        rmr[i].pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[i]);
        rmr[i].isolated = 1;
        rmr[i].keepInstances = 1;
        rmr[i].structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    }
    
    /* collect the corresponding description for the provided evt_name */
    len = snprintf( desc, PAPI_2MAX_STR_LEN, "%s. Units=(%s)",
                    getMetricPropertiesBeginParams.pDescription,
                    getMetricPropertiesBeginParams.pDimUnits);
    /* check to make sure that description length is not greater than 
       PAPI_2MAX_STR_LEN, which holds */
    if (len > PAPI_2MAX_STR_LEN) {
        ERRDBG("String formatting exceeded max string length.\n");
        return PAPI_ENOMEM;
    }

    /* perfworks api: instantiate a new struct to be passsed to NVPW_MetricsContext_GetMetricProperties_EndPtr */
    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
        // [in]
        .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
        .pPriv = NULL, //assign to NULL
        .pMetricsContext = pMetricsContext,
    };
    nvpa_err = NVPW_MetricsContext_GetMetricProperties_EndPtr(&getMetricPropertiesEndParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }
    /* perfworks api: instantiate a new stuct to be passed to NVPW_CUDA_RawMetricsConfig_CreatePtr */
    NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
        // [in]
        .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
        .pChipName = chip_name,
    };
    nvpa_err = NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    /* collects the total number of passes
       num_passes = numPipelinedPasses + numIsolatedPasses * numNestingLevels */
    papi_errno = calculate_num_passes( nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                                       num_dep, rmr, &passes );
    if ( papi_errno == PAPI_EMULPASS ) {
        /* at this point we just want the number of passes (stored in passes) */
    }

    /* perfworks api: instantiate a new struct to be passed to NVPW_RawMetricsConfig_DestroyPtr */
    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        // [in]
        .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL, // assign to NULL
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    nvpa_err = NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS) {
        return PAPI_EMISC;
    }

    /* add extra metadata to description */
    snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " Numpass=%d", passes);
    if (passes > 1) {
        snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " (multi-pass not supported)");
    }

    if (strstr(evt_name, token_sw_evt) != NULL) {
        snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " (SW event)");
    }

    /* free memory, copy description, and return successful error code */
    papi_free(rmr);

    strcpy(description, desc);

    return PAPI_OK;
}

/** @class retrieve_metric_rmr
  * @brief Collect the raw metric request for the provided evt_name.
  *
  * @param *pMetricsContext
  *   Structure providing context for evt_name. 
  * @param *evt_name
  *   Cuda native event name.
  * @param *numDep
  *   Number of dependencies for a cuda native event.
  * @param **pRMR
  *  Raw metric requests for a cuda native event.
*/
static int retrieve_metric_rmr( NVPA_MetricsContext *pMetricsContext, const char *evt_name,
                                int *numDep, NVPA_RawMetricRequest **pRMR )
{
    COMPDBG("Entering.\n");
    int num_dep, i;
    NVPA_Status nvpa_err;
    NVPA_RawMetricRequest *rmr;

    /* check to make sure an argument has been passed for evt_name */
    if ( evt_name == NULL ) {
        return PAPI_EINVAL;
    }

    /* instantiate a new metric properties structure with the provided evt_name */
    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
        .pMetricName = evt_name,
    };

    /* collect metric properties such as dependencies and description for the 
       structure created by the passed evt_name */
    nvpa_err = NVPW_MetricsContext_GetMetricProperties_BeginPtr(&getMetricPropertiesBeginParams);
    if (nvpa_err != NVPA_STATUS_SUCCESS || getMetricPropertiesBeginParams.ppRawMetricDependencies == NULL) {
        return PAPI_EINVAL;
    }

    for (num_dep = 0; getMetricPropertiesBeginParams.ppRawMetricDependencies[num_dep] != NULL; num_dep++) {;}

    rmr = (NVPA_RawMetricRequest *) papi_calloc(num_dep, sizeof(NVPA_RawMetricRequest));
    if (rmr == NULL) {
        return PAPI_ENOMEM;
    }

    for (i = 0; i < num_dep; i++) {
        rmr[i].pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[i]);
        rmr[i].isolated = 1;
        rmr[i].keepInstances = 1;
        rmr[i].structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    }

    /* store number of dependencies and raw metric requests */
    *numDep = num_dep;
    *pRMR = rmr;
    
    /* ending/deleting instantiated struct created by passed evt_name */
    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
    };

    /* ending pointer created by passed evt_name */
    nvpwCheckErrors( NVPW_MetricsContext_GetMetricProperties_EndPtr(&getMetricPropertiesEndParams), return PAPI_EMISC );

    return PAPI_OK;
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
int cuptip_evt_enum(uint64_t *event_code, int modifier)
{
    int papi_errno = PAPI_OK;
    event_info_t info;
    SUBDBG("ENTER: event_code: %lu, modifier: %d\n", *event_code, modifier);

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if(cuptiu_table_p->count == 0) {
                papi_errno = PAPI_ENOEVNT;
                break;
            }
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
                info.device = 0;
                info.flags = 0;
                info.nameid++;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_END;
            break;
        case PAPI_NTV_ENUM_UMASKS:
            papi_errno = evt_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }
            if (info.flags == 0){
                info.device = 0;
                info.flags = DEVICE_FLAG;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_END;
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
int cuptip_evt_code_to_descr(uint64_t event_code, char *descr, int len) 
{
    int papi_errno, str_len;
    event_info_t info;
    papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }    

    str_len = snprintf(descr, (size_t) len, "%s", cuptiu_table_p->events[event_code].desc);
    if (str_len > len) {
        ERRDBG("String formatting exceeded max string length.\n");
        return PAPI_ENOMEM;  
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
int cuptip_evt_name_to_code(const char *name, uint64_t *event_code)
{
    int htable_errno, device, flags, nameid, papi_errno = PAPI_OK;
    cuptiu_event_t *event;
    char base[PAPI_MAX_STR_LEN] = { 0 };
    SUBDBG("ENTER: name: %s, event_code: %p\n", name, event_code);

    papi_errno = evt_name_to_device(name, &device);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = evt_name_to_basename(name, base, PAPI_MAX_STR_LEN);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }

    /* flags = DEVICE_FLAG will need to be updated if more qualifiers are added,
       see implemtation in rocm (roc_profiler.c) */
    flags = (device >= 0) ? DEVICE_FLAG:0;
    if (flags == 0){
        papi_errno = PAPI_EINVAL;
        goto fn_exit;
    }

    nameid = (int) (event - cuptiu_table_p->events);
    event_info_t info = { device, flags, nameid };
    papi_errno = evt_id_create(&info, event_code);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = evt_id_to_info(*event_code, &info);

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
int cuptip_evt_code_to_name(uint64_t event_code, char *name, int len)
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
static int evt_code_to_name(uint64_t event_code, char *name, int len)
{
    int papi_errno, str_len;

    event_info_t info;
    papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch (info.flags) {
        case (DEVICE_FLAG):
            str_len = snprintf(name, len, "%s:device=%i", cuptiu_table_p->events[info.nameid].name, info.device);
            if (str_len > len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_ENOMEM;
            }
            break;
        default:
            str_len = snprintf(name, len, "%s", cuptiu_table_p->events[info.nameid].name);
            if (str_len > len) {
                ERRDBG("String formatting exceeded max string length.\n");
                return PAPI_ENOMEM;
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
int cuptip_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info)
{
    int papi_errno, i, gpu_id;
    char description[PAPI_HUGE_STR_LEN];

    /* get the events nameid and flags */
    event_info_t inf;
    papi_errno = evt_id_to_info(event_code, &inf);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    /* collect the description and calculated numpass for the Cuda event  */
    if (cuptiu_table_p->events[inf.nameid].desc[0] == '\0') {
        /* find a matching device id to get correct MetricsContext and chip name */
        for (i = 0; i < num_gpus; ++i) {
            if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                gpu_id = i;
                break;
            }
        }
        papi_errno = retrieve_metric_descr( cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams->pMetricsContext,
                                            cuptiu_table_p->events[inf.nameid].name, cuptiu_table_p->events[inf.nameid].desc,
                                            cuptiu_table_p->avail_gpu_info[gpu_id].pmetricsContextCreateParams->pChipName );
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    switch (inf.flags) {
        case (0):
            /* store details for the Cuda event */ 
            snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].name );
            snprintf( info->short_descr, PAPI_MIN_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            break;
        case DEVICE_FLAG:
        {
            int init_metric_dev_id;
            char devices[PAPI_MAX_STR_LEN] = { 0 };
            for (i = 0; i < num_gpus; ++i) {
                if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                    /* for an event, store the first device found to use with :device=#, 
                       as on a heterogenous system events may not appear on each device */
                    if (devices[0] == '\0') {
                        init_metric_dev_id = i;
                    }

                    sprintf(devices + strlen(devices), "%i,", i);
                }
            }
            *(devices + strlen(devices) - 1) = 0;

            /* store details for the Cuda event */
            snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:device=%i", cuptiu_table_p->events[inf.nameid].name, init_metric_dev_id );
            snprintf( info->short_descr, PAPI_MIN_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                     cuptiu_table_p->events[inf.nameid].desc, devices );
            snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                      cuptiu_table_p->events[inf.nameid].desc, devices );
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
        cuptiu_event_t *event;

        htable_errno = htable_find(cuptiu_table_p->htable, name, (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_EINVAL;
        }

        // Search for the first device the event exists for.
        for (i = 0; i < num_gpus; ++i) {
            if (cuptiu_dev_check(event->device_map, i)) {
                *device = i;
                break;
            }
        }
    }
    return PAPI_OK;
}
