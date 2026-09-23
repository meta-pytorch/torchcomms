/**
 * @file    cupti_dispatch.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include "cupti_config.h"
#include "papi_cupti_common.h"
#include "cupti_event_and_metric.h"
#include "cupti_dispatch.h"
#include "lcuda_debug.h"

#if defined(API_PERFWORKS)
#   include "cupti_profiler.h"
#endif

#if defined(API_LEGACY)
#   include "cupti_events.h"
#endif

int cuptid_shutdown(void)
{
    int papi_errno;
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        papi_errno = cuptip_shutdown();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        papi_errno = cuptie_shutdown();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
#endif

    }

    return cuptic_shutdown();
}

int cuptid_err_get_last(const char **error_str)
{
    return cuptic_err_get_last(error_str);
}

int cuptid_get_chip_name(int dev_num, char *name)
{
    return get_chip_name(dev_num, name);
}

int cuptid_device_get_count(int *num_gpus)
{
    return cuptic_device_get_count(num_gpus);
}

int cuptid_init(void)
{
    int papi_errno;
    int init_errno = cuptic_init();
    if (init_errno != PAPI_OK && init_errno != PAPI_PARTIAL) {
        papi_errno = init_errno;
        goto fn_exit;
    }

    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        papi_errno = cuptip_init();
        if (papi_errno == PAPI_OK) {
            if (init_errno == PAPI_PARTIAL) {
                papi_errno = init_errno;
            }
        }
#else
        cuptic_err_set_last("PAPI not built with NVIDIA profiler API support.");
        papi_errno = PAPI_ECMP;
        goto fn_exit;
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        papi_errno = cuptie_init();
        if (papi_errno == PAPI_OK) {
            if (init_errno == PAPI_PARTIAL) {
                papi_errno = init_errno;
            }
        }
#else
        cuptic_err_set_last("Unknown Event and Metric API issue.");
        papi_errno = PAPI_ECMP;
        goto fn_exit;
#endif

    } else {
        cuptic_err_set_last("CUDA configuration not supported.");
        papi_errno = PAPI_ECMP;
    }
fn_exit:
    return papi_errno;
}

int cuptid_thread_info_create(cuptid_info_t *info)
{
    return cuptic_ctxarr_create((cuptic_info_t *) info);
}

int cuptid_thread_info_destroy(cuptid_info_t *info)
{
    return cuptic_ctxarr_destroy((cuptic_info_t *) info);
}

int cuptid_ctx_create(cuptid_info_t info,  cuptip_control_t *pcupti_ctl, uint32_t *events_id, int num_events)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_create((cuptic_info_t) info, pcupti_ctl, events_id, num_events);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined (API_LEGACY)
        return cuptie_ctx_create((cuptic_info_t) info, (cuptie_control_t *) pcupti_ctl, events_id, num_events);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_start(cuptip_control_t cupti_ctl)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_start(cupti_ctl);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_ctx_start((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_read(cuptip_control_t cupti_ctl, long long **counters)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_read(cupti_ctl, counters);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_ctx_read((cuptie_control_t) cupti_ctl, counters);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_reset(cuptip_control_t cupti_ctl)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_reset(cupti_ctl);
#endif
    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_ctx_reset((cuptie_control_t) cupti_ctl);
#endif
    }
    return PAPI_ECMP;
}

int cuptid_ctx_stop(cuptip_control_t cupti_ctl)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_stop(cupti_ctl);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_ctx_stop((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_destroy(cuptip_control_t *pcupti_ctl)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_destroy(pcupti_ctl);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_ctx_destroy((cuptie_control_t *) pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_enum(uint32_t *event_code, int modifier)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_evt_enum(event_code, modifier);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_evt_enum(event_code, modifier);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_code_to_descr(uint32_t event_code, char *descr, int len)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_evt_code_to_descr(event_code, descr, len);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_evt_code_to_descr(event_code, descr, len);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_name_to_code(const char *name, uint32_t *event_code)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_evt_name_to_code(name, event_code);
#endif

    } else if (cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_evt_name_to_code(name, event_code);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_code_to_name(uint32_t event_code, char *name, int len)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_evt_code_to_name(event_code, name, len);
#endif

    } else if(cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_evt_code_to_name(event_code, name, len);
#endif

    }   
    return PAPI_ECMP;
}

int cuptid_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info)
{
    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api == API_PERFWORKS) {

#if defined(API_PERFWORKS)
        return cuptip_evt_code_to_info(event_code, info);
#endif

    } else if(cupti_api == API_LEGACY) {

#if defined(API_LEGACY)
        return cuptie_evt_code_to_info(event_code, info);
#endif

    }
    return PAPI_ECMP;
}
