/**
 * @file    cupti_events.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include <papi.h>
#include "cupti_events.h"
#include "papi_cupti_common.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
/* Functions needed by CUPTI Events API */
/* ... */

/* CUPTI Events component API functions */

int cuptie_init(void)
{
    cuptic_disabled_reason_set("CUDA events API not implemented.");
    return PAPI_ENOIMPL;
}

int cuptie_ctx_create(void *thr_info, cuptie_control_t *pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_start(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_read(cuptie_control_t ctl, long long **values)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_stop(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_reset(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_destroy(cuptie_control_t *pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_enum(uint64_t *event_code, int modifier)
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_code_to_descr(uint64_t event_code, char *descr, int len) 
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_name_to_code(const char *name, uint64_t *event_code)
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_code_to_name(uint64_t event_code, char *name, int len)
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info) 
{
    return PAPI_ENOIMPL;
}

int cuptie_shutdown(void)
{
    return PAPI_ENOIMPL;
}
