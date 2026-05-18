/**
 * @file    rocp_sdk.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief This implements a PAPI component that accesses hardware
 *  monitoring counters for AMD GPU and APU devices through the
 *  ROCprofiler-SDK library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <stdio.h>
#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "sdk_class.h"

#define ROCPROF_SDK_MAX_COUNTERS (64)
#define RPSDK_CTX_RUNNING (1)

unsigned int _rocp_sdk_lock;

/* Init and finalize */
static int rocp_sdk_init_component(int cid);
static int rocp_sdk_init_thread(hwd_context_t *ctx);
static int rocp_sdk_init_control_state(hwd_control_state_t *ctl);
static int rocp_sdk_init_private(void);
static int rocp_sdk_shutdown_component(void);
static int rocp_sdk_shutdown_thread(hwd_context_t *ctx);
static int rocp_sdk_cleanup_eventset(hwd_control_state_t *ctl);

/* Set and update component state */
static int rocp_sdk_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx);

/* Start and stop profiling of hardware events */
static int rocp_sdk_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocp_sdk_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int rocp_sdk_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocp_sdk_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);

/* Event conversion */
static int rocp_sdk_ntv_enum_events(unsigned int *event_code, int modifier);
static int rocp_sdk_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int rocp_sdk_ntv_name_to_code(const char *name, unsigned int *event_code);
static int rocp_sdk_ntv_code_to_descr(unsigned int event_code, char *descr, int len);
static int rocp_sdk_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

static int rocp_sdk_set_domain(hwd_control_state_t *ctl, int domain);
static int rocp_sdk_ctl_fn(hwd_context_t *ctx, int code, _papi_int_option_t *option);

typedef struct {
    int initialized;
    int state;
    int component_id;
} rocp_sdk_context_t;

typedef struct {
    int *events_id;
    int num_events;
    vendorp_ctx_t vendor_ctx;
} rocp_sdk_control_t;

papi_vector_t _rocp_sdk_vector = {
    .cmp_info = {
        .name = "rocp_sdk",
        .short_name = "rocp_sdk",
        .version = "1.0",
        .description = "GPU events and metrics via AMD ROCprofiler-SDK",
        .initialized = 0,
        .num_mpx_cntrs = ROCPROF_SDK_MAX_COUNTERS,
    },

    .size = {
        .context = sizeof(rocp_sdk_context_t),
        .control_state = sizeof(rocp_sdk_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = rocp_sdk_init_component,
    .init_thread = rocp_sdk_init_thread,
    .init_control_state = rocp_sdk_init_control_state,
    .shutdown_component = rocp_sdk_shutdown_component,
    .shutdown_thread = rocp_sdk_shutdown_thread,
    .cleanup_eventset = rocp_sdk_cleanup_eventset,

    .update_control_state = rocp_sdk_update_control_state,
    .start = rocp_sdk_start,
    .stop = rocp_sdk_stop,
    .read = rocp_sdk_read,
    .reset = rocp_sdk_reset,

    .ntv_enum_events = rocp_sdk_ntv_enum_events,
    .ntv_code_to_name = rocp_sdk_ntv_code_to_name,
    .ntv_name_to_code = rocp_sdk_ntv_name_to_code,
    .ntv_code_to_descr = rocp_sdk_ntv_code_to_descr,
    .ntv_code_to_info = rocp_sdk_ntv_code_to_info,

    .set_domain = rocp_sdk_set_domain,
    .ctl = rocp_sdk_ctl_fn,
};

static int check_n_initialize(void);

int
rocp_sdk_init_component(int cid)
{
    _rocp_sdk_vector.cmp_info.CmpIdx = cid;
    _rocp_sdk_vector.cmp_info.num_native_events = -1;
    _rocp_sdk_vector.cmp_info.num_cntrs = -1;
    _rocp_sdk_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cid;

    int papi_errno = rocprofiler_sdk_init_pre();
    if (papi_errno != PAPI_OK) {
        _rocp_sdk_vector.cmp_info.initialized = 1;
        _rocp_sdk_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        rocprofiler_sdk_err_get_last(&err_string);
        snprintf(_rocp_sdk_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", err_string);
        return papi_errno;
    }

    sprintf(_rocp_sdk_vector.cmp_info.disabled_reason, "Not initialized. Access component events to initialize it.");
    _rocp_sdk_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    return PAPI_EDELAY_INIT;
}

int
rocp_sdk_init_thread(hwd_context_t *ctx)
{
    rocp_sdk_context_t *rocp_sdk_ctx = (rocp_sdk_context_t *) ctx;
    memset(rocp_sdk_ctx, 0, sizeof(*rocp_sdk_ctx));
    rocp_sdk_ctx->initialized = 1;
    rocp_sdk_ctx->component_id = _rocp_sdk_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

int
rocp_sdk_init_control_state(hwd_control_state_t *ctl __attribute__((unused)))
{
    return check_n_initialize();
}

static int
evt_get_count(int *count)
{
    unsigned int event_code = 0;

    if (rocprofiler_sdk_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (rocprofiler_sdk_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

int
rocp_sdk_init_private(void)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(COMPONENT_LOCK);

    if (_rocp_sdk_vector.cmp_info.initialized) {
        papi_errno = _rocp_sdk_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = rocprofiler_sdk_init();
    if (papi_errno != PAPI_OK) {
        _rocp_sdk_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        rocprofiler_sdk_err_get_last(&err_string);
        snprintf(_rocp_sdk_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", err_string);
        goto fn_fail;
    }

    int count = 0;
    papi_errno = evt_get_count(&count);
    _rocp_sdk_vector.cmp_info.num_native_events = count;
    _rocp_sdk_vector.cmp_info.num_cntrs = count;

  fn_exit:
    _rocp_sdk_vector.cmp_info.initialized = 1;
    _rocp_sdk_vector.cmp_info.disabled = papi_errno;
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocp_sdk_shutdown_component(void)
{
    _rocp_sdk_vector.cmp_info.initialized = 0;
    return rocprofiler_sdk_shutdown();
}

int
rocp_sdk_shutdown_thread(hwd_context_t *ctx)
{
    rocp_sdk_context_t *rocp_sdk_ctx = (rocp_sdk_context_t *) ctx;
    rocp_sdk_ctx->initialized = 0;
    rocp_sdk_ctx->state = 0;
    return PAPI_OK;
}

int
rocp_sdk_cleanup_eventset(hwd_control_state_t *ctl)
{
    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;
    papi_free(rocp_sdk_ctl->events_id);
    rocp_sdk_ctl->events_id = NULL;
    rocp_sdk_ctl->num_events = 0;
    papi_free(rocp_sdk_ctl->vendor_ctx);
    rocp_sdk_ctl->vendor_ctx = NULL;
    return PAPI_OK;
}

int
update_native_events(rocp_sdk_control_t *ctl, NativeInfo_t *ntv_info, int ntv_count)
{
    int papi_errno = PAPI_OK;

    if (ntv_count != ctl->num_events) {
        ctl->events_id = papi_realloc(ctl->events_id, ntv_count * sizeof(*ctl->events_id));
        if (NULL == ctl->events_id) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }
        ctl->num_events = ntv_count;
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        ctl->events_id[i] = ntv_info[i].ni_event;
        ntv_info[i].ni_position = i;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->num_events = 0;
    goto fn_exit;
}

int
rocp_sdk_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx __attribute__((unused)))
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;
    if (rocp_sdk_ctl->vendor_ctx != NULL) {
        return PAPI_ECMP;
    }

    papi_errno = update_native_events(rocp_sdk_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}


int
rocp_sdk_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocp_sdk_context_t *rocp_sdk_ctx = (rocp_sdk_context_t *) ctx;
    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;

    if (rocp_sdk_ctx->state & RPSDK_CTX_RUNNING) {
        SUBDBG("Error! Cannot PAPI_start more than one eventset at a time for every component.");
        return PAPI_EINVAL;
    }

    if ( !(rocp_sdk_ctl->vendor_ctx) ) {
        papi_errno = rocprofiler_sdk_ctx_open(rocp_sdk_ctl->events_id, rocp_sdk_ctl->num_events, &rocp_sdk_ctl->vendor_ctx);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

    papi_errno = rocprofiler_sdk_start(rocp_sdk_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocp_sdk_ctx->state |= RPSDK_CTX_RUNNING;

  fn_exit:
    return papi_errno;
  fn_fail:
    rocp_sdk_ctx->state = 0;
    goto fn_exit;
}

int
rocp_sdk_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocp_sdk_context_t *rocp_sdk_ctx = (rocp_sdk_context_t *) ctx;
    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;

    papi_errno = rocprofiler_sdk_stop(rocp_sdk_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocp_sdk_ctl->vendor_ctx = NULL;

  fn_exit:
    rocp_sdk_ctx->state = 0;
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocp_sdk_read(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl, long long **val, int flags __attribute__((unused)))
{
    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;
    return rocprofiler_sdk_ctx_read(rocp_sdk_ctl->vendor_ctx, val);
}

int
rocp_sdk_reset(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl)
{
    rocp_sdk_control_t *rocp_sdk_ctl = (rocp_sdk_control_t *) ctl;
    return rocprofiler_sdk_ctx_reset(rocp_sdk_ctl->vendor_ctx);
}

int
rocp_sdk_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocprofiler_sdk_evt_enum(event_code, modifier);
}

int
rocp_sdk_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocprofiler_sdk_evt_code_to_name(event_code, name, len);
}

int
rocp_sdk_ntv_name_to_code(const char *name, unsigned int *code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    int papi_errcode = rocprofiler_sdk_evt_name_to_code(name, code);
    return papi_errcode;
}

int
rocp_sdk_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocprofiler_sdk_evt_code_to_descr(event_code, descr, len);
}

int
rocp_sdk_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    info->event_code = event_code;
    info->component_index = _rocp_sdk_vector.cmp_info.CmpIdx;

    return rocprofiler_sdk_evt_code_to_info(event_code, info);
}

int
rocp_sdk_set_domain(hwd_control_state_t *ctl __attribute__((unused)), int domain __attribute__((unused)))
{
    return PAPI_OK;
}

int
rocp_sdk_ctl_fn(hwd_context_t *ctx __attribute__((unused)), int code __attribute__((unused)), _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

int
check_n_initialize(void)
{
    if (!_rocp_sdk_vector.cmp_info.initialized) {
        return rocp_sdk_init_private();
    }
    return _rocp_sdk_vector.cmp_info.disabled;
}
