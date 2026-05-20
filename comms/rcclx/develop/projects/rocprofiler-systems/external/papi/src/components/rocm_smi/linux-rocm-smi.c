//-----------------------------------------------------------------------------
// @file    linux-rocm-smi.c
//
// @ingroup rocm_components
//
// @brief This implements a PAPI component that enables PAPI-C to access
// hardware system management controls for AMD ROCM GPU devices through the
// rocm_smi library.
//
// The open source software license for PAPI conforms to the BSD License
// template.
//-----------------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "rocs.h"

typedef struct {
    int initialized;
    int state;
    int component_id;
} rocmsmi_context_t;

typedef struct {
    unsigned int *events_id;
    int num_events;
    int component_id;
    rocs_ctx_t rocs_ctx;
} rocmsmi_control_t;

extern unsigned int _rocm_smi_lock;
papi_vector_t _rocm_smi_vector;

static int _rocm_smi_init_private(void);

static int
_rocm_smi_check_n_initialize(void)
{
  if (!_rocm_smi_vector.cmp_info.initialized)
      return _rocm_smi_init_private();
  return _rocm_smi_vector.cmp_info.disabled;
}

static int
_rocm_smi_init_thread(hwd_context_t *ctx)
{
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    memset(rocmsmi_ctx, 0, sizeof(*rocmsmi_ctx));
    rocmsmi_ctx->initialized = 1;
    rocmsmi_ctx->component_id = _rocm_smi_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}


static int
_rocm_smi_init_component(int cidx)
{
    _rocm_smi_vector.cmp_info.CmpIdx = cidx;
    _rocm_smi_vector.cmp_info.num_native_events = -1;
    _rocm_smi_vector.cmp_info.num_cntrs = -1;
    _rocm_smi_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    sprintf(_rocm_smi_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    _rocm_smi_vector.cmp_info.disabled = PAPI_EDELAY_INIT;

    return PAPI_EDELAY_INIT;
}

static int
evt_get_count(int *count)
{
    unsigned int event_code = 0;

    if (rocs_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }

    while (rocs_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

static int
_rocm_smi_init_private(void)
{
    int papi_errno = PAPI_OK;

    PAPI_lock(COMPONENT_LOCK);

    if (_rocm_smi_vector.cmp_info.initialized) {
        papi_errno = _rocm_smi_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = rocs_init();
    if (papi_errno != PAPI_OK) {
        _rocm_smi_vector.cmp_info.disabled = papi_errno;
        const char *error_str;
        rocs_err_get_last(&error_str);
        sprintf(_rocm_smi_vector.cmp_info.disabled_reason, "%s", error_str);
        goto fn_fail;
    }

    int count = 0;
    papi_errno = evt_get_count(&count);
    _rocm_smi_vector.cmp_info.num_native_events = count;
    _rocm_smi_vector.cmp_info.num_cntrs = count;
    _rocm_smi_vector.cmp_info.num_mpx_cntrs = count;

  fn_exit:
    _rocm_smi_vector.cmp_info.initialized = 1;
    _rocm_smi_vector.cmp_info.disabled = papi_errno;
    PAPI_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}


static int
_rocm_smi_init_control_state(hwd_control_state_t *ctrl __attribute__((unused)))
{
    return _rocm_smi_check_n_initialize();
}

static int update_native_events(rocmsmi_control_t *, NativeInfo_t *, int);
static int try_open_events(rocmsmi_control_t *);

static int
_rocm_smi_update_control_state(hwd_control_state_t *ctrl, NativeInfo_t *nativeInfo, int nativeCount, hwd_context_t *ctx)
{
    int papi_errno =_rocm_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;

    if (rocmsmi_ctx->state & ROCS_EVENTS_RUNNING) {
        return PAPI_EMISC;
    }

    papi_errno = update_native_events(rocmsmi_ctl, nativeInfo, nativeCount);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return try_open_events(rocmsmi_ctl);
}

int
update_native_events(rocmsmi_control_t *ctl, NativeInfo_t *ntv_info, int ntv_count)
{
    int papi_errno = PAPI_OK;

    unsigned int *events = papi_calloc(ntv_count, sizeof(*events));
    if (events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        events[i] = ntv_info[i].ni_event;
        ntv_info[i].ni_position = i;
    }

    papi_free(ctl->events_id);
    ctl->events_id = events;
    ctl->num_events = ntv_count;

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->num_events = 0;
    goto fn_exit;
}

int
try_open_events(rocmsmi_control_t *rocmsmi_ctl __attribute__((unused)))
{
    return PAPI_OK;
}

static int
_rocm_smi_start(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
    int papi_errno = PAPI_OK;
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (rocmsmi_ctx->state & ROCS_EVENTS_OPENED) {
        return PAPI_EMISC;
    }

    papi_errno = rocs_ctx_open(rocmsmi_ctl->events_id, rocmsmi_ctl->num_events, &rocmsmi_ctl->rocs_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocmsmi_ctx->state = ROCS_EVENTS_OPENED;

    papi_errno = rocs_ctx_start(rocmsmi_ctl->rocs_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocmsmi_ctx->state |= ROCS_EVENTS_RUNNING;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (rocmsmi_ctx->state & ROCS_EVENTS_OPENED) {
        rocs_ctx_close(rocmsmi_ctl->rocs_ctx);
    }
    rocmsmi_ctx->state = 0;
    goto fn_exit;
}

static int
_rocm_smi_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **values, int flags __attribute__((unused)))
{
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (!(rocmsmi_ctx->state & ROCS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }

    return rocs_ctx_read(rocmsmi_ctl->rocs_ctx, values);
}

static int
_rocm_smi_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *values)
{
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (!(rocmsmi_ctx->state & ROCS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }

    return rocs_ctx_write(rocmsmi_ctl->rocs_ctx, values);
}

static int
_rocm_smi_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
    int papi_errno = PAPI_OK;
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (!(rocmsmi_ctx->state & ROCS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }

    papi_errno = rocs_ctx_stop(rocmsmi_ctl->rocs_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocmsmi_ctx->state &= ~ROCS_EVENTS_RUNNING;

    papi_errno = rocs_ctx_close(rocmsmi_ctl->rocs_ctx);

    rocmsmi_ctx->state = 0;
    rocmsmi_ctl->rocs_ctx = NULL;

    return papi_errno;
}

static int
_rocm_smi_cleanup_eventset(hwd_control_state_t *ctrl)
{
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (rocmsmi_ctl->rocs_ctx != NULL) {
        return PAPI_EMISC;
    }

    papi_free(rocmsmi_ctl->events_id);
    rocmsmi_ctl->events_id = NULL;
    rocmsmi_ctl->num_events = 0;

    return PAPI_OK;
}

static int
_rocm_smi_shutdown_thread(hwd_context_t *ctx)
{
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_ctx->state = 0;
    rocmsmi_ctx->initialized = 0;
    return PAPI_OK;
}

static int
_rocm_smi_shutdown_component(void)
{
    if (!_rocm_smi_vector.cmp_info.initialized) {
        return PAPI_EMISC;
    }

    if (_rocm_smi_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_EMISC;
    }

    int papi_errno = rocs_shutdown();
    _rocm_smi_vector.cmp_info.initialized = 0;
    return papi_errno;
}

static int
_rocm_smi_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
    rocmsmi_context_t *rocmsmi_ctx = (rocmsmi_context_t *) ctx;
    rocmsmi_control_t *rocmsmi_ctl = (rocmsmi_control_t *) ctrl;

    if (!(rocmsmi_ctx->state & ROCS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }

    return rocs_ctx_reset(rocmsmi_ctl->rocs_ctx);
}

static int
_rocm_smi_ctrl(hwd_context_t *ctx __attribute__((unused)), int code __attribute__((unused)), _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

static int
_rocm_smi_set_domain(hwd_control_state_t *ctrl __attribute__((unused)), int domain __attribute__((unused)))
{
    return PAPI_OK;
}

static int
_rocm_smi_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    int papi_errno =_rocm_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocs_evt_enum(EventCode, modifier);
}

static int
_rocm_smi_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    int papi_errno =_rocm_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocs_evt_code_to_name(EventCode, name, len);
}

static int
_rocm_smi_ntv_name_to_code(const char *name, unsigned int *EventCode)
{
    int papi_errno =_rocm_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocs_evt_name_to_code(name, EventCode);
}

static int
_rocm_smi_ntv_code_to_descr(unsigned int EventCode, char *desc, int len)
{
    int papi_errno =_rocm_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return rocs_evt_code_to_descr(EventCode, desc, len);
}


papi_vector_t _rocm_smi_vector = {
    .cmp_info = {
        .name = "rocm_smi",
        .short_name = "rocm_smi",
        .version = "2.0",
        .description = "AMD GPU System Management Interface via rocm_smi_lib",
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .initialized = 0,
    },
    .size = {
        .context = sizeof(rocmsmi_context_t),
        .control_state = sizeof(rocmsmi_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },
    .start = _rocm_smi_start,
    .stop  = _rocm_smi_stop,
    .read  = _rocm_smi_read,
    .write = _rocm_smi_write,
    .reset = _rocm_smi_reset,
    .cleanup_eventset = _rocm_smi_cleanup_eventset,
    .init_component = _rocm_smi_init_component,
    .init_thread = _rocm_smi_init_thread,
    .init_control_state = _rocm_smi_init_control_state,
    .update_control_state = _rocm_smi_update_control_state,
    .ctl = _rocm_smi_ctrl,
    .set_domain = _rocm_smi_set_domain,
    .ntv_enum_events = _rocm_smi_ntv_enum_events,
    .ntv_code_to_name = _rocm_smi_ntv_code_to_name,
    .ntv_name_to_code = _rocm_smi_ntv_name_to_code,
    .ntv_code_to_descr = _rocm_smi_ntv_code_to_descr,
    .shutdown_thread = _rocm_smi_shutdown_thread,
    .shutdown_component = _rocm_smi_shutdown_component,
};
