/**
 * @file    linux-amd-smi.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "amds.h"
#include "amds_priv.h"
extern unsigned int _amd_smi_lock;

typedef struct {
    int initialized;
    int state;
    int component_id;
} amdsmi_context_t;

typedef struct {
    unsigned int *events_id;
    int num_events;
    int component_id;
    amds_ctx_t amds_ctx;
} amdsmi_control_t;

papi_vector_t _amd_smi_vector;

static int _amd_smi_init_private(void);

static int _amd_smi_check_n_initialize(void) {
    if (!_amd_smi_vector.cmp_info.initialized)
        return _amd_smi_init_private();
    return _amd_smi_vector.cmp_info.disabled;
}

static int _amd_smi_init_thread(hwd_context_t *ctx) {
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    memset(amdsmi_ctx, 0, sizeof(*amdsmi_ctx));
    amdsmi_ctx->initialized = 1;
    amdsmi_ctx->component_id = _amd_smi_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

static int _amd_smi_init_component(int cidx) {
    _amd_smi_vector.cmp_info.CmpIdx = cidx;
    _amd_smi_vector.cmp_info.num_native_events = -1;
    _amd_smi_vector.cmp_info.num_cntrs = -1;
    _amd_smi_vector.cmp_info.num_mpx_cntrs = -1;
    _amd_smi_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    CHECK_SNPRINTF(_amd_smi_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
             "Not initialized. Access an AMD SMI event to initialize.");
    _amd_smi_vector.cmp_info.disabled = PAPI_EDELAY_INIT;

    return PAPI_EDELAY_INIT;
}

static int evt_get_count(int *count) {
    unsigned int event_code = 0;
    if (amds_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (amds_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }
    return PAPI_OK;
}

static int _amd_smi_init_private(void) {
    int papi_errno = PAPI_OK;
    int initialized = 0;
    PAPI_lock(COMPONENT_LOCK);

    if (_amd_smi_vector.cmp_info.initialized) {
        papi_errno = _amd_smi_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = amds_init();  // initialize AMD SMI library and events
    if (papi_errno != PAPI_OK) {
        _amd_smi_vector.cmp_info.disabled = papi_errno;
        const char *error_str = NULL;
        amds_err_get_last(&error_str);
        if (!error_str || !error_str[0])
            error_str = "AMD SMI component initialization failed";
        CHECK_SNPRINTF(_amd_smi_vector.cmp_info.disabled_reason,
                 sizeof _amd_smi_vector.cmp_info.disabled_reason, "%s",
                 error_str);
        goto fn_fail;
    }

    int count = 0;
    papi_errno = evt_get_count(&count);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    _amd_smi_vector.cmp_info.num_native_events = count;

    uint32_t counter_slots = amds_get_counter_slot_capacity();
    _amd_smi_vector.cmp_info.num_cntrs = count + (int)counter_slots;
    _amd_smi_vector.cmp_info.num_mpx_cntrs = count;

fn_exit:
    initialized = (papi_errno == PAPI_OK);
    _amd_smi_vector.cmp_info.disabled = papi_errno;
    _amd_smi_vector.cmp_info.initialized = initialized;
    if (initialized) {
        _amd_smi_vector.cmp_info.disabled_reason[0] = '\0';
    }
    PAPI_unlock(COMPONENT_LOCK);
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int _amd_smi_init_control_state(hwd_control_state_t *ctrl) {
    (void) ctrl;
    return _amd_smi_check_n_initialize();
}

static int update_native_events(amdsmi_control_t *ctl, NativeInfo_t *ntvInfo, int ntvCount)
{
    if (!ctl) return PAPI_EINVAL;
    if (ntvCount < 0) return PAPI_EINVAL;

    if (ntvCount == 0) {
        if (ctl->events_id) papi_free(ctl->events_id);
        ctl->events_id = NULL;
        ctl->num_events = 0;
        return PAPI_OK;
    }

    if (!ntvInfo) return PAPI_EINVAL;

    // Allocate a new array; leave ctl unchanged until success.
    unsigned int *events = papi_calloc((size_t)ntvCount, sizeof(*events));
    if (!events) {
        // Old ctl->events_id/num_events remain intact on allocation failure.
        return PAPI_ENOMEM;
    }

    int i;
    for (i = 0; i < ntvCount; ++i) {
        events[i] = ntvInfo[i].ni_event;
        ntvInfo[i].ni_position = i;
    }

    // Swap in the new array atomically.
    if (ctl->events_id) papi_free(ctl->events_id);
    ctl->events_id = events;
    ctl->num_events = ntvCount;

    return PAPI_OK;
}

static int _amd_smi_update_control_state(hwd_control_state_t *ctrl, NativeInfo_t *nativeInfo,
                                         int nativeCount, hwd_context_t *ctx) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *) ctrl;
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    if (amdsmi_ctx->state & AMDS_EVENTS_RUNNING) {
        return PAPI_EMISC;
    }
    papi_errno = update_native_events(amdsmi_ctl, nativeInfo, nativeCount);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return PAPI_OK;
}

static int _amd_smi_start(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
    int papi_errno = PAPI_OK;
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *) ctrl;

    if (amdsmi_ctx->state & AMDS_EVENTS_RUNNING) {
        return PAPI_EMISC;
    }
    papi_errno = amds_ctx_open(amdsmi_ctl->events_id, amdsmi_ctl->num_events, &amdsmi_ctl->amds_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    amdsmi_ctx->state = AMDS_EVENTS_OPENED;

    papi_errno = amds_ctx_start(amdsmi_ctl->amds_ctx);
    if (papi_errno != PAPI_OK) {
        // If start fails, close the context and reset state
        amds_ctx_close(amdsmi_ctl->amds_ctx);
        amdsmi_ctx->state = 0;
        amdsmi_ctl->amds_ctx = NULL;
        return papi_errno;
    }
    amdsmi_ctx->state |= AMDS_EVENTS_RUNNING;
    return PAPI_OK;
}

static int _amd_smi_read(hwd_context_t *ctx, hwd_control_state_t *ctrl,
                         long long **values, int flags) {
  (void)ctx; (void)flags;
  amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *)ctx;
  amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *)ctrl;
  if (!(amdsmi_ctx->state & AMDS_EVENTS_RUNNING) || !amdsmi_ctl->amds_ctx)           // fail only if ctx is gone
    return PAPI_EMISC;
  return amds_ctx_read(amdsmi_ctl->amds_ctx, values);
}


static int _amd_smi_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *values) {
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *) ctrl;
    if (!(amdsmi_ctx->state & AMDS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    return amds_ctx_write(amdsmi_ctl->amds_ctx, values);
}

static int _amd_smi_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
  amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *)ctx;
  amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *)ctrl;
  if (!(amdsmi_ctx->state & AMDS_EVENTS_RUNNING)) return PAPI_EMISC;

  int papi_errno = amds_ctx_stop(amdsmi_ctl->amds_ctx);
  amdsmi_ctx->state &= ~AMDS_EVENTS_RUNNING;
  return papi_errno;
}

static int _amd_smi_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *) ctrl;
    if (!(amdsmi_ctx->state & AMDS_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    return amds_ctx_reset(amdsmi_ctl->amds_ctx);
}

static int _amd_smi_cleanup_eventset(hwd_control_state_t *ctrl) {
  amdsmi_control_t *amdsmi_ctl = (amdsmi_control_t *)ctrl;

  if (amdsmi_ctl->amds_ctx) {
    (void)amds_ctx_stop(amdsmi_ctl->amds_ctx);  // safe if not running
    (void)amds_ctx_close(amdsmi_ctl->amds_ctx);
    amdsmi_ctl->amds_ctx = NULL;
  }

  if (amdsmi_ctl->events_id) {
    papi_free(amdsmi_ctl->events_id);
    amdsmi_ctl->events_id = NULL;
    amdsmi_ctl->num_events = 0;
  }
  return PAPI_OK;
}



static int _amd_smi_shutdown_thread(hwd_context_t *ctx) {
    amdsmi_context_t *amdsmi_ctx = (amdsmi_context_t *) ctx;
    amdsmi_ctx->state = 0;
    amdsmi_ctx->initialized = 0;
    return PAPI_OK;
}

static int _amd_smi_shutdown_component(void) {
    if (!_amd_smi_vector.cmp_info.initialized) {
        return PAPI_EMISC;
    }
    if (_amd_smi_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_EMISC;
    }
    int papi_errno = amds_shutdown();
    _amd_smi_vector.cmp_info.initialized = 0;
    return papi_errno;
}

static int _amd_smi_ctrl(hwd_context_t *ctx, int code, _papi_int_option_t *option) {
    (void) ctx; (void) code; (void) option;
    // No special control actions needed for this component
    return PAPI_OK;
}

static int _amd_smi_set_domain(hwd_control_state_t *ctrl, int domain) {
    (void) ctrl; (void) domain;
    // Only default user/kernel domain is supported
    return PAPI_OK;
}

/* Native event API functions */
static int _amd_smi_ntv_enum_events(unsigned int *EventCode, int modifier) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return amds_evt_enum(EventCode, modifier);
}

static int _amd_smi_ntv_code_to_name(unsigned int EventCode, char *name, int len) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return amds_evt_code_to_name(EventCode, name, len);
}

static int _amd_smi_ntv_name_to_code(const char *name, unsigned int *EventCode) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return amds_evt_name_to_code(name, EventCode);
}

static int _amd_smi_ntv_code_to_descr(unsigned int EventCode, char *desc, int len) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return amds_evt_code_to_descr(EventCode, desc, len);
}

static int _amd_smi_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return amds_evt_code_to_info(EventCode, info);
}

/* Export the component interface */
papi_vector_t _amd_smi_vector = {
    .cmp_info = {
        .name = "amd_smi",
        .short_name = "amd_smi",
        .version = "1.0",
        .description = "AMD GPU System Management Interface via AMD SMI library",
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
    },
    .size = {
        .context = sizeof(amdsmi_context_t),
        .control_state = sizeof(amdsmi_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },
    .init_thread =       _amd_smi_init_thread,
    .init_component =    _amd_smi_init_component,
    .init_control_state = _amd_smi_init_control_state,
    .update_control_state = _amd_smi_update_control_state,
    .start =            _amd_smi_start,
    .stop =             _amd_smi_stop,
    .read =             _amd_smi_read,
    .write =            _amd_smi_write,
    .reset =            _amd_smi_reset,
    .cleanup_eventset = _amd_smi_cleanup_eventset,
    .shutdown_thread =  _amd_smi_shutdown_thread,
    .shutdown_component = _amd_smi_shutdown_component,
    .ctl =              _amd_smi_ctrl,
    .set_domain =       _amd_smi_set_domain,
    .ntv_enum_events =  _amd_smi_ntv_enum_events,
    .ntv_code_to_name = _amd_smi_ntv_code_to_name,
    .ntv_name_to_code = _amd_smi_ntv_name_to_code,
    .ntv_code_to_descr = _amd_smi_ntv_code_to_descr,
    .ntv_code_to_info = _amd_smi_ntv_code_to_info,
};
