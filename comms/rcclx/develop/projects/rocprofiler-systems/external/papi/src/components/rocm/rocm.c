/**
 * @file    rocm.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief This implements a PAPI component that enables PAPI-C to
 *  access hardware monitoring counters for AMD ROCM GPU devices
 *  through the ROC-profiler library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include "papi_vector.h"
#include "extras.h"
#include "roc_dispatch.h"

/* Init and finalize */
static int rocm_init_component(int cid);
static int rocm_init_thread(hwd_context_t *ctx);
static int rocm_init_control_state(hwd_control_state_t *ctl);
static int rocm_init_private(void);
static int rocm_shutdown_component(void);
static int rocm_shutdown_thread(hwd_context_t *ctx);
static int rocm_cleanup_eventset(hwd_control_state_t *ctl);
static void rocm_dispatch_timer(int n, hwd_siginfo_t *info, void *uc);

/* set and update component state */
static int rocm_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, hwd_context_t *ctx);
static int rocm_set_domain(hwd_control_state_t *ctl, int domain);
static int rocm_ctrl(hwd_context_t *ctx, int code, _papi_int_option_t *option);

/* start and stop monitoring of hardware counters */
static int rocm_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocm_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
                     long long **val, int flags);
static int rocm_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocm_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);

/* event conversion utilities */
static int rocm_ntv_enum_events(unsigned int *event_code, int modifier);
static int rocm_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int rocm_ntv_name_to_code(const char *name, unsigned int *event_code);
static int rocm_ntv_code_to_descr(unsigned int event_code, char *descr,
                                  int len);
static int rocm_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

typedef struct {
    int initialized;
    int state;
    int component_id;
} rocm_context_t;

typedef struct {
    int num_events;
    unsigned int domain;
    unsigned int granularity;
    unsigned int overflow;
    unsigned int overflow_signal;
    unsigned int attached;
    int component_id;
    uint64_t *events_id;
    rocd_ctx_t rocd_ctx;
} rocm_control_t;

papi_vector_t _rocm_vector = {
    .cmp_info = {
        .name = "rocm",
        .short_name = "rocm",
        .version = "2.0",
        .description = "GPU events and metrics via AMD ROCm-PL API",
        .num_mpx_cntrs = PAPI_ROCM_MAX_COUNTERS,
        .num_cntrs = PAPI_ROCM_MAX_COUNTERS,
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
        .context = sizeof(rocm_context_t),
        .control_state = sizeof(rocm_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = rocm_init_component,
    .init_thread = rocm_init_thread,
    .init_control_state = rocm_init_control_state,
    .shutdown_component = rocm_shutdown_component,
    .shutdown_thread = rocm_shutdown_thread,
    .cleanup_eventset = rocm_cleanup_eventset,
    .dispatch_timer = rocm_dispatch_timer,

    .update_control_state = rocm_update_control_state,
    .set_domain = rocm_set_domain,
    .ctl = rocm_ctrl,

    .start = rocm_start,
    .stop = rocm_stop,
    .read = rocm_read,
    .reset = rocm_reset,

    .ntv_enum_events = rocm_ntv_enum_events,
    .ntv_code_to_name = rocm_ntv_code_to_name,
    .ntv_name_to_code = rocm_ntv_name_to_code,
    .ntv_code_to_descr = rocm_ntv_code_to_descr,
    .ntv_code_to_info = rocm_ntv_code_to_info,
};

static int check_n_initialize(void);

int
rocm_init_component(int cid)
{
    _rocm_vector.cmp_info.CmpIdx = cid;
    _rocm_vector.cmp_info.num_native_events = -1;
    _rocm_vector.cmp_info.num_cntrs = -1;
    _rocm_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cid;
    SUBDBG("ENTER: cid: %d\n", cid);

    int papi_errno = rocd_init_environment();
    if (papi_errno != PAPI_OK) {
        _rocm_vector.cmp_info.initialized = 1;
        _rocm_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        rocd_err_get_last(&err_string);
        int expect = snprintf(_rocm_vector.cmp_info.disabled_reason,
                              PAPI_MAX_STR_LEN, "%s", err_string);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("disabled_reason truncated");
        }
        goto fn_fail;
    }

    sprintf(_rocm_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    papi_errno = PAPI_EDELAY_INIT;
    _rocm_vector.cmp_info.disabled = papi_errno;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_init_thread(hwd_context_t *ctx)
{
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    memset(rocm_ctx, 0, sizeof(*rocm_ctx));
    rocm_ctx->initialized = 1;
    rocm_ctx->component_id = _rocm_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

int
rocm_init_control_state(hwd_control_state_t *ctl __attribute__((unused)))
{
    return check_n_initialize();
}

static int
evt_get_count(int *count)
{
    uint64_t event_code = 0;

    if (rocd_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (rocd_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

int
rocm_init_private(void)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(COMPONENT_LOCK);
    SUBDBG("ENTER\n");

    if (_rocm_vector.cmp_info.initialized) {
        papi_errno = _rocm_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = rocd_init();
    if (papi_errno != PAPI_OK) {
        _rocm_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        rocd_err_get_last(&err_string);
        int expect = snprintf(_rocm_vector.cmp_info.disabled_reason,
                              PAPI_MAX_STR_LEN, "%s", err_string);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("disabled_reason truncated");
        }

        goto fn_fail;
    }

    int count = 0;
    papi_errno = evt_get_count(&count);
    _rocm_vector.cmp_info.num_native_events = count;
    _rocm_vector.cmp_info.num_cntrs = count;
    _rocm_vector.cmp_info.initialized = 1;
    int strLen = snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", "");
    if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
        SUBDBG("Failed to fully write disabled_reason.\n");
    }

    goto fn_exit;

  fn_fail:
    _rocm_vector.cmp_info.initialized = 0;

  fn_exit:
    _rocm_vector.cmp_info.disabled = papi_errno;
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
}

int
rocm_shutdown_component(void)
{
    int papi_errno = PAPI_OK;
    int orig_state = _rocm_vector.cmp_info.initialized;
    _rocm_vector.cmp_info.initialized = 0;
    SUBDBG("ENTER\n");

    if (!_rocm_vector.cmp_info.initialized) {
        goto fn_exit;
    }

    if (_rocm_vector.cmp_info.disabled != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = rocd_shutdown();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    SUBDBG("EXIT\n");
    return papi_errno;
  fn_fail:
    _rocm_vector.cmp_info.initialized = orig_state;
    goto fn_exit;
}

int
rocm_shutdown_thread(hwd_context_t *ctx)
{
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_ctx->initialized = 0;
    rocm_ctx->state = 0;
    return PAPI_OK;
}

int
rocm_cleanup_eventset(hwd_control_state_t *ctl)
{
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocd_ctx != NULL) {
        SUBDBG("Cannot cleanup an eventset that is running.");
        return PAPI_ECMP;
    }

    papi_free(rocm_ctl->events_id);
    rocm_ctl->events_id = NULL;
    rocm_ctl->num_events = 0;

    return PAPI_OK;
}

static int
counter_sampling_compatible_with_prof_mode(void)
{
    return (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE);
}

void
rocm_dispatch_timer(int n __attribute__((unused)), hwd_siginfo_t *info, void *uc)
{
    _papi_hwi_context_t hw_context;
    vptr_t address;
    EventSetInfo_t *ESI;
    ThreadInfo_t *thread;
    int cidx;

    if (!counter_sampling_compatible_with_prof_mode()) {
        SUBDBG("Counter sampling is not compatible with intercept mode");
        return;
    }

    _papi_hwi_lock(_rocm_lock);

    cidx = _rocm_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread(0);

    if (thread == NULL) {
        SUBDBG("thread == NULL in user_signal_handler!");
        goto fn_exit;
    }

    ESI = thread->running_eventset[cidx];
    rocm_control_t *rocm_ctl = (rocm_control_t *) ESI->ctl_state;

    if (rocm_ctl->rocd_ctx == NULL) {
        SUBDBG("ESI == NULL in user_signal_handler!");
        goto fn_exit;
    }

    hw_context.si = info;
    hw_context.ucontext = (hwd_ucontext_t *) uc;

    address = GET_OVERFLOW_ADDRESS(hw_context);

    _papi_hwi_dispatch_overflow_signal((void *)&hw_context, address, NULL, 0, 0,
                                       &thread, cidx);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return;
}

static int update_native_events(rocm_control_t *, NativeInfo_t *, int);
static int try_open_events(rocm_control_t *);

int
rocm_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info,
                          int ntv_count,
                          hwd_context_t *ctx __attribute__((unused)))
{
    SUBDBG("ENTER: ctl: %p, ntv_info: %p, ntv_count: %d, ctx: %p\n", ctl, ntv_info, ntv_count, ctx);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocd_ctx != NULL) {
        SUBDBG("Cannot update events in an eventset that has been already "
               "started.");
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    papi_errno = update_native_events(rocm_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = try_open_events(rocm_ctl);

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

struct event_map_item {
    int event_id;
    int frontend_idx;
};

static int
compare(const void *a, const void *b)
{
    struct event_map_item *A = (struct event_map_item *) a;
    struct event_map_item *B = (struct event_map_item *) b;
    return  A->event_id - B->event_id;
}

int
update_native_events(rocm_control_t *ctl, NativeInfo_t *ntv_info,
                     int ntv_count)
{
    int papi_errno = PAPI_OK;
    struct event_map_item sorted_events[PAPI_ROCM_MAX_COUNTERS];

    if (ntv_count != ctl->num_events) {
        ctl->num_events = ntv_count;
        if (ntv_count == 0) {
            papi_free(ctl->events_id);
            ctl->events_id = NULL;
            goto fn_exit;
        }
        else {
            ctl->events_id = papi_realloc(ctl->events_id, ntv_count * sizeof(*ctl->events_id));
            if (ctl->events_id == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }
        }
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        sorted_events[i].event_id = ntv_info[i].ni_event;
        sorted_events[i].frontend_idx = i;
    }

    qsort(sorted_events, ntv_count, sizeof(struct event_map_item), compare);

    for (i = 0; i < ntv_count; ++i) {
        ctl->events_id[i] = sorted_events[i].event_id;
        ntv_info[sorted_events[i].frontend_idx].ni_position = i;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->num_events = 0;
    goto fn_exit;
}

int
try_open_events(rocm_control_t *rocm_ctl)
{
    int papi_errno = PAPI_OK;
    rocd_ctx_t rocd_ctx;

    if (rocm_prof_mode != ROCM_PROFILE_SAMPLING_MODE) {
        /* Do not try open for intercept mode */
        return papi_errno;
    }

    if (rocm_ctl->num_events <= 0) {
        return PAPI_OK;
    }

    papi_errno = rocd_ctx_open(rocm_ctl->events_id, rocm_ctl->num_events,
                               &rocd_ctx);
    if (papi_errno != PAPI_OK) {
        rocm_cleanup_eventset(rocm_ctl);
        return papi_errno;
    }

    return rocd_ctx_close(rocd_ctx);
}

int
rocm_set_domain(hwd_control_state_t *ctl __attribute__((unused)),
                int domain __attribute__((unused)))
{
    return PAPI_OK;
}

int
rocm_ctrl(hwd_context_t *ctx __attribute__((unused)),
          int code __attribute__((unused)),
          _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

int
rocm_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;
    SUBDBG("ENTER: ctx: %p, ctl: %p\n", ctx, ctl);

    if (rocm_ctx->state & ROCM_EVENTS_OPENED) {
        SUBDBG("Error! Cannot PAPI_start more than one eventset at a time for every component.");
        papi_errno = PAPI_ECNFLCT;
        goto fn_fail;
    }

    papi_errno = rocd_ctx_open(rocm_ctl->events_id, rocm_ctl->num_events,
                               &rocm_ctl->rocd_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_ctx->state = ROCM_EVENTS_OPENED;

    papi_errno = rocd_ctx_start(rocm_ctl->rocd_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_ctx->state |= ROCM_EVENTS_RUNNING;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    if (rocm_ctx->state == ROCM_EVENTS_OPENED) {
        rocd_ctx_close(rocm_ctl->rocd_ctx);
    }
    rocm_ctx->state = 0;
    goto fn_exit;
}

int
rocm_read(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl,
          long long **val, int flags __attribute__((unused)))
{
    int papi_errno = PAPI_OK;
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;
    SUBDBG("ENTER: ctx: %p, ctl: %p, val: %p, flags: %d\n", ctx, ctl, val, flags);

    if (rocm_ctl->rocd_ctx == NULL) {
        SUBDBG("Error! Cannot PAPI_read counters for an eventset that has not been PAPI_start'ed.");
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = rocd_ctx_read(rocm_ctl->rocd_ctx, val);

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;
    SUBDBG("ENTER: ctx: %p, ctl: %p\n", ctx, ctl);

    if (!(rocm_ctx->state & ROCM_EVENTS_OPENED)) {
        SUBDBG("Error! Cannot PAPI_stop counters for an eventset that has not been PAPI_start'ed.");
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = rocd_ctx_stop(rocm_ctl->rocd_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_ctx->state &= ~ROCM_EVENTS_RUNNING;

    papi_errno = rocd_ctx_close(rocm_ctl->rocd_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_ctx->state = 0;
    rocm_ctl->rocd_ctx = NULL;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_reset(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl)
{
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocd_ctx == NULL) {
        SUBDBG("Cannot reset counters for an eventset that has not been started.");
        return PAPI_EMISC;
    }

    return rocd_ctx_reset(rocm_ctl->rocd_ctx);
}

int
rocm_ntv_enum_events(unsigned int *event_code, int modifier)
{
    SUBDBG("ENTER: event_code: %u, modifier: %d\n", *event_code, modifier);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    uint64_t code = *(uint64_t *) event_code;
    papi_errno = rocd_evt_enum(&code, modifier);
    *event_code = (unsigned int) code;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    SUBDBG("ENTER: event_code: %u, name: %p, len: %d\n", event_code, name, len);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = rocd_evt_code_to_name((uint64_t) event_code, name, len);

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_ntv_name_to_code(const char *name, unsigned int *code)
{
    SUBDBG("ENTER: name: %s, code: %p\n", name, code);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    uint64_t event_code;
    papi_errno = rocd_evt_name_to_code(name, &event_code);
    *code = (unsigned int) event_code;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    SUBDBG("ENTER: event_code: %u, descr: %p, len: %d\n", event_code, descr, len);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = rocd_evt_code_to_descr((uint64_t) event_code, descr, len);

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    SUBDBG("ENTER: event_code: %u, info: %p\n", event_code, info);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = rocd_evt_code_to_info((uint64_t) event_code, info);

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
check_n_initialize(void)
{
    if (!_rocm_vector.cmp_info.initialized) {
        return rocm_init_private();
    }
    return _rocm_vector.cmp_info.disabled;
}
