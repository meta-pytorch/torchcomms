#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "vendor_dispatch.h"

#define TEMPLATE_MAX_COUNTERS (16)

/* Init and finalize */
static int templ_init_component(int cid);
static int templ_init_thread(hwd_context_t *ctx);
static int templ_init_control_state(hwd_control_state_t *ctl);
static int templ_init_private(void);
static int templ_shutdown_component(void);
static int templ_shutdown_thread(hwd_context_t *ctx);
static int templ_cleanup_eventset(hwd_control_state_t *ctl);

/* Set and update component state */
static int templ_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx);

/* Start and stop profiling of hardware events */
static int templ_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int templ_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int templ_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int templ_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);

/* Event conversion */
static int templ_ntv_enum_events(unsigned int *event_code, int modifier);
static int templ_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int templ_ntv_name_to_code(const char *name, unsigned int *event_code);
static int templ_ntv_code_to_descr(unsigned int event_code, char *descr, int len);
static int templ_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

static int templ_set_domain(hwd_control_state_t *ctl, int domain);
static int templ_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option);

typedef struct {
    int initialized;
    int state;
    int component_id;
} templ_context_t;

typedef struct {
    unsigned int *events_id;
    int num_events;
    vendord_ctx_t vendor_ctx;
} templ_control_t;

papi_vector_t _template_vector = {
    .cmp_info = {
        .name = "templ",
        .short_name = "templ",
        .version = "1.0",
        .description = "Template component for new components",
        .initialized = 0,
        .num_mpx_cntrs = TEMPLATE_MAX_COUNTERS,
    },

    .size = {
        .context = sizeof(templ_context_t),
        .control_state = sizeof(templ_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = templ_init_component,
    .init_thread = templ_init_thread,
    .init_control_state = templ_init_control_state,
    .shutdown_component = templ_shutdown_component,
    .shutdown_thread = templ_shutdown_thread,
    .cleanup_eventset = templ_cleanup_eventset,

    .update_control_state = templ_update_control_state,
    .start = templ_start,
    .stop = templ_stop,
    .read = templ_read,
    .reset = templ_reset,

    .ntv_enum_events = templ_ntv_enum_events,
    .ntv_code_to_name = templ_ntv_code_to_name,
    .ntv_name_to_code = templ_ntv_name_to_code,
    .ntv_code_to_descr = templ_ntv_code_to_descr,
    .ntv_code_to_info = templ_ntv_code_to_info,

    .set_domain = templ_set_domain,
    .ctl = templ_ctl,
};

static int check_n_initialize(void);

int
templ_init_component(int cid)
{
    _template_vector.cmp_info.CmpIdx = cid;
    _template_vector.cmp_info.num_native_events = -1;
    _template_vector.cmp_info.num_cntrs = -1;
    _templ_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cid;

    int papi_errno = vendord_init_pre();
    if (papi_errno != PAPI_OK) {
        _template_vector.cmp_info.initialized = 1;
        _template_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        vendord_err_get_last(&err_string);
        snprintf(_template_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", err_string);
        return papi_errno;
    }

    sprintf(_template_vector.cmp_info.disabled_reason, "Not initialized. Access component events to initialize it.");
    _template_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    return PAPI_EDELAY_INIT;
}

int
templ_init_thread(hwd_context_t *ctx)
{
    templ_context_t *templ_ctx = (templ_context_t *) ctx;
    memset(templ_ctx, 0, sizeof(*templ_ctx));
    templ_ctx->initialized = 1;
    templ_ctx->component_id = _template_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

int
templ_init_control_state(hwd_control_state_t *ctl __attribute__((unused)))
{
    return check_n_initialize();
}

static int
evt_get_count(int *count)
{
    unsigned int event_code = 0;

    if (vendord_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (vendord_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

int
templ_init_private(void)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(COMPONENT_LOCK);

    if (_template_vector.cmp_info.initialized) {
        papi_errno = _template_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = vendord_init();
    if (papi_errno != PAPI_OK) {
        _template_vector.cmp_info.disabled = papi_errno;
        const char *err_string;
        vendord_err_get_last(&err_string);
        snprintf(_template_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", err_string);
        goto fn_fail;
    }

    int count = 0;
    papi_errno = evt_get_count(&count);
    _template_vector.cmp_info.num_native_events = count;
    _template_vector.cmp_info.num_cntrs = count;

  fn_exit:
    _template_vector.cmp_info.initialized = 1;
    _template_vector.cmp_info.disabled = papi_errno;
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
templ_shutdown_component(void)
{
    _template_vector.cmp_info.initialized = 0;
    return vendord_shutdown();
}

int
templ_shutdown_thread(hwd_context_t *ctx)
{
    templ_context_t *templ_ctx = (templ_context_t *) ctx;
    templ_ctx->initialized = 0;
    templ_ctx->state = 0;
    return PAPI_OK;
}

int
templ_cleanup_eventset(hwd_control_state_t *ctl)
{
    templ_control_t *templ_ctl = (templ_control_t *) ctl;
    papi_free(templ_ctl->events_id);
    templ_ctl->events_id = NULL;
    templ_ctl->num_events = 0;
    return PAPI_OK;
}

static int update_native_events(templ_control_t *, NativeInfo_t *, int);
static int try_open_events(templ_control_t *);

int
templ_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx __attribute__((unused)))
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    templ_control_t *templ_ctl = (templ_control_t *) ctl;
    if (templ_ctl->vendor_ctx != NULL) {
        return PAPI_ECMP;
    }

    papi_errno = update_native_events(templ_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return try_open_events(templ_ctl);
}

int
update_native_events(templ_control_t *ctl, NativeInfo_t *ntv_info, int ntv_count)
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
try_open_events(templ_control_t *templ_ctl)
{
    int papi_errno = PAPI_OK;
    vendord_ctx_t vendor_ctx;

    papi_errno = vendord_ctx_open(templ_ctl->events_id, templ_ctl->num_events, &vendor_ctx);
    if (papi_errno != PAPI_OK) {
        templ_cleanup_eventset(templ_ctl);
        return papi_errno;
    }

    return vendord_ctx_close(vendor_ctx);
}

int
templ_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    templ_context_t *templ_ctx = (templ_context_t *) ctx;
    templ_control_t *templ_ctl = (templ_control_t *) ctl;

    if (templ_ctx->state & TEMPL_CTX_OPENED) {
        return PAPI_EINVAL;
    }

    papi_errno = vendord_ctx_open(templ_ctl->events_id, templ_ctl->num_events, &templ_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    templ_ctx->state = TEMPL_CTX_OPENED;

    papi_errno = vendord_ctx_start(templ_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    templ_ctx->state |= TEMPL_CTX_RUNNING;

  fn_exit:
    return papi_errno;
  fn_fail:
    vendord_ctx_close(templ_ctl->vendor_ctx);
    templ_ctx->state = 0;
    goto fn_exit;
}

int
templ_read(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl, long long **val, int flags __attribute__((unused)))
{
    templ_control_t *templ_ctl = (templ_control_t *) ctl;
    return vendord_ctx_read(templ_ctl->vendor_ctx, val);
}

int
templ_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    templ_context_t *templ_ctx = (templ_context_t *) ctx;
    templ_control_t *templ_ctl = (templ_control_t *) ctl;

    if (!(templ_ctx->state & TEMPL_CTX_OPENED)) {
        return PAPI_EINVAL;
    }

    papi_errno = vendord_ctx_stop(templ_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    templ_ctx->state &= ~TEMPL_CTX_RUNNING;

    papi_errno = vendord_ctx_close(templ_ctl->vendor_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    templ_ctx->state = 0;
    templ_ctl->vendor_ctx = NULL;

    return papi_errno;
}

int
templ_reset(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl)
{
    templ_control_t *templ_ctl = (templ_control_t *) ctl;
    return vendord_ctx_reset(templ_ctl->vendor_ctx);
}

int
templ_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendord_evt_enum(event_code, modifier);
}

int
templ_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendord_evt_code_to_name(event_code, name, len);
}

int
templ_ntv_name_to_code(const char *name, unsigned int *code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendord_evt_name_to_code(name, code);
}

int
templ_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendord_evt_code_to_descr(event_code, descr, len);
}

int
templ_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendord_evt_code_to_info(event_code, info);
}

int
templ_set_domain(hwd_control_state_t *ctl __attribute__((unused)), int domain __attribute__((unused)))
{
    return PAPI_OK;
}

int
templ_ctl(hwd_context_t *ctx __attribute__((unused)), int code __attribute__((unused)), _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

int
check_n_initialize(void)
{
    if (!_template_vector.cmp_info.initialized) {
        return templ_init_private();
    }
    return _template_vector.cmp_info.disabled;
}
