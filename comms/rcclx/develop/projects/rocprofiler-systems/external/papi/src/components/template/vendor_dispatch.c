#include "vendor_dispatch.h"
#include "vendor_common.h"
#include "vendor_profiler_v1.h"

int
vendord_init_pre(void)
{
    return vendorp1_init_pre();
}

int
vendord_init(void)
{
    int papi_errno = vendorc_init();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendorp1_init();
}

int
vendord_shutdown(void)
{
    int papi_errno = vendorp1_shutdown();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    return vendorc_shutdown();
}

int
vendord_ctx_open(unsigned int *events_id, int num_events, vendord_ctx_t *ctx)
{
    return vendorp1_ctx_open(events_id, num_events, ctx);
}

int
vendord_ctx_start(vendord_ctx_t ctx)
{
    return vendorp1_ctx_start(ctx);
}

int
vendord_ctx_read(vendord_ctx_t ctx, long long **counters)
{
    return vendorp1_ctx_read(ctx, counters);
}

int
vendord_ctx_stop(vendord_ctx_t ctx)
{
    return vendorp1_ctx_stop(ctx);
}

int
vendord_ctx_reset(vendord_ctx_t ctx)
{
    return vendorp1_ctx_reset(ctx);
}

int
vendord_ctx_close(vendord_ctx_t ctx)
{
    return vendorp1_ctx_close(ctx);
}

int
vendord_err_get_last(const char **error)
{
    return vendorc_err_get_last(error);
}

int
vendord_evt_enum(unsigned int *event_code, int modifier)
{
    return vendorp1_evt_enum(event_code, modifier);
}

int
vendord_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    return vendorp1_evt_code_to_name(event_code, name, len);
}

int
vendord_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    return vendorp1_evt_code_to_descr(event_code, descr, len);
}

int
vendord_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    return vendorp1_evt_code_to_info(event_code, info);
}

int
vendord_evt_name_to_code(const char *name, unsigned int *event_code)
{
    return vendorp1_evt_name_to_code(name, event_code);
}
