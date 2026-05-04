/**
 * @file    roc_dispatch.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @brief rocm component dispatch layer. Dispatches profiling
 * to the appropriate backend interface (e.g. rocprofiler).
 */

#include "roc_dispatch.h"
#include "roc_common.h"
#include "roc_profiler.h"

int
rocd_init_environment(void)
{
    return rocp_init_environment();
}

int
rocd_init(void)
{
    int papi_errno = rocc_init();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    papi_errno = rocp_init();
    return papi_errno;
}

int
rocd_shutdown(void)
{
    int papi_errno = rocp_shutdown();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    papi_errno = rocc_shutdown();
    return papi_errno;
}

int
rocd_evt_enum(uint64_t *event_code, int modifier)
{
    return rocp_evt_enum(event_code, modifier);
}

int
rocd_evt_code_to_descr(uint64_t event_code, char *descr, int len)
{
    return rocp_evt_code_to_descr(event_code, descr, len);
}

int
rocd_evt_name_to_code(const char *name, uint64_t *event_code)
{
    return rocp_evt_name_to_code(name, event_code);
}

int
rocd_evt_code_to_name(uint64_t event_code, char *name, int len)
{
    return rocp_evt_code_to_name(event_code, name, len);
}

int
rocd_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info)
{
    return rocp_evt_code_to_info(event_code, info);
}

int
rocd_err_get_last(const char **error_str)
{
    return rocc_err_get_last(error_str);
}

int
rocd_ctx_open(uint64_t *events_id, int num_events, rocd_ctx_t *ctx)
{
    return rocp_ctx_open(events_id, num_events, ctx);
}

int
rocd_ctx_close(rocd_ctx_t ctx)
{
    return rocp_ctx_close(ctx);
}

int
rocd_ctx_start(rocd_ctx_t ctx)
{
    return rocp_ctx_start(ctx);
}

int
rocd_ctx_stop(rocd_ctx_t ctx)
{
    return rocp_ctx_stop(ctx);
}

int
rocd_ctx_read(rocd_ctx_t ctx, long long **counters)
{
    return rocp_ctx_read(ctx, counters);
}

int
rocd_ctx_reset(rocd_ctx_t ctx)
{
    return rocp_ctx_reset(ctx);
}
