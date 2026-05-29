#ifndef __VENDOR_PROFILER_V1_H__
#define __VENDOR_PROFILER_V1_H__

typedef struct vendord_ctx *vendorp_ctx_t;

extern int rocprofiler_sdk_init_pre(void);
extern int rocprofiler_sdk_init(void);
extern int rocprofiler_sdk_shutdown(void);

extern int rocprofiler_sdk_ctx_open(int *events_id, int num_events, vendorp_ctx_t *ctx);
extern int rocprofiler_sdk_start(vendorp_ctx_t ctx);
extern int rocprofiler_sdk_stop(vendorp_ctx_t ctx);
extern int rocprofiler_sdk_ctx_read(vendorp_ctx_t ctx, long long **counters);
extern int rocprofiler_sdk_ctx_stop(vendorp_ctx_t ctx);
extern int rocprofiler_sdk_ctx_reset(vendorp_ctx_t ctx);
extern int rocprofiler_sdk_ctx_close(vendorp_ctx_t ctx);

extern int rocprofiler_sdk_evt_enum(unsigned int *event_code, int modifier);
extern int rocprofiler_sdk_evt_code_to_name(unsigned int event_code, char *name, int len);
extern int rocprofiler_sdk_evt_code_to_descr(unsigned int event_code, char *descr, int len);
extern int rocprofiler_sdk_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info);
extern int rocprofiler_sdk_evt_name_to_code(const char *name, unsigned int *event_code);

extern int rocprofiler_sdk_err_get_last(const char **err_string);


#endif
