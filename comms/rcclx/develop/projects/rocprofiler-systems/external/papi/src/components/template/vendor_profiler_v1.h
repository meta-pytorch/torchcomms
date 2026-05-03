#ifndef __VENDOR_PROFILER_V1_H__
#define __VENDOR_PROFILER_V1_H__

typedef struct vendord_ctx *vendorp_ctx_t;

int vendorp1_init_pre(void);
int vendorp1_init(void);
int vendorp1_shutdown(void);

int vendorp1_ctx_open(unsigned int *events_id, int num_events, vendorp_ctx_t *ctx);
int vendorp1_ctx_start(vendorp_ctx_t ctx);
int vendorp1_ctx_read(vendorp_ctx_t ctx, long long **counters);
int vendorp1_ctx_stop(vendorp_ctx_t ctx);
int vendorp1_ctx_reset(vendorp_ctx_t ctx);
int vendorp1_ctx_close(vendorp_ctx_t ctx);

int vendorp1_evt_enum(unsigned int *event_code, int modifier);
int vendorp1_evt_code_to_name(unsigned int event_code, char *name, int len);
int vendorp1_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int vendorp1_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info);
int vendorp1_evt_name_to_code(const char *name, unsigned int *event_code);

#endif
