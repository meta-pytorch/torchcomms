#ifndef __VENDOR_DISPATCH_H__
#define __VENDOR_DISPATCH_H__

#include "vendor_config.h"

typedef struct vendord_ctx *vendord_ctx_t;

int vendord_init_pre(void);
int vendord_init(void);
int vendord_shutdown(void);

int vendord_ctx_open(unsigned int *events_id, int num_events, vendord_ctx_t *ctx);
int vendord_ctx_start(vendord_ctx_t ctx);
int vendord_ctx_read(vendord_ctx_t ctx, long long **counters);
int vendord_ctx_stop(vendord_ctx_t ctx);
int vendord_ctx_reset(vendord_ctx_t ctx);
int vendord_ctx_close(vendord_ctx_t ctx);

int vendord_err_get_last(const char **error);

int vendord_evt_enum(unsigned int *event_code, int modifier);
int vendord_evt_code_to_name(unsigned int event_code, char *name, int len);
int vendord_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int vendord_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info);
int vendord_evt_name_to_code(const char *name, unsigned int *event_code);

#endif
