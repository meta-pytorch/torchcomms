/**
 * @file    amds.h
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#ifndef __AMDS_H__
#define __AMDS_H__

#include "papi.h"

#define AMDS_EVENTS_OPENED  (0x1)
#define AMDS_EVENTS_RUNNING (0x2)

typedef struct amds_ctx *amds_ctx_t;

/* initialization and shutdown */
int amds_init(void);
int amds_shutdown(void);

/* native event queries */
int amds_evt_enum(unsigned int *EventCode, int modifier);
int amds_evt_code_to_descr(unsigned int EventCode, char *descr, int len);
int amds_evt_name_to_code(const char *name, unsigned int *EventCode);
int amds_evt_code_to_name(unsigned int EventCode, char *name, int len);
int amds_evt_code_to_info(unsigned int EventCode, PAPI_event_info_t *info);

/* error handling */
int amds_err_get_last(const char **err_string);

/* profiling context operations */
int amds_ctx_open(unsigned int *event_ids, int num_events, amds_ctx_t *ctx);
int amds_ctx_close(amds_ctx_t ctx);
int amds_ctx_start(amds_ctx_t ctx);
int amds_ctx_stop(amds_ctx_t ctx);
int amds_ctx_read(amds_ctx_t ctx, long long **counts);
int amds_ctx_write(amds_ctx_t ctx, long long *counts);
int amds_ctx_reset(amds_ctx_t ctx);

#endif /* __AMDS_H__ */
