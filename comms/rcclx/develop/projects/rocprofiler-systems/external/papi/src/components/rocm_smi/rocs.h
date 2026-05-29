#ifndef __ROCS_H__
#define __ROCS_H__

#define ROCS_EVENTS_OPENED  (0x1)
#define ROCS_EVENTS_RUNNING (0x2)

typedef struct rocs_ctx *rocs_ctx_t;

/* init and shutdown interfaces */
int rocs_init(void);
int rocs_shutdown(void);

/* native event interfaces */
int rocs_evt_enum(unsigned int *event_code, int modifier);
int rocs_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int rocs_evt_name_to_code(const char *name, unsigned int *event_code);
int rocs_evt_code_to_name(unsigned int event_code, char *name, int len);

/* error handling interfaces */
int rocs_err_get_last(const char **err_string);

/* profiling context handling interfaces */
int rocs_ctx_open(unsigned int *event_ids, int num_events, rocs_ctx_t *ctx);
int rocs_ctx_close(rocs_ctx_t ctx);
int rocs_ctx_start(rocs_ctx_t ctx);
int rocs_ctx_stop(rocs_ctx_t ctx);
int rocs_ctx_read(rocs_ctx_t ctx, long long **counts);
int rocs_ctx_write(rocs_ctx_t ctx, long long *counts);
int rocs_ctx_reset(rocs_ctx_t ctx);

#endif /* End of __ROCS_H__ */
