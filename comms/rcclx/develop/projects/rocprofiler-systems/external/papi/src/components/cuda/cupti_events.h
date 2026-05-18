/**
 * @file    cupti_events.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

#include "cupti_utils.h"

#include <stdint.h>

typedef void *cuptie_control_t;

/* init and shutdown interfaces */
int cuptie_init(void);
int cuptie_shutdown(void);

/* native event interfaces */
int cuptie_evt_enum(uint64_t *event_code, int modifier); 
int cuptie_evt_code_to_descr(uint64_t event_code, char *descr, int len);
int cuptie_evt_name_to_code(const char *name, uint64_t *event_code);
int cuptie_evt_code_to_name(uint64_t event_code, char *name, int len);
int cuptie_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptie_ctx_create(void *thr_info, cuptie_control_t *pctl);
int cuptie_ctx_start(cuptie_control_t ctl);
int cuptie_ctx_read(cuptie_control_t ctl, long long **counters);
int cuptie_ctx_reset(cuptie_control_t ctl);
int cuptie_ctx_stop(cuptie_control_t ctl);
int cuptie_ctx_destroy(cuptie_control_t *pctl);

#endif  /* __CUPTI_EVENTS_H__ */
