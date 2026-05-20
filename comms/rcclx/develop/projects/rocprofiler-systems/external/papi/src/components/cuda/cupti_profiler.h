/**
 * @file    cupti_profiler.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.) 
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_PROFILER_H__
#define __CUPTI_PROFILER_H__

#include "cupti_utils.h"

typedef struct cuptip_control_s     *cuptip_control_t;

/* used to determine collection method in cupti_profiler.c, see cuptip_ctx_read */
#define CUDA_AVG 0x1
#define CUDA_MAX 0x2
#define CUDA_MIN 0x3
#define CUDA_SUM 0x4
#define CUDA_DEFAULT 0x5

/* init and shutdown interfaces */
int cuptip_init(void);
int cuptip_shutdown(void);

/* native event interfaces */
int cuptip_evt_enum(uint64_t *event_code, int modifier);
int cuptip_evt_code_to_descr(uint64_t event_code, char *descr, int len);
int cuptip_evt_name_to_code(const char *name, uint64_t *event_code);
int cuptip_evt_code_to_name(uint64_t event_code, char *name, int len);
int cuptip_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptip_ctx_create(cuptic_info_t thr_info, cuptip_control_t *pstate,  uint64_t *events_id, int num_events);
int cuptip_ctx_destroy(cuptip_control_t *pstate);
int cuptip_ctx_start(cuptip_control_t state);
int cuptip_ctx_stop(cuptip_control_t state);
int cuptip_ctx_read(cuptip_control_t state, long long **counters);
int cuptip_ctx_reset(cuptip_control_t state);

#endif  /* __CUPTI_PROFILER_H__ */
