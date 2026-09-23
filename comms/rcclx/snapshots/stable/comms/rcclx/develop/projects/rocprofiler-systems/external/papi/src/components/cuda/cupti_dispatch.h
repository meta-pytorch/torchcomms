/**
 * @file    cupti_dispatch.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_DISPATCH_H__
#define __CUPTI_DISPATCH_H__

#include "cupti_utils.h"
#include "cupti_config.h"

extern unsigned int _cuda_lock;

typedef struct cuptip_control_s *cuptip_control_t;
typedef void *cuptid_info_t;
typedef cuptiu_event_table_t *ntv_event_table_t;
typedef cuptiu_event_t *ntv_event_t;

/* init and shutdown interfaces */
int cuptid_init(void);
int cuptid_shutdown(void);

/* native event interfaces */
int cuptid_evt_enum(uint32_t *event_code, int modifier);
int cuptid_evt_code_to_descr(uint32_t event_code, char *descr, int len);
int cuptid_evt_name_to_code(const char *name, uint32_t *event_code);
int cuptid_evt_code_to_name(uint32_t event_code, char *name, int len);
int cuptid_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptid_ctx_create(cuptid_info_t thread_info, cuptip_control_t *pcupti_ctl, uint32_t *events_id, int num_events);
int cuptid_ctx_start(cuptip_control_t ctl);
int cuptid_ctx_read(cuptip_control_t ctl, long long **counters);
int cuptid_ctx_reset(cuptip_control_t ctl);
int cuptid_ctx_stop(cuptip_control_t ctl);
int cuptid_ctx_destroy(cuptip_control_t *ctl);

/* thread interfaces */
int cuptid_thread_info_create(cuptid_info_t *info);
int cuptid_thread_info_destroy(cuptid_info_t *info);

/* misc. */
int cuptid_err_get_last(const char **error_str);
int cuptid_get_chip_name(int dev_num, char *name);
int cuptid_device_get_count(int *num_gpus);

#endif /* __CUPTI_DISPATCH_H__ */
