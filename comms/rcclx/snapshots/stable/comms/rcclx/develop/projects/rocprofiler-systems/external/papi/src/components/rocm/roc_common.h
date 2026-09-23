#ifndef __ROC_COMMON_H__
#define __ROC_COMMON_H__

#include <hsa.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "roc_profiler_config.h"

#ifndef PAPI_ROCM_MAX_DEV_COUNT
#define PAPI_ROCM_MAX_DEV_COUNT (32)
#endif

typedef int64_t rocc_bitmap_t;
typedef int (*rocc_dev_get_map_cb)(uint64_t event_id, int *dev_id);

typedef struct {
    hsa_agent_t devices[PAPI_ROCM_MAX_DEV_COUNT];
    int count;
} device_table_t;

extern hsa_status_t (*hsa_init_p)(void);
extern hsa_status_t (*hsa_shut_down_p)(void);
extern hsa_status_t (*hsa_iterate_agents_p)(hsa_status_t (*)(hsa_agent_t, void *), void *);
extern hsa_status_t (*hsa_system_get_info_p)(hsa_system_info_t, void *);
extern hsa_status_t (*hsa_agent_get_info_p)(hsa_agent_t, hsa_agent_info_t, void *);
extern hsa_status_t (*hsa_queue_destroy_p)(hsa_queue_t *);
extern hsa_status_t (*hsa_status_string_p)(hsa_status_t, const char **);

extern char error_string[PAPI_MAX_STR_LEN];
extern device_table_t *device_table_p;

int rocc_init(void);
int rocc_shutdown(void);
int rocc_err_get_last(const char **error_string);
int rocc_dev_get_map(rocc_dev_get_map_cb cb, uint64_t *events_id, int num_events, rocc_bitmap_t *bitmap);
int rocc_dev_acquire(rocc_bitmap_t bitmap);
int rocc_dev_release(rocc_bitmap_t bitmap);
int rocc_dev_get_count(rocc_bitmap_t bitmap, int *num_devices);
int rocc_dev_get_id(rocc_bitmap_t bitmap, int dev_count, int *device_id);
int rocc_dev_get_agent_id(hsa_agent_t agent, int *dev_id);
int rocc_dev_set(rocc_bitmap_t *bitmap, int i);
int rocc_dev_check(rocc_bitmap_t bitmap, int i);

int rocc_thread_get_id(unsigned long *tid);

#endif /* End of __ROC_COMMON_H__ */
