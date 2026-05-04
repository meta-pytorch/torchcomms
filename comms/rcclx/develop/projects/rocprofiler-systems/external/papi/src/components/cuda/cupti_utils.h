/**
 * @file    cupti_utils.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_UTILS_H__
#define __CUPTI_UTILS_H__

#include <papi.h>

#include <nvperf_cuda_host.h> 

#include <stdint.h>

typedef int64_t cuptiu_bitmap_t;
typedef int (*cuptiu_dev_get_map_cb)(uint64_t event_id, int *dev_id);
typedef NVPW_CUDA_MetricsContext_Create_Params MCCP_t;

typedef struct event_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_HUGE_STR_LEN];
    cuptiu_bitmap_t device_map;
} cuptiu_event_t;

typedef struct gpu_record_s {
    char chip_name[PAPI_MIN_STR_LEN];
    MCCP_t *pmetricsContextCreateParams;
    int num_metrics;
    const char* const* metric_names;
} gpu_record_t;

typedef struct event_table_s {
    int count;
    unsigned int capacity;
    char cuda_evts[30][PAPI_2MAX_STR_LEN];
    int cuda_devs[30];
    int evt_pos[30];
    gpu_record_t *avail_gpu_info;
    cuptiu_event_t *events;
    void *htable;
} cuptiu_event_table_t;

/* These functions form a simple API to handle dynamic list of strings */
int cuptiu_event_table_create_init_capacity(int capacity, int sizeof_rec, cuptiu_event_table_t **pevt_table);
void cuptiu_event_table_destroy(cuptiu_event_table_t **pevt_table);

/* Utility to locate a file in a given path */
#define CUPTIU_MAX_FILES 100
int cuptiu_files_search_in_path(const char *file_name, const char *search_path, char **file_paths);

#endif  /* __CUPTI_UTILS_H__ */
