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
#include <cuda.h>

#include <stdint.h>

typedef int64_t cuptiu_bitmap_t;
typedef int (*cuptiu_dev_get_map_cb)(uint64_t event_id, int *dev_id);

typedef struct gpu_record_s {
    char chipName[PAPI_MIN_STR_LEN];
    int totalMetricCount;
    char **metricNames;
} gpu_record_t;

typedef struct {
    char **arrayMetricStatistics ;   
    size_t size;   
    size_t capacity;
} StringVector;

typedef struct event_record_s {
    char name[PAPI_HUGE_STR_LEN];
    char basenameWithStatReplaced[PAPI_HUGE_STR_LEN];
    char desc[PAPI_HUGE_STR_LEN];
    StringVector * stat;
    cuptiu_bitmap_t device_map;
} cuptiu_event_t;

typedef struct event_table_s {
    unsigned int count;
    unsigned int event_stats_count;
    unsigned int capacity;
    char cuda_evts[30][PAPI_HUGE_STR_LEN];
    int cuda_devs[30];
    int evt_pos[30];
    gpu_record_t *avail_gpu_info;
    cuptiu_event_t *events;
    StringVector   *event_stats;
    void *htable;
} cuptiu_event_table_t;

void init_vector(StringVector *vec);
int push_back(StringVector *vec, const char *str);
void free_vector(StringVector *vec);

/* These functions form a simple API to handle dynamic list of strings */
int cuptiu_event_table_create_init_capacity(int capacity, int sizeof_rec, cuptiu_event_table_t **pevt_table);
void cuptiu_event_table_destroy(cuptiu_event_table_t **pevt_table);

/* Utility to locate a file in a given path */
#define CUPTIU_MAX_FILES 100
int cuptiu_files_search_in_path(const char *file_name, const char *search_path, char **file_paths);

#endif  /* __CUPTI_UTILS_H__ */
