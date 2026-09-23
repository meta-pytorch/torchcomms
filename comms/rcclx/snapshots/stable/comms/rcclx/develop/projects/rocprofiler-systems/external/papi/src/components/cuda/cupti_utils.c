/**
 * @file    cupti_utils.c
 *
 * @author  Treece Burges  tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)s
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include <string.h>
#include "papi_memory.h"

#include "cupti_utils.h"
#include "htable.h"
#include "lcuda_debug.h"

#define ADDED_EVENTS_INITIAL_CAPACITY 64

int cuptiu_event_table_create_init_capacity(int capacity, int sizeof_rec, cuptiu_event_table_t **pevt_table)
{
    cuptiu_event_table_t *evt_table = (cuptiu_event_table_t *) malloc(sizeof(cuptiu_event_table_t));
    if (evt_table == NULL) {
        goto fn_fail;
    }

    evt_table->capacity = capacity;
    evt_table->count = 0;
    evt_table->event_stats_count = 0;
    
    if (htable_init(&(evt_table->htable)) != HTABLE_SUCCESS) {
        cuptiu_event_table_destroy(&evt_table);
        goto fn_fail;
    }
    
    *pevt_table = evt_table;
    return 0;
fn_fail:
    *pevt_table = NULL;
    return PAPI_ENOMEM;
}

void cuptiu_event_table_destroy(cuptiu_event_table_t **pevt_table)
{
    cuptiu_event_table_t *evt_table = *pevt_table;
    if (evt_table == NULL)
        return;

    if (evt_table->htable) {
        htable_shutdown(evt_table->htable);
        evt_table->htable = NULL;
    }

    free(evt_table);
    *pevt_table = NULL;
}

int cuptiu_files_search_in_path(const char *file_name, const char *search_path, char **file_paths)
{
    char path[PATH_MAX];
    char command[PATH_MAX];
    snprintf(command, PATH_MAX, "find %s -name %s", search_path, file_name);

    FILE *fp;
    fp = popen(command, "r");
    if (fp == NULL) {
        ERRDBG("Failed to run system command find using popen.\n");
        return -1;
    }

    int count = 0;
    while (fgets(path, PATH_MAX, fp) != NULL) {
        path[strcspn(path, "\n")] = 0;
        file_paths[count] = strdup(path);
        count++;
        if (count >= CUPTIU_MAX_FILES) {
            break;
        }
    }

    pclose(fp);
    if (count == 0) {
        ERRDBG("%s not found in path PAPI_CUDA_ROOT.\n", file_name);
    }
    return count;
}

// Initialize the stat Stringvector
void init_vector(StringVector *vec) {
    vec->arrayMetricStatistics = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

// Add a string to the vector 
int push_back(StringVector *vec, const char *str) {
    size_t i;
    for (i = 0; i < vec->size; i++) {
      if (strcmp(vec->arrayMetricStatistics[i], str) == 0) {
          return PAPI_OK;        
      }
    }

    // Resize if necessary
    if (vec->size == vec->capacity) {
        size_t new_capacity = (vec->capacity == 0) ? 4 : vec->capacity * 2;
        char **new_data = realloc(vec->arrayMetricStatistics, new_capacity * sizeof(char*));
        if (new_data == NULL) {
            ERRDBG ("Memory allocation failed\n");
            return PAPI_ENOMEM;
        }
        vec->arrayMetricStatistics = new_data;
        vec->capacity = new_capacity;
    }

    // Allocate memory for the new string and copy it
    vec->arrayMetricStatistics[vec->size] = malloc(strlen(str) + 1); 
    if (vec->arrayMetricStatistics[vec->size] == NULL) {
        ERRDBG ("Memory allocation failed\n");
        return PAPI_ENOMEM;
    }
    int strLen = snprintf(vec->arrayMetricStatistics[vec->size], strlen(str) + 1, "%s", str);
    if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            SUBDBG("Failed to fully write added Cuda native event name.\n");
            return PAPI_ENOMEM;
    }
    
    vec->size++; // Increase the size
    return PAPI_OK;
}

// Free the memory used by the vector
void free_vector(StringVector *vec) {
    for (size_t i = 0; i < vec->size; i++) {
        free(vec->arrayMetricStatistics[i]); 
    }
    free(vec->arrayMetricStatistics); 
    vec->arrayMetricStatistics = NULL;
}
