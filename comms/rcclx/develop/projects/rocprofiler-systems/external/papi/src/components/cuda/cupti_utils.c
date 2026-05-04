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
    cuptiu_event_table_t *evt_table = (cuptiu_event_table_t *) papi_malloc(sizeof(cuptiu_event_table_t));
    if (evt_table == NULL) {
        goto fn_fail;
    }

    evt_table->capacity = capacity;
    evt_table->count = 0;
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
    papi_free(evt_table);
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
