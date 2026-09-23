/*
 * This file contains all the functions that are shared between
 * different vendor versions of the profiling library. This can
 * include vendor runtime functionalities.
 */
#ifndef __VENDOR_COMMON_H__
#define __VENDOR_COMMON_H__

#include <string.h>
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "vendor_config.h"

typedef struct {
    unsigned int id;
} device_t;

typedef struct {
    device_t *devices;
    int num_devices;
} device_table_t;

extern char error_string[PAPI_MAX_STR_LEN];
extern device_table_t *device_table_p;

int vendorc_init(void);
int vendorc_shutdown(void);
int vendorc_err_get_last(const char **error);

#endif
