#include "vendor_common.h"

char error_string[PAPI_MAX_STR_LEN];
unsigned int _templ_lock;

device_table_t device_table;
device_table_t *device_table_p;

static int load_common_symbols(void);
static int unload_common_symbols(void);
static int initialize_device_table(void);
static int finalize_device_table(void);

int
vendorc_init(void)
{
    int papi_errno;

    papi_errno = load_common_symbols();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = initialize_device_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    device_table_p = &device_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_common_symbols();
    goto fn_exit;
}

int
vendorc_shutdown(void)
{
    finalize_device_table();
    device_table_p = NULL;
    unload_common_symbols();
    return PAPI_OK;
}

int
vendorc_err_get_last(const char **error)
{
    *error = error_string;
    return PAPI_OK;
}

int
load_common_symbols(void)
{
    return PAPI_OK;
}

int
unload_common_symbols(void)
{
    return PAPI_OK;
}

int
initialize_device_table(void)
{
#define MAX_DEVICE_COUNT (8)
    device_t *devices = papi_calloc(MAX_DEVICE_COUNT, sizeof(device_t));
    if (NULL == devices) {
        return PAPI_ENOMEM;
    }

    int i;
    for (i = 0; i < MAX_DEVICE_COUNT; ++i) {
        devices[i].id = (unsigned int) i;
    }

    device_table.devices = devices;
    device_table.num_devices = MAX_DEVICE_COUNT;

    return PAPI_OK;
}

int
finalize_device_table(void)
{
    papi_free(device_table_p->devices);
    device_table_p->num_devices = 0;
    return PAPI_OK;
}
