#include <dlfcn.h>
#include <string.h>
#include "papi.h"
#include "papi_memory.h"
#include "roc_common.h"

/* hsa function pointers */
hsa_status_t (*hsa_init_p)(void);
hsa_status_t (*hsa_shut_down_p)(void);
hsa_status_t (*hsa_iterate_agents_p)(hsa_status_t (*)(hsa_agent_t, void *), void *);
hsa_status_t (*hsa_system_get_info_p)(hsa_system_info_t, void *);
hsa_status_t (*hsa_agent_get_info_p)(hsa_agent_t, hsa_agent_info_t, void *);
hsa_status_t (*hsa_queue_destroy_p)(hsa_queue_t *);
hsa_status_t (*hsa_status_string_p)(hsa_status_t, const char **);

static void *hsa_dlp;
char error_string[PAPI_MAX_STR_LEN];
static device_table_t device_table;
device_table_t *device_table_p;
static rocc_bitmap_t global_device_map;

static int load_hsa_sym(void);
static int unload_hsa_sym(void);
static int init_device_table(void);
static void init_thread_id_fn(void);
static unsigned long (*thread_id_fn)(void);

int
rocc_init(void)
{
    int papi_errno = load_hsa_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    hsa_status_t status = hsa_init_p();
    if (status != HSA_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = init_device_table();
    if (papi_errno != PAPI_OK) {
        (*hsa_shut_down_p)();
        goto fn_fail;
    }

    device_table_p = &device_table;
    init_thread_id_fn();

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_hsa_sym();
    goto fn_exit;
}

int
rocc_shutdown(void)
{
    hsa_shut_down_p();
    unload_hsa_sym();
    return PAPI_OK;
}

int
rocc_err_get_last(const char **err_string)
{
    *err_string = error_string;
    return PAPI_OK;
}

int
rocc_dev_get_map(rocc_dev_get_map_cb query_dev_id, uint64_t *events_id, int num_events, rocc_bitmap_t *bitmap)
{
    int i;
    rocc_bitmap_t device_map_acq = 0;

    for (i = 0; i < num_events; ++i) {
        int dev_id;
        if (query_dev_id(events_id[i], &dev_id)) {
            return PAPI_EMISC;
        }

        device_map_acq |= (1 << dev_id);
    }

    *bitmap = device_map_acq;
    return PAPI_OK;
}

int
rocc_dev_acquire(rocc_bitmap_t bitmap)
{
    rocc_bitmap_t device_map_acq = bitmap;

    if (device_map_acq & global_device_map) {
        return PAPI_EINVAL;
    }
    global_device_map |= device_map_acq;

    return PAPI_OK;
}

int
rocc_dev_release(rocc_bitmap_t bitmap)
{
    rocc_bitmap_t device_map_rel = bitmap;

    if ((device_map_rel & global_device_map) != device_map_rel) {
        return PAPI_EINVAL;
    }
    global_device_map &= ~device_map_rel;

    return PAPI_OK;
}

static int dev_get_count(rocc_bitmap_t bitmap, int *num_devices);

int
rocc_dev_get_count(rocc_bitmap_t bitmap, int *num_devices)
{
    return dev_get_count(bitmap, num_devices);
}

int
dev_get_count(rocc_bitmap_t bitmap, int *num_devices)
{
    *num_devices = 0;

    while (bitmap) {
        bitmap -= bitmap & (~bitmap + 1);
        ++(*num_devices);
    }

    return PAPI_OK;
}

int
rocc_dev_get_id(rocc_bitmap_t bitmap, int dev_count, int *device_id)
{
    int count = 0;

    dev_get_count(bitmap, &count);
    if (dev_count >= count) {
        return PAPI_EMISC;
    }

    count = 0;
    rocc_bitmap_t lsb = 0;
    while (bitmap) {
        lsb = bitmap & (~bitmap + 1);
        bitmap -= lsb;
        if (count++ == dev_count) {
            break;
        }
    }

    *device_id = 0;
    while (!(lsb & 0x1)) {
        ++(*device_id);
        lsb >>= 1;
    }

    return PAPI_OK;
}

int
rocc_dev_get_agent_id(hsa_agent_t agent, int *dev_id)
{
    for (*dev_id = 0; *dev_id < device_table_p->count; ++(*dev_id)) {
        if (memcmp(&device_table_p->devices[*dev_id], &agent, sizeof(agent)) == 0) {
            break;
        }
    }
    return PAPI_OK;
}

int
rocc_dev_set(rocc_bitmap_t *bitmap, int i)
{
    *bitmap |= (1ULL << i);
    return PAPI_OK;
}

int
rocc_dev_check(rocc_bitmap_t bitmap, int i)
{
    return (bitmap & (1ULL << i));
}

int
rocc_thread_get_id(unsigned long *tid)
{
    *tid = thread_id_fn();
    return PAPI_OK;
}

int
load_hsa_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = { 0 };
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root == NULL) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Can't load libhsa-runtime64.so, PAPI_ROCM_ROOT not set.");
        goto fn_fail;
    }

    sprintf(pathname, "%s/lib/libhsa-runtime64.so", rocm_root);

    hsa_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (hsa_dlp == NULL) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "%s", dlerror());
        goto fn_fail;
    }

    hsa_init_p            = dlsym(hsa_dlp, "hsa_init");
    hsa_shut_down_p       = dlsym(hsa_dlp, "hsa_shut_down");
    hsa_iterate_agents_p  = dlsym(hsa_dlp, "hsa_iterate_agents");
    hsa_system_get_info_p = dlsym(hsa_dlp, "hsa_system_get_info");
    hsa_agent_get_info_p  = dlsym(hsa_dlp, "hsa_agent_get_info");
    hsa_queue_destroy_p   = dlsym(hsa_dlp, "hsa_queue_destroy");
    hsa_status_string_p   = dlsym(hsa_dlp, "hsa_status_string");

    int hsa_not_initialized = (!hsa_init_p            ||
                               !hsa_shut_down_p       ||
                               !hsa_iterate_agents_p  ||
                               !hsa_system_get_info_p ||
                               !hsa_agent_get_info_p  ||
                               !hsa_queue_destroy_p   ||
                               !hsa_status_string_p);

    papi_errno = (hsa_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Error while loading hsa symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

int
unload_hsa_sym(void)
{
    if (hsa_dlp == NULL) {
        return PAPI_OK;
    }

    hsa_init_p            = NULL;
    hsa_shut_down_p       = NULL;
    hsa_iterate_agents_p  = NULL;
    hsa_system_get_info_p = NULL;
    hsa_agent_get_info_p  = NULL;
    hsa_queue_destroy_p   = NULL;
    hsa_status_string_p   = NULL;

    dlclose(hsa_dlp);

    return PAPI_OK;
}

static hsa_status_t get_agent_handle_cb(hsa_agent_t, void *);

int
init_device_table(void)
{
    int papi_errno = PAPI_OK;

    hsa_status_t hsa_errno = hsa_iterate_agents_p(get_agent_handle_cb, &device_table);
    if (hsa_errno != HSA_STATUS_SUCCESS) {
        const char *error_string_p;
        hsa_status_string_p(hsa_errno, &error_string_p);
        snprintf(error_string, PAPI_MAX_STR_LEN, "%s", error_string_p);
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC;
    device_table.count = 0;
    goto fn_exit;
}

hsa_status_t
get_agent_handle_cb(hsa_agent_t agent, void *device_table)
{
    hsa_device_type_t type;
    device_table_t *device_table_ = (device_table_t *) device_table;

    hsa_status_t hsa_errno = hsa_agent_get_info_p(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (hsa_errno != HSA_STATUS_SUCCESS) {
        return hsa_errno;
    }

    if (type == HSA_DEVICE_TYPE_GPU) {
        assert(device_table_->count < PAPI_ROCM_MAX_DEV_COUNT);
        device_table_->devices[device_table_->count] = agent;
        ++device_table_->count;
    }

    return HSA_STATUS_SUCCESS;
}

void
init_thread_id_fn(void)
{
    if (thread_id_fn) {
        return;
    }

    thread_id_fn = (_papi_hwi_thread_id_fn) ?
        _papi_hwi_thread_id_fn : _papi_getpid;
}
