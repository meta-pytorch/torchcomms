/**
 * @file    amd_gpu.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  Scan functions for AMD GPU subsystems.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "sysdetect.h"
#include "amd_gpu.h"

#ifdef HAVE_ROCM
#include "hsa.h"
#include "hsa_ext_amd.h"

static void *rocm_dlp = NULL;

static hsa_status_t (*hsa_initPtr)( void ) = NULL;
static hsa_status_t (*hsa_shut_downPtr)( void ) = NULL;
static hsa_status_t (*hsa_iterate_agentsPtr)( hsa_status_t (*)(hsa_agent_t agent,
                                                               void *value),
                                              void *value ) = NULL;
static hsa_status_t (*hsa_agent_get_infoPtr)( hsa_agent_t agent,
                                              hsa_agent_info_t attribute,
                                              void *value ) = NULL;
static hsa_status_t (*hsa_amd_agent_iterate_memory_poolsPtr)( hsa_agent_t agent,
                                                              hsa_status_t (*)(hsa_amd_memory_pool_t pool,
                                                                               void *value),
                                                              void *value ) = NULL;
static hsa_status_t (*hsa_amd_memory_pool_get_infoPtr)( hsa_amd_memory_pool_t pool,
                                                        hsa_amd_memory_pool_info_t attribute,
                                                        void *value ) = NULL;
static hsa_status_t (*hsa_status_stringPtr)( hsa_status_t status,
                                             const char **string ) = NULL;

#define ROCM_CALL(call, err_handle) do {   \
    hsa_status_t _status = (call);         \
    if (_status == HSA_STATUS_SUCCESS ||   \
        _status == HSA_STATUS_INFO_BREAK)  \
        break;                             \
    err_handle;                            \
} while(0)

static hsa_status_t count_devices( hsa_agent_t agent, void *data );
static hsa_status_t get_device_count( int *count );
static hsa_status_t get_device_memory( hsa_amd_memory_pool_t pool, void *info );
static hsa_status_t get_device_properties( hsa_agent_t agent, void *info );

static void fill_dev_info( _sysdetect_gpu_info_u *dev_info );
static int hsa_is_enabled( void );
static int load_hsa_sym( char *status );
static int unload_hsa_sym( void );
#endif /* HAVE_ROCM */

#ifdef HAVE_ROCM_SMI
#include "rocm_smi.h"

static void *rsmi_dlp = NULL;

static rsmi_status_t (*rsmi_initPtr)( unsigned long init_flags ) = NULL;
static rsmi_status_t (*rsmi_shut_downPtr)( void ) = NULL;
static rsmi_status_t (*rsmi_dev_pci_id_getPtr)( unsigned int dev_idx, unsigned long *bdfid ) = NULL;

#define ROCM_SMI_CALL(call, err_handle) do {    \
    rsmi_status_t _status = (call);             \
    if (_status == RSMI_STATUS_SUCCESS)         \
        break;                                  \
    err_handle;                                 \
} while(0)

static void fill_dev_affinity_info( _sysdetect_gpu_info_u *dev_info, int dev_count );
static int rsmi_is_enabled( void );
static int load_rsmi_sym( char *status );
static int unload_rsmi_sym( void );
#endif /* HAVE_ROCM_SMI */

#ifdef HAVE_ROCM
hsa_status_t
count_devices( hsa_agent_t agent, void *data )
{
    int *count = (int *) data;

    hsa_device_type_t type;
    ROCM_CALL((*hsa_agent_get_infoPtr)(agent, HSA_AGENT_INFO_DEVICE, &type),
              return _status);

    if (type == HSA_DEVICE_TYPE_GPU) {
        ++(*count);
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_count( int *count )
{
    *count = 0;

    ROCM_CALL((*hsa_iterate_agentsPtr)(&count_devices, count),
              return _status);

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_memory( hsa_amd_memory_pool_t pool, void *info )
{
    hsa_region_segment_t seg_info;
    _sysdetect_gpu_info_u *dev_info = info;

    ROCM_CALL((*hsa_amd_memory_pool_get_infoPtr)(pool,
                                                 HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                                 &seg_info),
              return _status);

    if (seg_info == HSA_REGION_SEGMENT_GROUP) {
        ROCM_CALL((*hsa_amd_memory_pool_get_infoPtr)(pool,
                                                     HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                     &dev_info->amd.max_shmmem_per_workgroup),
                  return _status);
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_device_properties( hsa_agent_t agent, void *info )
{
    static int count;

    hsa_device_type_t type;
    ROCM_CALL((*hsa_agent_get_infoPtr)(agent, HSA_AGENT_INFO_DEVICE, &type),
              return _status);

    if (type == HSA_DEVICE_TYPE_GPU) {
        /* query attributes for this device */
        _sysdetect_gpu_info_u *dev_info = &((_sysdetect_gpu_info_u *) info)[count];

        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_NAME,
                                           dev_info->amd.name),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_WAVEFRONT_SIZE,
                                           &dev_info->amd.wavefront_size),
                  return _status);
        unsigned short wg_dims[3];
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                           wg_dims),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                                           &dev_info->amd.max_threads_per_workgroup),
                  return _status);
        hsa_dim3_t gr_dims;
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_GRID_MAX_DIM,
                                           &gr_dims),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_VERSION_MAJOR,
                                           &dev_info->amd.major),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent,
                                           HSA_AGENT_INFO_VERSION_MINOR,
                                           &dev_info->amd.minor),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                           HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
                                           &dev_info->amd.simd_per_compute_unit),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                           HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                                           &dev_info->amd.compute_unit_count),
                  return _status);
        ROCM_CALL((*hsa_agent_get_infoPtr)(agent, (hsa_agent_info_t)
                                           HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                           &dev_info->amd.max_waves_per_compute_unit),
                  return _status);
        ROCM_CALL((*hsa_amd_agent_iterate_memory_poolsPtr)(agent,
                                                           &get_device_memory,
                                                           dev_info),
                  return _status);

        dev_info->amd.max_workgroup_dim_x = wg_dims[0];
        dev_info->amd.max_workgroup_dim_y = wg_dims[1];
        dev_info->amd.max_workgroup_dim_z = wg_dims[2];
        dev_info->amd.max_grid_dim_x  = gr_dims.x;
        dev_info->amd.max_grid_dim_y  = gr_dims.y;
        dev_info->amd.max_grid_dim_z  = gr_dims.z;

        ++count;
    }

    return HSA_STATUS_SUCCESS;
}

void
fill_dev_info( _sysdetect_gpu_info_u *dev_info )
{
    hsa_status_t status = HSA_STATUS_SUCCESS;
    const char *string = NULL;

    ROCM_CALL((*hsa_iterate_agentsPtr)(&get_device_properties, dev_info),
             status = _status);

    if (status != HSA_STATUS_SUCCESS) {
        (*hsa_status_stringPtr)(status, &string);
        SUBDBG( "error: %s\n", string );
    }
}

int
hsa_is_enabled( void )
{
    return (hsa_initPtr                           != NULL &&
            hsa_shut_downPtr                      != NULL &&
            hsa_iterate_agentsPtr                 != NULL &&
            hsa_agent_get_infoPtr                 != NULL &&
            hsa_amd_agent_iterate_memory_poolsPtr != NULL &&
            hsa_amd_memory_pool_get_infoPtr       != NULL &&
            hsa_status_stringPtr                  != NULL);
}

int
load_hsa_sym( char *status )
{
    char pathname[PATH_MAX] = "libhsa-runtime64.so";
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root != NULL) {
        sprintf(pathname, "%s/lib/libhsa-runtime64.so", rocm_root);
    }

    rocm_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rocm_dlp == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    hsa_initPtr                           = dlsym(rocm_dlp, "hsa_init");
    hsa_shut_downPtr                      = dlsym(rocm_dlp, "hsa_shut_down");
    hsa_iterate_agentsPtr                 = dlsym(rocm_dlp, "hsa_iterate_agents");
    hsa_agent_get_infoPtr                 = dlsym(rocm_dlp, "hsa_agent_get_info");
    hsa_amd_agent_iterate_memory_poolsPtr = dlsym(rocm_dlp, "hsa_amd_agent_iterate_memory_pools");
    hsa_amd_memory_pool_get_infoPtr       = dlsym(rocm_dlp, "hsa_amd_memory_pool_get_info");
    hsa_status_stringPtr                  = dlsym(rocm_dlp, "hsa_status_string");

    if (!hsa_is_enabled() || (*hsa_initPtr)()) {
        const char *message = "dlsym() of HSA symbols failed or hsa_init() "
                              "failed";
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_hsa_sym( void )
{
    if (rocm_dlp != NULL) {
        (*hsa_shut_downPtr)();
        dlclose(rocm_dlp);
    }

    hsa_initPtr                           = NULL;
    hsa_shut_downPtr                      = NULL;
    hsa_iterate_agentsPtr                 = NULL;
    hsa_agent_get_infoPtr                 = NULL;
    hsa_amd_agent_iterate_memory_poolsPtr = NULL;
    hsa_amd_memory_pool_get_infoPtr       = NULL;
    hsa_status_stringPtr                  = NULL;

    return hsa_is_enabled();
}
#endif /* HAVE_ROCM */

#ifdef HAVE_ROCM_SMI
void
fill_dev_affinity_info( _sysdetect_gpu_info_u *info, int dev_count )
{
    int dev;
    for (dev = 0; dev < dev_count; ++dev) {
        unsigned long uid;
        ROCM_SMI_CALL((*rsmi_dev_pci_id_getPtr)(dev, &uid), return);

        _sysdetect_gpu_info_u *dev_info = &info[dev];
        dev_info->amd.uid = uid;
    }
}

int
rsmi_is_enabled( void )
{
    return (rsmi_initPtr           != NULL &&
            rsmi_shut_downPtr      != NULL &&
            rsmi_dev_pci_id_getPtr != NULL);
}

int
load_rsmi_sym( char *status )
{
    char pathname[PATH_MAX] = "librocm_smi64.so";
    char *rsmi_root = getenv("PAPI_ROCM_ROOT");
    if (rsmi_root != NULL) {
        sprintf(pathname, "%s/lib/librocm_smi64.so", rsmi_root);
    }

    rsmi_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rsmi_dlp == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    rsmi_initPtr           = dlsym(rsmi_dlp, "rsmi_init");
    rsmi_shut_downPtr      = dlsym(rsmi_dlp, "rsmi_shut_down");
    rsmi_dev_pci_id_getPtr = dlsym(rsmi_dlp, "rsmi_dev_pci_id_get");

    if (!rsmi_is_enabled() || (*rsmi_initPtr)(0)) {
        const char *message = "dlsym() of RSMI symbols failed or rsmi_init() "
                              "failed";
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_rsmi_sym( void )
{
    if (rsmi_dlp != NULL) {
        (*rsmi_shut_downPtr)();
        dlclose(rsmi_dlp);
    }

    rsmi_initPtr           = NULL;
    rsmi_shut_downPtr      = NULL;
    rsmi_dev_pci_id_getPtr = NULL;

    return rsmi_is_enabled();
}
#endif /* HAVE_ROCM_SMI */

void
open_amd_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    memset(dev_type_info, 0, sizeof(*dev_type_info));
    dev_type_info->id = PAPI_DEV_TYPE_ID__ROCM;
    strcpy(dev_type_info->vendor, "AMD/ATI");
    strcpy(dev_type_info->status, "Device Initialized");

#ifdef HAVE_ROCM
    if (load_hsa_sym(dev_type_info->status)) {
        return;
    }

    int dev_count = 0;
    hsa_status_t status = get_device_count(&dev_count);
    if (status != HSA_STATUS_SUCCESS) {
        if (status != HSA_STATUS_ERROR_NOT_INITIALIZED) {
            const char *string;
            (*hsa_status_stringPtr)(status, &string);
            printf( "error: %s\n", string );
        }
        return;
    }
    dev_type_info->num_devices = dev_count;

    _sysdetect_gpu_info_u *arr = papi_calloc(dev_count, sizeof(*arr));
    fill_dev_info(arr);

#ifdef HAVE_ROCM_SMI
    if (!load_rsmi_sym(dev_type_info->status)) {
        fill_dev_affinity_info(arr, dev_count);
        unload_rsmi_sym();
    }
#else
    const char *message = "RSMI not configured, no device affinity available";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Error message truncated.");
    }
#endif /* HAVE_ROCM_SMI */

    unload_hsa_sym();
    dev_type_info->dev_info_arr = (_sysdetect_dev_info_u *)arr;
#else
    const char *message = "ROCm not configured, no ROCm device available";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Error message truncated.");
    }
#endif /* HAVE_ROCM */
}

void
close_amd_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    papi_free(dev_type_info->dev_info_arr);
}
