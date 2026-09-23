/**
 * @file    nvidia_gpu.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  Scan functions for NVIDIA GPU subsystems.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "sysdetect.h"
#include "nvidia_gpu.h"

#ifdef HAVE_CUDA
#include "cuda.h"

static void *cuda_dlp = NULL;

static CUresult (*cuInitPtr)( unsigned int flags ) = NULL;
static CUresult (*cuDeviceGetPtr)( CUdevice *device, int ordinal ) = NULL;
static CUresult (*cuDeviceGetNamePtr)( char *name, int len, CUdevice dev ) = NULL;
static CUresult (*cuDeviceGetCountPtr)( int *count ) = NULL;
static CUresult (*cuDeviceGetAttributePtr)( int *pi, CUdevice_attribute attrib,
                                            CUdevice dev ) = NULL;
static CUresult (*cuDeviceGetPCIBusIdPtr)( char *bus_id_string, int len,
                                           CUdevice dev ) = NULL;

#define CU_CALL(call, err_handle) do {                                          \
    CUresult _status = (call);                                                  \
    if (_status != CUDA_SUCCESS) {                                              \
        if (_status == CUDA_ERROR_NOT_INITIALIZED) {                            \
            if ((*cuInitPtr)(0) == CUDA_SUCCESS) {                              \
                _status = (call);                                               \
                if (_status == CUDA_SUCCESS) {                                  \
                    break;                                                      \
                }                                                               \
            }                                                                   \
        }                                                                       \
        SUBDBG("error: function %s failed with error %d.\n", #call, _status);   \
        err_handle;                                                             \
    }                                                                           \
} while(0)

static void fill_dev_info( _sysdetect_gpu_info_u *dev_info, int dev );
static int cuda_is_enabled( void );
static int load_cuda_sym( char *status );
static int unload_cuda_sym( void );
#endif /* HAVE_CUDA */

#ifdef HAVE_NVML
#include "nvml.h"

static void *nvml_dlp = NULL;

static nvmlReturn_t (*nvmlInitPtr)( void );
static nvmlReturn_t (*nvmlDeviceGetCountPtr)( unsigned int *deviceCount ) = NULL;
static nvmlReturn_t (*nvmlDeviceGetHandleByPciBusIdPtr)( const char *bus_id_str,
                                                         nvmlDevice_t *device ) = NULL;
static nvmlReturn_t (*nvmlDeviceGetUUIDPtr)( nvmlDevice_t device, char *uuid,
                                             unsigned int length ) = NULL;

#define NVML_CALL(call, err_handle) do {                                        \
    nvmlReturn_t _status = (call);                                              \
    if (_status != NVML_SUCCESS) {                                              \
        if (_status == NVML_ERROR_UNINITIALIZED) {                              \
            if ((*nvmlInitPtr)() == NVML_SUCCESS) {                             \
                _status = (call);                                               \
                if (_status == NVML_SUCCESS) {                                  \
                    break;                                                      \
                }                                                               \
            }                                                                   \
        }                                                                       \
        SUBDBG("error: function %s failed with error %d.\n", #call, _status);   \
        err_handle;                                                             \
    }                                                                           \
} while(0)

static void fill_dev_affinity_info( _sysdetect_gpu_info_u *dev_info, int dev_count );
static int nvml_is_enabled( void );
static int load_nvml_sym( char *status );
static int unload_nvml_sym( void );
static unsigned long hash(unsigned char *str);
#endif /* HAVE_NVML */

#ifdef HAVE_CUDA
void
fill_dev_info( _sysdetect_gpu_info_u *dev_info, int dev )
{
    CUdevice device;
    CU_CALL((*cuDeviceGetPtr)(&device,
                              dev),
            return);
    CU_CALL((*cuDeviceGetNamePtr)(dev_info->nvidia.name,
                                  PAPI_2MAX_STR_LEN,
                                  device),
            dev_info->nvidia.name[0] = '\0');
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.warp_size,
                                       CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                       device),
            dev_info->nvidia.warp_size = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_shmmem_per_block,
                                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                       device),
            dev_info->nvidia.max_shmmem_per_block = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_shmmem_per_multi_proc,
                                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                       device),
            dev_info->nvidia.max_shmmem_per_multi_proc = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_block_dim_x,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                       device),
            dev_info->nvidia.max_block_dim_x = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_block_dim_y,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                       device),
            dev_info->nvidia.max_block_dim_y = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_block_dim_z,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                                       device),
            dev_info->nvidia.max_block_dim_z = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_grid_dim_x,
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                       device),
            dev_info->nvidia.max_grid_dim_x = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_grid_dim_y,
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                       device),
            dev_info->nvidia.max_grid_dim_y = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_grid_dim_z,
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                                       device),
            dev_info->nvidia.max_grid_dim_z = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_threads_per_block,
                                       CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                       device),
            dev_info->nvidia.max_threads_per_block = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.multi_processor_count,
                                       CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                       device),
            dev_info->nvidia.multi_processor_count = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.multi_kernel_per_ctx,
                                       CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                                       device),
            dev_info->nvidia.multi_kernel_per_ctx = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.can_map_host_mem,
                                       CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
                                       device),
            dev_info->nvidia.can_map_host_mem = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.can_overlap_comp_and_data_xfer,
                                       CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                                       device),
            dev_info->nvidia.can_overlap_comp_and_data_xfer = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.unified_addressing,
                                       CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                                       device),
            dev_info->nvidia.unified_addressing = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.managed_memory,
                                       CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                                       device),
            dev_info->nvidia.managed_memory = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.major,
                                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                       device),
            dev_info->nvidia.major = -1);
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.minor,
                                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                       device),
            dev_info->nvidia.minor = -1);

#if CUDA_VERSION >= 11000
    CU_CALL((*cuDeviceGetAttributePtr)(&dev_info->nvidia.max_blocks_per_multi_proc,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                                       device),
            dev_info->nvidia.max_blocks_per_multi_proc = -1);
#else
    dev_info->nvidia.max_blocks_per_multi_proc = -1;
#endif /* CUDA_VERSION */
}

int
cuda_is_enabled( void )
{
    return (cuInitPtr               != NULL &&
            cuDeviceGetPtr          != NULL &&
            cuDeviceGetNamePtr      != NULL &&
            cuDeviceGetCountPtr     != NULL &&
            cuDeviceGetAttributePtr != NULL &&
            cuDeviceGetPCIBusIdPtr  != NULL);
}

int
load_cuda_sym( char *status )
{
    cuda_dlp = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (cuda_dlp == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    cuInitPtr               = dlsym(cuda_dlp, "cuInit");
    cuDeviceGetPtr          = dlsym(cuda_dlp, "cuDeviceGet");
    cuDeviceGetNamePtr      = dlsym(cuda_dlp, "cuDeviceGetName");
    cuDeviceGetCountPtr     = dlsym(cuda_dlp, "cuDeviceGetCount");
    cuDeviceGetAttributePtr = dlsym(cuda_dlp, "cuDeviceGetAttribute");
    cuDeviceGetPCIBusIdPtr  = dlsym(cuda_dlp, "cuDeviceGetPCIBusId");

    if (!cuda_is_enabled()) {
        const char *message = "dlsym() of CUDA symbols failed";
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_cuda_sym( void )
{
    if (cuda_dlp != NULL) {
        dlclose(cuda_dlp);
    }

    cuInitPtr               = NULL;
    cuDeviceGetPtr          = NULL;
    cuDeviceGetNamePtr      = NULL;
    cuDeviceGetCountPtr     = NULL;
    cuDeviceGetAttributePtr = NULL;
    cuDeviceGetPCIBusIdPtr  = NULL;

    return cuda_is_enabled();
}
#endif /* HAVE_CUDA */

#ifdef HAVE_NVML
void
fill_dev_affinity_info( _sysdetect_gpu_info_u *info, int dev_count )
{
    int dev;
    for (dev = 0; dev < dev_count; ++dev) {
        char bus_id_str[20] = { 0 };
        CU_CALL((*cuDeviceGetPCIBusIdPtr)(bus_id_str, 20, dev), return);

        nvmlDevice_t device;
        NVML_CALL((*nvmlDeviceGetHandleByPciBusIdPtr)(bus_id_str, &device),
                  return);

        char uuid_str[PAPI_NVML_DEV_BUFFER_SIZE] = { 0 };
        NVML_CALL((*nvmlDeviceGetUUIDPtr)(device, uuid_str,
                                          PAPI_NVML_DEV_BUFFER_SIZE),
                  return);

        _sysdetect_gpu_info_u *dev_info = &info[dev];
        dev_info->nvidia.uid = hash((unsigned char *) uuid_str);
    }
}

unsigned long
hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

int
nvml_is_enabled( void )
{
    return (nvmlInitPtr                      != NULL &&
            nvmlDeviceGetCountPtr            != NULL &&
            nvmlDeviceGetHandleByPciBusIdPtr != NULL &&
            nvmlDeviceGetUUIDPtr             != NULL);
}

int
load_nvml_sym( char *status )
{
    nvml_dlp = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (nvml_dlp == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    nvmlInitPtr                      = dlsym(nvml_dlp, "nvmlInit_v2");
    nvmlDeviceGetCountPtr            = dlsym(nvml_dlp, "nvmlDeviceGetCount_v2");
    nvmlDeviceGetHandleByPciBusIdPtr = dlsym(nvml_dlp, "nvmlDeviceGetHandleByPciBusId_v2");
    nvmlDeviceGetUUIDPtr             = dlsym(nvml_dlp, "nvmlDeviceGetUUID");

    if (!nvml_is_enabled()) {
        const char *message = "dlsym() of NVML symbols failed";
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_nvml_sym( void )
{
    if (nvml_dlp != NULL) {
        dlclose(nvml_dlp);
    }

    nvmlInitPtr                      = NULL;
    nvmlDeviceGetCountPtr            = NULL;
    nvmlDeviceGetHandleByPciBusIdPtr = NULL;
    nvmlDeviceGetUUIDPtr             = NULL;

    return nvml_is_enabled();
}
#endif /* HAVE_NVML */

void
open_nvidia_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    memset(dev_type_info, 0, sizeof(*dev_type_info));
    dev_type_info->id = PAPI_DEV_TYPE_ID__CUDA;
    strcpy(dev_type_info->vendor, "NVIDIA");
    strcpy(dev_type_info->status, "Device Initialized");

#ifdef HAVE_CUDA
    if (load_cuda_sym(dev_type_info->status)) {
        return;
    }

    int dev, dev_count;
    CU_CALL((*cuDeviceGetCountPtr)(&dev_count), return);
    dev_type_info->num_devices = dev_count;
    if (dev_count == 0) {
        return;
    }

    _sysdetect_gpu_info_u *arr = papi_calloc(dev_count, sizeof(*arr));
    for (dev = 0; dev < dev_count; ++dev) {
        fill_dev_info(&arr[dev], dev);
    }

#ifdef HAVE_NVML
    if (!load_nvml_sym(dev_type_info->status)) {
        fill_dev_affinity_info(arr, dev_count);
        unload_nvml_sym();
    }
#else
    const char *message = "NVML not configured, no device affinity available";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Status string truncated.");
    }
#endif /* HAVE_NVML */

    unload_cuda_sym();
    dev_type_info->dev_info_arr = (_sysdetect_dev_info_u *)arr;
#else
    const char *message = "CUDA not configured, no CUDA device available";
    int count = snprintf(dev_type_info->status, PAPI_MAX_STR_LEN, "%s", message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Status string truncated.");
    }
#endif /* HAVE_CUDA */
}

void
close_nvidia_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    papi_free(dev_type_info->dev_info_arr);
}
