/**
 * @file    sysdetect.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a system info detection component, it provides general hardware
 *  information across the system, additionally to CPU, such as GPU, Network,
 *  installed runtime libraries, etc.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#include "sysdetect.h"
#include "nvidia_gpu.h"
#include "amd_gpu.h"
#include "cpu.h"

papi_vector_t _sysdetect_vector;

typedef struct {
    void (*open) ( _sysdetect_dev_type_info_t *dev_type_info );
    void (*close)( _sysdetect_dev_type_info_t *dev_type_info );
} dev_fn_ptr_vector;

dev_fn_ptr_vector dev_fn_vector[PAPI_DEV_TYPE_ID__MAX_NUM] = {
    {
        open_cpu_dev_type,
        close_cpu_dev_type,
    },
    {
        open_nvidia_gpu_dev_type,
        close_nvidia_gpu_dev_type,
    },
    {
        open_amd_gpu_dev_type,
        close_amd_gpu_dev_type,
    },
};

static _sysdetect_dev_type_info_t dev_type_info_arr[PAPI_DEV_TYPE_ID__MAX_NUM];

static int _sysdetect_enum_dev_type( int enum_modifier, void **handle );
static int _sysdetect_get_dev_type_attr( void *handle,
                                         PAPI_dev_type_attr_e attr, void *val );
static int _sysdetect_get_dev_attr( void *handle, int id, PAPI_dev_attr_e attr,
                                    void *val );
static void get_num_threads_per_numa( _sysdetect_cpu_info_t *cpu_info );

static void
init_dev_info( void )
{
    int id;

    for (id = 0; id < PAPI_DEV_TYPE_ID__MAX_NUM; ++id) {
        dev_fn_vector[id].open( &dev_type_info_arr[id] );
    }
}

static void
cleanup_dev_info( void )
{
    int id;

    for (id = 0; id < PAPI_DEV_TYPE_ID__MAX_NUM; ++id) {
        dev_fn_vector[id].close( &dev_type_info_arr[id] );
    }
}

/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_sysdetect_init_component( int cidx )
{

    SUBDBG( "_sysdetect_init_component...\n" );

    /* Export the component id */
    _sysdetect_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}

static int
_sysdetect_init_thread( hwd_context_t *ctx __attribute__((unused)) )
{
    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_sysdetect_shutdown_component( void )
{

    SUBDBG( "_sysdetect_shutdown_component...\n" );

    cleanup_dev_info( );

    return PAPI_OK;
}

static int
_sysdetect_shutdown_thread(hwd_context_t *ctx __attribute__((unused)) )
{
    return PAPI_OK;
}

static void
_sysdetect_init_private( void  )
{
    static int initialized;

    if (initialized) {
        return;
    }

    init_dev_info( );

    initialized = 1;
}

/** Trigered by PAPI_{enum,get}_dev_xxx interfaces */
int
_sysdetect_user( int unused __attribute__((unused)), void *in, void *out )
{
    int papi_errno = PAPI_OK;

    _sysdetect_init_private();

    _papi_hwi_sysdetect_t *args = (_papi_hwi_sysdetect_t *) in;
    int modifier;
    void *handle;
    int id;
    PAPI_dev_type_attr_e dev_type_attr;
    PAPI_dev_attr_e dev_attr;

    switch (args->query_type) {
        case PAPI_SYSDETECT_QUERY__DEV_TYPE_ENUM:
            modifier = args->query.enumerate.modifier;
            papi_errno = _sysdetect_enum_dev_type(modifier, out);
            break;
        case PAPI_SYSDETECT_QUERY__DEV_TYPE_ATTR:
            handle = args->query.dev_type.handle;
            dev_type_attr = args->query.dev_type.attr;
            papi_errno = _sysdetect_get_dev_type_attr(handle, dev_type_attr,
                                                      out);
            break;
        case PAPI_SYSDETECT_QUERY__DEV_ATTR:
            handle = args->query.dev.handle;
            id = args->query.dev.id;
            dev_attr = args->query.dev.attr;
            papi_errno = _sysdetect_get_dev_attr(handle, id, dev_attr, out);
            break;
        default:
            papi_errno = PAPI_EMISC;
    }

    return papi_errno;
}

int
_sysdetect_enum_dev_type( int enum_modifier, void **handle )
{
    static int dev_type_id;

    if (PAPI_DEV_TYPE_ENUM__FIRST == enum_modifier) {
        dev_type_id = 0;
        *(void **) handle = &dev_type_info_arr[dev_type_id];
        return PAPI_OK;
    }

    int not_found = 1;
    while (not_found && dev_type_id < PAPI_DEV_TYPE_ID__MAX_NUM) {
        if ((1 << dev_type_info_arr[dev_type_id].id) & enum_modifier) {
            *handle = &dev_type_info_arr[dev_type_id];
            not_found = 0;
        }
        ++dev_type_id;
    }

    if (not_found) {
        *handle = NULL;
        dev_type_id = 0;
        return PAPI_EINVAL;
    }

    return PAPI_OK;
}

int
_sysdetect_get_dev_type_attr( void *handle, PAPI_dev_type_attr_e attr, void *val )
{
    int papi_errno = PAPI_OK;

    _sysdetect_dev_type_info_t *dev_type_info =
        (_sysdetect_dev_type_info_t *) handle;

    switch(attr) {
        case PAPI_DEV_TYPE_ATTR__INT_PAPI_ID:
            *(int *) val = dev_type_info->id;
            break;
        case PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID:
            *(int *) val = dev_type_info->vendor_id;
            break;
        case PAPI_DEV_TYPE_ATTR__CHAR_NAME:
            *(const char **) val = dev_type_info->vendor;
            break;
        case PAPI_DEV_TYPE_ATTR__INT_COUNT:
            *(int *) val = dev_type_info->num_devices;
            break;
        case PAPI_DEV_TYPE_ATTR__CHAR_STATUS:
            *(const char **) val = dev_type_info->status;
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
_sysdetect_get_dev_attr( void *handle, int id, PAPI_dev_attr_e attr, void *val )
{
    int papi_errno = PAPI_OK;

    _sysdetect_dev_type_info_t *dev_type_info =
        (_sysdetect_dev_type_info_t *) handle;
    /* there is only one cpu vendor/model per system, hence id = 0 */
    _sysdetect_cpu_info_t *cpu_info =
        (_sysdetect_cpu_info_t *) &dev_type_info->dev_info_arr[0];
    _sysdetect_gpu_info_u *gpu_info =
        (_sysdetect_gpu_info_u *) (dev_type_info->dev_info_arr) + id;

    switch(attr) {
        /* CPU attributes */
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[0].cache[0].size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[0].cache[1].size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[1].cache[0].size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[2].cache[0].size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[0].cache[0].line_size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[0].cache[1].line_size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[1].cache[0].line_size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE:
            *(unsigned int *) val = cpu_info->clevel[2].cache[0].line_size;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT:
            *(unsigned int *) val = cpu_info->clevel[0].cache[0].num_lines;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT:
            *(unsigned int *) val = cpu_info->clevel[0].cache[1].num_lines;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT:
            *(unsigned int *) val = cpu_info->clevel[1].cache[0].num_lines;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT:
            *(unsigned int *) val = cpu_info->clevel[2].cache[0].num_lines;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC:
            *(unsigned int *) val = cpu_info->clevel[0].cache[0].associativity;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC:
            *(unsigned int *) val = cpu_info->clevel[0].cache[1].associativity;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC:
            *(unsigned int *) val = cpu_info->clevel[1].cache[0].associativity;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC:
            *(unsigned int *) val = cpu_info->clevel[2].cache[0].associativity;
            break;
        case PAPI_DEV_ATTR__CPU_CHAR_NAME:
            *(const char **) val = cpu_info->name;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_FAMILY:
            *(unsigned int *) val = cpu_info->cpuid_family;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_MODEL:
            *(unsigned int *) val = cpu_info->cpuid_model;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_STEPPING:
            *(unsigned int *) val = cpu_info->cpuid_stepping;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT:
            *(unsigned int *) val = cpu_info->sockets;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT:
            *(unsigned int *) val = cpu_info->numas;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT:
            *(int *) val = cpu_info->cores;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT:
            *(int *) val = cpu_info->threads * cpu_info->cores * cpu_info->sockets;
            break;
        case PAPI_DEV_ATTR__CPU_UINT_THR_NUMA_AFFINITY:
            *(int *) val = cpu_info->numa_affinity[id];
            break;
        case PAPI_DEV_ATTR__CPU_UINT_THR_PER_NUMA:
            get_num_threads_per_numa(cpu_info);
            *(int *) val = cpu_info->num_threads_per_numa[id];
            break;
        case PAPI_DEV_ATTR__CPU_UINT_NUMA_MEM_SIZE:
            *(unsigned int *) val = (cpu_info->numa_memory[id] >> 10);
            break;
        /* NVIDIA GPU attributes */
        case PAPI_DEV_ATTR__CUDA_ULONG_UID:
            *(unsigned long *) val = gpu_info->nvidia.uid;
            break;
        case PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME:
            *(const char **) val = gpu_info->nvidia.name;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE:
            *(unsigned int *) val = gpu_info->nvidia.warp_size;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_THR_PER_BLK:
            *(unsigned int *) val = gpu_info->nvidia.max_threads_per_block;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_PER_SM:
            *(unsigned int *) val = gpu_info->nvidia.max_blocks_per_multi_proc;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_BLK:
            *(unsigned int *) val = gpu_info->nvidia.max_shmmem_per_block;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_SM:
            *(unsigned int *) val = gpu_info->nvidia.max_shmmem_per_multi_proc;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_X:
            *(unsigned int *) val = gpu_info->nvidia.max_block_dim_x;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Y:
            *(unsigned int *) val = gpu_info->nvidia.max_block_dim_y;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Z:
            *(unsigned int *) val = gpu_info->nvidia.max_block_dim_z;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_X:
            *(unsigned int *) val = gpu_info->nvidia.max_grid_dim_x;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Y:
            *(unsigned int *) val = gpu_info->nvidia.max_grid_dim_y;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Z:
            *(unsigned int *) val = gpu_info->nvidia.max_grid_dim_z;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_SM_COUNT:
            *(unsigned int *) val = gpu_info->nvidia.multi_processor_count;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_MULTI_KERNEL:
            *(unsigned int *) val = gpu_info->nvidia.multi_kernel_per_ctx;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_MAP_HOST_MEM:
            *(unsigned int *) val = gpu_info->nvidia.can_map_host_mem;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_MEMCPY_OVERLAP:
            *(unsigned int *) val = gpu_info->nvidia.can_overlap_comp_and_data_xfer;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_UNIFIED_ADDR:
            *(unsigned int *) val = gpu_info->nvidia.unified_addressing;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_MANAGED_MEM:
            *(unsigned int *) val = gpu_info->nvidia.managed_memory;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR:
            *(unsigned int *) val = gpu_info->nvidia.major;
            break;
        case PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR:
            *(unsigned int *) val = gpu_info->nvidia.minor;
            break;
        /* AMD GPU attributes */
        case PAPI_DEV_ATTR__ROCM_ULONG_UID:
            *(unsigned long *) val = gpu_info->amd.uid;
            break;
        case PAPI_DEV_ATTR__ROCM_CHAR_DEVICE_NAME:
            *(const char **) val = gpu_info->amd.name;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_SIMD_PER_CU:
            *(unsigned int *) val = gpu_info->amd.simd_per_compute_unit;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WORKGROUP_SIZE:
            *(unsigned int *) val = gpu_info->amd.max_threads_per_workgroup;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WAVEFRONT_SIZE:
            *(unsigned int *) val = gpu_info->amd.wavefront_size;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WAVE_PER_CU:
            *(unsigned int *) val = gpu_info->amd.max_waves_per_compute_unit;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_SHM_PER_WG:
            *(unsigned int *) val = gpu_info->amd.max_shmmem_per_workgroup;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_X:
            *(unsigned int *) val = gpu_info->amd.max_workgroup_dim_x;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Y:
            *(unsigned int *) val = gpu_info->amd.max_workgroup_dim_y;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Z:
            *(unsigned int *) val = gpu_info->amd.max_workgroup_dim_z;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_X:
            *(unsigned int *) val = gpu_info->amd.max_grid_dim_x;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Y:
            *(unsigned int *) val = gpu_info->amd.max_grid_dim_y;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Z:
            *(unsigned int *) val = gpu_info->amd.max_grid_dim_z;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_CU_COUNT:
            *(unsigned int *) val = gpu_info->amd.compute_unit_count;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MAJOR:
            *(unsigned int *) val = gpu_info->amd.major;
            break;
        case PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MINOR:
            *(unsigned int *) val = gpu_info->amd.minor;
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

void
get_num_threads_per_numa( _sysdetect_cpu_info_t *cpu_info )
{
    static int initialized;
    int k;

    if (initialized) {
        return;
    }

    int threads = cpu_info->threads * cpu_info->cores * cpu_info->sockets;
    for (k = 0; k < threads; ++k) {
        int tmp = cpu_info->numa_affinity[k];
        // If gaps exist in the core numbering, the numa_affinity entries
        // for the skipped core ids will be negative.
        if( tmp >= 0 )
            cpu_info->num_threads_per_numa[cpu_info->numa_affinity[k]]++;
    }

    initialized = 1;
}

/** Vector that points to entry points for our component */
papi_vector_t _sysdetect_vector = {
    .cmp_info = {
                 .name = "sysdetect",
                 .short_name = "sysdetect",
                 .description = "System info detection component",
                 .version = "1.0",
                 .support_version = "n/a",
                 .kernel_version = "n/a",
                },

    /* Sizes of framework-opaque component-private structures */
    .size = {
        .context = 1, /* unused */
        .control_state = 1, /* unused */
        .reg_value = 1, /* unused */
        .reg_alloc = 1, /* unused */
    },

    /* Used for general PAPI interactions */
    .init_component = _sysdetect_init_component,
    .init_thread = _sysdetect_init_thread,
    .shutdown_component = _sysdetect_shutdown_component,
    .shutdown_thread = _sysdetect_shutdown_thread,
    .user = _sysdetect_user,
};
