#ifndef __SYSDETECT_H__
#define __SYSDETECT_H__

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

typedef union {
    struct {
        unsigned long uid;
        char name[PAPI_2MAX_STR_LEN];
        int warp_size;
        int max_threads_per_block;
        int max_blocks_per_multi_proc;
        int max_shmmem_per_block;
        int max_shmmem_per_multi_proc;
        int max_block_dim_x;
        int max_block_dim_y;
        int max_block_dim_z;
        int max_grid_dim_x;
        int max_grid_dim_y;
        int max_grid_dim_z;
        int multi_processor_count;
        int multi_kernel_per_ctx;
        int can_map_host_mem;
        int can_overlap_comp_and_data_xfer;
        int unified_addressing;
        int managed_memory;
        int major;
        int minor;
    } nvidia;

    struct {
        unsigned long uid;
        char name[PAPI_2MAX_STR_LEN];
        unsigned int wavefront_size;
        unsigned int simd_per_compute_unit;
        unsigned int max_threads_per_workgroup;
        unsigned int max_waves_per_compute_unit;
        unsigned int max_shmmem_per_workgroup;
        unsigned short max_workgroup_dim_x;
        unsigned short max_workgroup_dim_y;
        unsigned short max_workgroup_dim_z;
        unsigned int max_grid_dim_x;
        unsigned int max_grid_dim_y;
        unsigned int max_grid_dim_z;
        unsigned int compute_unit_count;
        unsigned int major;
        unsigned int minor;
    } amd;
} _sysdetect_gpu_info_u;

typedef struct {
    int num_caches;
    PAPI_mh_cache_info_t cache[PAPI_MH_MAX_LEVELS];
} _sysdetect_cache_level_info_t;

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    int cpuid_family;
    int cpuid_model;
    int cpuid_stepping;
    int sockets;
    int numas;
    int cores;
    int threads;
    int cache_levels;
    _sysdetect_cache_level_info_t clevel[PAPI_MAX_MEM_HIERARCHY_LEVELS];
#define PAPI_MAX_NUM_NODES 8
    int numa_memory[PAPI_MAX_NUM_NODES];
#define PAPI_MAX_NUM_THREADS 512
    int numa_affinity[PAPI_MAX_NUM_THREADS];
#define PAPI_MAX_THREADS_PER_NUMA (PAPI_MAX_NUM_THREADS / PAPI_MAX_NUM_NODES)
    int num_threads_per_numa[PAPI_MAX_THREADS_PER_NUMA];
} _sysdetect_cpu_info_t;

typedef union {
    _sysdetect_gpu_info_u gpu;
    _sysdetect_cpu_info_t cpu;
} _sysdetect_dev_info_u;

typedef struct {
    PAPI_dev_type_id_e id;
    char vendor[PAPI_MAX_STR_LEN];
    int vendor_id;
    char status[PAPI_MAX_STR_LEN];
    int num_devices;
    _sysdetect_dev_info_u *dev_info_arr;
} _sysdetect_dev_type_info_t;

#endif /* End of __SYSDETECT_H__ */
