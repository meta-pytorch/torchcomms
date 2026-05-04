/**
 * @file    cpu.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  Scan functions for all Vendor CPU systems.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "sysdetect.h"
#include "cpu.h"
#include "cpu_utils.h"

#define CPU_CALL(call, err_handle) do {                                         \
    int _status = (call);                                                       \
    if (_status != CPU_SUCCESS) {                                               \
        SUBDBG("error: function %s failed with error %d\n", #call, _status);    \
        err_handle;                                                             \
    }                                                                           \
} while(0)

static void
fill_cpu_info( _sysdetect_cpu_info_t *info )
{
    CPU_CALL(cpu_get_name(info->name),
             strcpy(info->name, "UNKNOWN"));
    CPU_CALL(cpu_get_attribute(CPU_ATTR__CPUID_FAMILY, &info->cpuid_family),
             info->cpuid_family = -1);
    CPU_CALL(cpu_get_attribute(CPU_ATTR__CPUID_MODEL, &info->cpuid_model),
             info->cpuid_model = -1);
    CPU_CALL(cpu_get_attribute(CPU_ATTR__CPUID_STEPPING, &info->cpuid_stepping),
             info->cpuid_stepping = -1);
    CPU_CALL(cpu_get_attribute( CPU_ATTR__NUM_SOCKETS, &info->sockets ),
             info->sockets = -1);
    CPU_CALL(cpu_get_attribute( CPU_ATTR__NUM_NODES, &info->numas ),
             info->numas = -1);
    CPU_CALL(cpu_get_attribute(CPU_ATTR__NUM_CORES, &info->cores),
             info->cores = -1);
    CPU_CALL(cpu_get_attribute(CPU_ATTR__NUM_THREADS, &info->threads),
             info->threads = -1);

    int cache_levels;
    CPU_CALL(cpu_get_attribute(CPU_ATTR__CACHE_MAX_NUM_LEVELS, &cache_levels),
             cache_levels = 0);

    int level, a, b, c;
    for (level = 1; level <= cache_levels; ++level) {
        CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_INST_PRESENT, level, &a), a = 0);
        CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_DATA_PRESENT, level, &b), b = 0);
        CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_UNIF_PRESENT, level, &c), c = 0);

        if (!(a || b || c)) {
            /* No caches at this level */
            break;
        }

        int *num_caches = &info->clevel[level-1].num_caches;
        if (a) {
            info->clevel[level-1].cache[*num_caches].type = PAPI_MH_TYPE_INST;
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_INST_TOT_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].size),
                     info->clevel[level-1].cache[*num_caches].size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_INST_LINE_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].line_size),
                     info->clevel[level-1].cache[*num_caches].line_size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_INST_NUM_LINES, level,
                                          &info->clevel[level-1].cache[*num_caches].num_lines),
                     info->clevel[level-1].cache[*num_caches].num_lines = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_INST_ASSOCIATIVITY, level,
                                          &info->clevel[level-1].cache[*num_caches].associativity),
                     info->clevel[level-1].cache[*num_caches].associativity = 0);
            ++(*num_caches);
        }

        if (b) {
            info->clevel[level-1].cache[*num_caches].type = PAPI_MH_TYPE_DATA;
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_DATA_TOT_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].size),
                     info->clevel[level-1].cache[*num_caches].size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_DATA_LINE_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].line_size),
                     info->clevel[level-1].cache[*num_caches].line_size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_DATA_NUM_LINES, level,
                                          &info->clevel[level-1].cache[*num_caches].num_lines),
                     info->clevel[level-1].cache[*num_caches].num_lines = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_DATA_ASSOCIATIVITY, level,
                                          &info->clevel[level-1].cache[*num_caches].associativity),
                     info->clevel[level-1].cache[*num_caches].associativity = 0);
            ++(*num_caches);
        }

        if (c) {
            info->clevel[level-1].cache[*num_caches].type = PAPI_MH_TYPE_UNIFIED;
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_UNIF_TOT_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].size),
                     info->clevel[level-1].cache[*num_caches].size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_UNIF_LINE_SIZE, level,
                                          &info->clevel[level-1].cache[*num_caches].line_size),
                     info->clevel[level-1].cache[*num_caches].line_size = 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_UNIF_NUM_LINES, level,
                                          &info->clevel[level-1].cache[*num_caches].num_lines),
                     info->clevel[level-1].cache[*num_caches].num_lines= 0);
            CPU_CALL(cpu_get_attribute_at(CPU_ATTR__CACHE_UNIF_ASSOCIATIVITY, level,
                                          &info->clevel[level-1].cache[*num_caches].associativity),
                     info->clevel[level-1].cache[*num_caches].associativity = 0);
            ++(*num_caches);
        }
    }

    for (a = 0; a < info->numas; ++a) {
        CPU_CALL(cpu_get_attribute_at(CPU_ATTR__NUMA_MEM_SIZE, a, &info->numa_memory[a]),
                 info->numa_memory[a] = 0);
    }

    for (a = 0; a < info->threads * info->cores * info->sockets; ++a) {
        int status = cpu_get_attribute_at(CPU_ATTR__HWTHREAD_NUMA_AFFINITY, a, &info->numa_affinity[a]);
        if( CPU_SUCCESS != status ){
            SUBDBG("Warning: core ids might not be contiguous.\n");
        }
    }

    info->cache_levels = level;
}

void
open_cpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    memset(dev_type_info, 0, sizeof(*dev_type_info));
    dev_type_info->id = PAPI_DEV_TYPE_ID__CPU;

    CPU_CALL(cpu_get_vendor(dev_type_info->vendor),
             strcpy(dev_type_info->vendor, "UNKNOWN"));

    CPU_CALL(cpu_get_attribute(CPU_ATTR__VENDOR_ID, &dev_type_info->vendor_id),
             dev_type_info->vendor_id = -1);

    strcpy(dev_type_info->status, "Device Initialized");
    dev_type_info->num_devices = 1;

    _sysdetect_cpu_info_t *arr = papi_calloc(1, sizeof(*arr));
    fill_cpu_info(arr);
    dev_type_info->dev_info_arr = (_sysdetect_dev_info_u *)arr;
}

void
close_cpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info )
{
    papi_free(dev_type_info->dev_info_arr);
}
