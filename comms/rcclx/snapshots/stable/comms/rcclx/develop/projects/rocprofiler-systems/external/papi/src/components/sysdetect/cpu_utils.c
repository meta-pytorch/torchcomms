/**
 * @file    cpu_utils.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @brief
 *  Returns information about CPUs
 */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#include "sysdetect.h"
#include "cpu_utils.h"

#if defined(__x86_64__) || defined(__amd64__)
#include "x86_cpu_utils.h"
#elif defined(__powerpc__)
#include "powerpc_cpu_utils.h"
#elif defined(__arm__) || defined(__aarch64__)
#include "arm_cpu_utils.h"
#endif

#include "os_cpu_utils.h"

int
cpu_get_vendor( char *vendor )
{
#if defined(__x86_64__) || defined(__amd64__)
    return x86_cpu_get_vendor(vendor);
#elif defined(__powerpc__)
    return powerpc_cpu_get_vendor(vendor);
#elif defined(__arm__) || defined(__aarch64__)
    return arm_cpu_get_vendor(vendor);
#endif
    return os_cpu_get_vendor(vendor);
}

int
cpu_get_name( char *name )
{
#if defined(__x86_64__) || defined(__amd64__)
    return x86_cpu_get_name(name);
#elif defined(__powerpc__)
    return powerpc_cpu_get_name(name);
#elif defined(__arm__) || defined(__aarch64__)
    return arm_cpu_get_name(name);
#endif
    return os_cpu_get_name(name);
}

int
cpu_get_attribute( CPU_attr_e attr, int *value )
{
#if defined(__x86_64__) || defined(__amd64__)
    return x86_cpu_get_attribute(attr, value);
#elif defined(__powerpc__)
    return powerpc_cpu_get_attribute(attr, value);
#elif defined(__arm__) || defined(__aarch64__)
    return arm_cpu_get_attribute(attr, value);
#endif
    return os_cpu_get_attribute(attr, value);
}

int
cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
#if defined(__x86_64__) || defined(__amd64__)
    return x86_cpu_get_attribute_at(attr, loc, value);
#elif defined(__powerpc__)
    return powerpc_cpu_get_attribute_at(attr, loc, value);
#elif defined(__arm__) || defined(__aarch64__)
    return arm_cpu_get_attribute_at(attr, loc, value);
#endif
    return os_cpu_get_attribute_at(attr, loc, value);
}

static int
get_cache_level( _sysdetect_cache_level_info_t *clevel_ptr, int type )
{
    int i;

    for (i = 0; i < clevel_ptr->num_caches; ++i) {
        if (clevel_ptr->cache[i].type == type)
            return i + 1;
    }

    return 0;
}

int
cpu_get_cache_info( CPU_attr_e attr, int level, _sysdetect_cache_level_info_t *clevel_ptr, int *value )
{
    int status = CPU_SUCCESS;
    int i;

    *value = 0;

    if (level >= PAPI_MAX_MEM_HIERARCHY_LEVELS)
        return CPU_ERROR;

    switch(attr) {
        case CPU_ATTR__CACHE_INST_PRESENT:
            if (get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_INST)) {
                *value = 1;
            }
            break;
        case CPU_ATTR__CACHE_DATA_PRESENT:
            if (get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_DATA)) {
                *value = 1;
            }
            break;
        case CPU_ATTR__CACHE_UNIF_PRESENT:
            if (get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_UNIFIED)) {
                *value = 1;
            }
            break;
        case CPU_ATTR__CACHE_INST_TOT_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_INST))) {
                *value = clevel_ptr[level-1].cache[i-1].size;
            }
            break;
        case CPU_ATTR__CACHE_INST_LINE_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_INST))) {
                *value = clevel_ptr[level-1].cache[i-1].line_size;
            }
            break;
        case CPU_ATTR__CACHE_INST_NUM_LINES:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_INST))) {
                *value = clevel_ptr[level-1].cache[i-1].num_lines;
            }
            break;
        case CPU_ATTR__CACHE_INST_ASSOCIATIVITY:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_INST))) {
                *value = clevel_ptr[level-1].cache[i-1].associativity;
            }
            break;
        case CPU_ATTR__CACHE_DATA_TOT_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_DATA))) {
                *value = clevel_ptr[level-1].cache[i-1].size;
            }
            break;
        case CPU_ATTR__CACHE_DATA_LINE_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_DATA))) {
                *value = clevel_ptr[level-1].cache[i-1].line_size;
            }
            break;
        case CPU_ATTR__CACHE_DATA_NUM_LINES:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_DATA))) {
                *value = clevel_ptr[level-1].cache[i-1].num_lines;
            }
            break;
        case CPU_ATTR__CACHE_DATA_ASSOCIATIVITY:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_DATA))) {
                *value = clevel_ptr[level-1].cache[i-1].associativity;
            }
            break;
        case CPU_ATTR__CACHE_UNIF_TOT_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_UNIFIED))) {
                *value = clevel_ptr[level-1].cache[i-1].size;
            }
            break;
        case CPU_ATTR__CACHE_UNIF_LINE_SIZE:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_UNIFIED))) {
                *value = clevel_ptr[level-1].cache[i-1].line_size;
            }
            break;
        case CPU_ATTR__CACHE_UNIF_NUM_LINES:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_UNIFIED))) {
                *value = clevel_ptr[level-1].cache[i-1].num_lines;
            }
            break;
        case CPU_ATTR__CACHE_UNIF_ASSOCIATIVITY:
            if ((i = get_cache_level(&clevel_ptr[level-1], PAPI_MH_TYPE_UNIFIED))) {
                *value = clevel_ptr[level-1].cache[i-1].associativity;
            }
            break;
        default:
            status = CPU_ERROR;
    }

    return status;
}
