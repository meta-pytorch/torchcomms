#ifndef __CPU_UTILS_H__
#define __CPU_UTILS_H__

#define CPU_SUCCESS  0
#define CPU_ERROR   -1

typedef enum {
    CPU_ATTR__NUM_SOCKETS,
    CPU_ATTR__NUM_NODES,
    CPU_ATTR__NUM_CORES,
    CPU_ATTR__NUM_THREADS,
    CPU_ATTR__VENDOR_ID,
    CPU_ATTR__CPUID_FAMILY,
    CPU_ATTR__CPUID_MODEL,
    CPU_ATTR__CPUID_STEPPING,
    /* Cache Attributes */
    CPU_ATTR__CACHE_MAX_NUM_LEVELS,
    CPU_ATTR__CACHE_INST_PRESENT,
    CPU_ATTR__CACHE_DATA_PRESENT,
    CPU_ATTR__CACHE_UNIF_PRESENT,
    CPU_ATTR__CACHE_INST_TOT_SIZE,
    CPU_ATTR__CACHE_INST_LINE_SIZE,
    CPU_ATTR__CACHE_INST_NUM_LINES,
    CPU_ATTR__CACHE_INST_ASSOCIATIVITY,
    CPU_ATTR__CACHE_DATA_TOT_SIZE,
    CPU_ATTR__CACHE_DATA_LINE_SIZE,
    CPU_ATTR__CACHE_DATA_NUM_LINES,
    CPU_ATTR__CACHE_DATA_ASSOCIATIVITY,
    CPU_ATTR__CACHE_UNIF_TOT_SIZE,
    CPU_ATTR__CACHE_UNIF_LINE_SIZE,
    CPU_ATTR__CACHE_UNIF_NUM_LINES,
    CPU_ATTR__CACHE_UNIF_ASSOCIATIVITY,
    /* Hardware Thread Affinity Attributes */
    CPU_ATTR__HWTHREAD_NUMA_AFFINITY,
    /* Memory Attributes */
    CPU_ATTR__NUMA_MEM_SIZE,
} CPU_attr_e;

int cpu_init( void );
int cpu_finalize( void );
int cpu_get_vendor( char *vendor );
int cpu_get_name( char *name );
int cpu_get_attribute( CPU_attr_e attr, int *value );
int cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );
int cpu_get_cache_info( CPU_attr_e attr, int level, _sysdetect_cache_level_info_t *clevel_ptr, int *value );

#endif /* End of __CPU_UTILS_H__ */
