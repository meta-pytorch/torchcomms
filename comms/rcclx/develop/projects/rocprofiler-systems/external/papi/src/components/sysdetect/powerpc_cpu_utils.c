#include "sysdetect.h"
#include "powerpc_cpu_utils.h"
#include "os_cpu_utils.h"

_sysdetect_cache_level_info_t ppc970_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 65536, 128, 512, 1},
            {PAPI_MH_TYPE_DATA, 32768, 128, 256, 2}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 524288, 128, 4096, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power5_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 65536, 128, 512, 2},
            {PAPI_MH_TYPE_DATA, 32768, 128, 256, 4}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 1966080, 128, 15360, 10},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 37748736, 256, 147456, 12},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power6_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 65536, 128, 512, 4},
            {PAPI_MH_TYPE_DATA, 65536, 128, 512, 8}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 4194304, 128, 16384, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 33554432, 128, 262144, 16},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power7_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 32768, 128, 64, 4},
            {PAPI_MH_TYPE_DATA, 32768, 128, 32, 8}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 524288, 128, 256, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 4194304, 128, 4096, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power8_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 32768, 128, 64, 8},
            {PAPI_MH_TYPE_DATA, 65536, 128, 512, 8}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 262144, 128, 256, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 8388608, 128, 65536, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power9_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 32768, 128, 256, 8},
            {PAPI_MH_TYPE_DATA, 32768, 128, 256, 8}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 524288, 128, 4096, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 10485760, 128, 81920, 20},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t power10_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 49152, 128, 384, 6},
            {PAPI_MH_TYPE_DATA, 32768, 128, 256, 8}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 1048576, 128, 8192, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 4194304, 128, 32768, 16},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

#define SPRN_PVR            0x11F /* Processor Version Register */
#define PVR_PROCESSOR_SHIFT 16

static unsigned int mfpvr( void );
static int get_cache_info( CPU_attr_e attr, int level, int *value );

int
powerpc_cpu_init( void )
{
    return CPU_SUCCESS;
}

int
powerpc_cpu_finalize( void )
{
    return CPU_SUCCESS;
}

int
powerpc_cpu_get_vendor( char *vendor )
{
    return os_cpu_get_vendor(vendor);
}

int
powerpc_cpu_get_name( char *name )
{
    return os_cpu_get_name(name);
}

int
powerpc_cpu_get_attribute( CPU_attr_e attr, int *value )
{
    return os_cpu_get_attribute(attr, value);
}

int
powerpc_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
    int status = CPU_SUCCESS;

    switch(attr) {
        case CPU_ATTR__CACHE_INST_PRESENT:
        case CPU_ATTR__CACHE_DATA_PRESENT:
        case CPU_ATTR__CACHE_UNIF_PRESENT:
        case CPU_ATTR__CACHE_INST_TOT_SIZE:
        case CPU_ATTR__CACHE_INST_LINE_SIZE:
        case CPU_ATTR__CACHE_INST_NUM_LINES:
        case CPU_ATTR__CACHE_INST_ASSOCIATIVITY:
        case CPU_ATTR__CACHE_DATA_TOT_SIZE:
        case CPU_ATTR__CACHE_DATA_LINE_SIZE:
        case CPU_ATTR__CACHE_DATA_NUM_LINES:
        case CPU_ATTR__CACHE_DATA_ASSOCIATIVITY:
        case CPU_ATTR__CACHE_UNIF_TOT_SIZE:
        case CPU_ATTR__CACHE_UNIF_LINE_SIZE:
        case CPU_ATTR__CACHE_UNIF_NUM_LINES:
        case CPU_ATTR__CACHE_UNIF_ASSOCIATIVITY:
            status = get_cache_info(attr, loc, value);
            break;
        case CPU_ATTR__NUMA_MEM_SIZE:
        case CPU_ATTR__HWTHREAD_NUMA_AFFINITY:
            status = os_cpu_get_attribute_at(attr, loc, value);
            break;
        default:
            status = CPU_ERROR;
    }

    return status;
}

int
get_cache_info( CPU_attr_e attr, int level, int *value )
{
    unsigned int pvr = mfpvr() >> PVR_PROCESSOR_SHIFT;
    static _sysdetect_cache_level_info_t *clevel_ptr;

    if (clevel_ptr) {
        return cpu_get_cache_info(attr, level, clevel_ptr, value);
    }

    switch(pvr) {
        case 0x39:               /* PPC970 */
        case 0x3C:               /* PPC970FX */
        case 0x44:               /* PPC970MP */
        case 0x45:               /* PPC970GX */
            clevel_ptr = ppc970_cache_info;
            break;
        case 0x3A:               /* POWER5 */
        case 0x3B:               /* POWER5+ */
            clevel_ptr = power5_cache_info;
            break;
        case 0x3E:               /* POWER6 */
            clevel_ptr = power6_cache_info;
            break;
        case 0x3F:               /* POWER7 */
            clevel_ptr = power7_cache_info;
            break;
        case 0x4b:               /* POWER8 */
            clevel_ptr = power8_cache_info;
            break;
        case 0x4e:               /* POWER9 */
            clevel_ptr = power9_cache_info;
            break;
        case 0x80:               /* POWER10 */
            clevel_ptr = power10_cache_info;
            break;
        default:
            return CPU_ERROR;
    }

    return cpu_get_cache_info(attr, level, clevel_ptr, value);
}

unsigned int
mfpvr( void )
{
    unsigned long pvr;
    __asm__ ("mfspr %0,%1"
             : "=r" (pvr)
             : "i" (SPRN_PVR));
    return pvr;
}
