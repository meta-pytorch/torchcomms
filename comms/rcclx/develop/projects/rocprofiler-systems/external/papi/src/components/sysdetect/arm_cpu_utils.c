#include <stdio.h>
#include "sysdetect.h"
#include "arm_cpu_utils.h"
#include "os_cpu_utils.h"
#include <string.h>

#define VENDOR_ARM_ARM       65
#define VENDOR_ARM_BROADCOM  66
#define VENDOR_ARM_CAVIUM    67
#define VENDOR_ARM_FUJITSU   70
#define VENDOR_ARM_HISILICON 72
#define VENDOR_ARM_APM       80
#define VENDOR_ARM_QUALCOMM  81
#define NAMEID_ARM_1176           0xb76
#define NAMEID_ARM_CORTEX_A7      0xc07
#define NAMEID_ARM_CORTEX_A8      0xc08
#define NAMEID_ARM_CORTEX_A9      0xc09
#define NAMEID_ARM_CORTEX_A15     0xc0f
#define NAMEID_ARM_CORTEX_A53     0xd03
#define NAMEID_ARM_CORTEX_A57     0xd07
#define NAMEID_ARM_CORTEX_A76     0xd0b
#define NAMEID_ARM_NEOVERSE_N1    0xd0c
#define NAMEID_ARM_NEOVERSE_N2    0xd49
#define NAMEID_ARM_NEOVERSE_V1    0xd40
#define NAMEID_ARM_NEOVERSE_V2    0xd4f
#define NAMEID_BROADCOM_THUNDERX2 0x516
#define NAMEID_CAVIUM_THUNDERX2   0x0af
#define NAMEID_FUJITSU_A64FX      0x001
#define NAMEID_FUJITSU_MONAKA     0x003
#define NAMEID_HISILICON_KUNPENG  0xd01
#define NAMEID_APM_XGENE          0x000
#define NAMEID_QUALCOMM_KRAIT     0x040

_sysdetect_cache_level_info_t fujitsu_a64fx_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 65536, 256, 64, 4},
            {PAPI_MH_TYPE_DATA, 65536, 256, 64, 4}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 8388608, 256, 2048, 16},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

_sysdetect_cache_level_info_t arm_neoverse_v2_cache_info[] = {
    { // level 1 begins
        2,
        {
            {PAPI_MH_TYPE_INST, 65536, 64, 256, 4},
            {PAPI_MH_TYPE_DATA, 65536, 64, 256, 4}
        }
    },
    { // level 2 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 1048576, 64, 2048, 8},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
    { // level 3 begins
        1,
        {
            {PAPI_MH_TYPE_UNIFIED, 119537664, 64, 155648, 12},
            {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
    },
};

static int get_cache_info( CPU_attr_e attr, int level, int *value );
static int name_id_arm_cpu_get_name( int name_id, char *name );
static int name_id_broadcom_cpu_get_name( int name_id, char *name );
static int name_id_cavium_cpu_get_name( int name_id, char *name );
static int name_id_fujitsu_cpu_get_name( int name_id, char *name );
static int name_id_hisilicon_cpu_get_name( int name_id, char *name );
static int name_id_apm_cpu_get_name( int name_id, char *name );
static int name_id_qualcomm_cpu_get_name( int name_id, char *name );

int
arm_cpu_init( void )
{
    return CPU_SUCCESS;
}

int
arm_cpu_finalize( void )
{
    return CPU_SUCCESS;
}

int
arm_cpu_get_vendor( char *vendor )
{
    int papi_errno;

    char tmp[PAPI_MAX_STR_LEN];
    papi_errno = os_cpu_get_vendor(tmp);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int vendor_id;
    sscanf(tmp, "%x", &vendor_id);

    switch(vendor_id) {
        case VENDOR_ARM_ARM:
            strcpy(vendor, "Arm");
            break;
        case VENDOR_ARM_BROADCOM:
            strcpy(vendor, "Broadcom");
            break;
        case VENDOR_ARM_CAVIUM:
            strcpy(vendor, "Cavium");
            break;
        case VENDOR_ARM_FUJITSU:
            strcpy(vendor, "Fujitsu");
            break;
        case VENDOR_ARM_HISILICON:
            strcpy(vendor, "Hisilicon");
            break;
        case VENDOR_ARM_APM:
            strcpy(vendor, "Apm");
            break;
        case VENDOR_ARM_QUALCOMM:
            strcpy(vendor, "Qualcomm");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
arm_cpu_get_name( char *name )
{
    int papi_errno;

    papi_errno = os_cpu_get_name(name);
    if (strlen(name) != 0) {
        return  papi_errno;
    }

    char tmp[PAPI_MAX_STR_LEN];
    papi_errno = os_cpu_get_vendor(tmp);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int vendor_id;
    sscanf(tmp, "%x", &vendor_id);

    int name_id;
    papi_errno = os_cpu_get_attribute(CPU_ATTR__CPUID_MODEL, &name_id);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch(vendor_id) {
        case VENDOR_ARM_ARM:
            papi_errno = name_id_arm_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_BROADCOM:
            papi_errno = name_id_broadcom_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_CAVIUM:
            papi_errno = name_id_cavium_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_FUJITSU:
            papi_errno = name_id_fujitsu_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_HISILICON:
            papi_errno = name_id_hisilicon_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_APM:
            papi_errno = name_id_apm_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_QUALCOMM:
            papi_errno = name_id_qualcomm_cpu_get_name(name_id, name);
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
arm_cpu_get_attribute( CPU_attr_e attr, int *value )
{
    return os_cpu_get_attribute(attr, value);
}

int
arm_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
    int status = CPU_SUCCESS;

    switch(attr) {
        case CPU_ATTR__CACHE_INST_PRESENT:
            //fall through
        case CPU_ATTR__CACHE_DATA_PRESENT:
            //fall through
        case CPU_ATTR__CACHE_UNIF_PRESENT:
            //fall through
        case CPU_ATTR__CACHE_INST_TOT_SIZE:
            //fall through
        case CPU_ATTR__CACHE_INST_LINE_SIZE:
            //fall through
        case CPU_ATTR__CACHE_INST_NUM_LINES:
            //fall through
        case CPU_ATTR__CACHE_INST_ASSOCIATIVITY:
            //fall through
        case CPU_ATTR__CACHE_DATA_TOT_SIZE:
            //fall through
        case CPU_ATTR__CACHE_DATA_LINE_SIZE:
            //fall through
        case CPU_ATTR__CACHE_DATA_NUM_LINES:
            //fall through
        case CPU_ATTR__CACHE_DATA_ASSOCIATIVITY:
            //fall through
        case CPU_ATTR__CACHE_UNIF_TOT_SIZE:
            //fall through
        case CPU_ATTR__CACHE_UNIF_LINE_SIZE:
            //fall through
        case CPU_ATTR__CACHE_UNIF_NUM_LINES:
            //fall through
        case CPU_ATTR__CACHE_UNIF_ASSOCIATIVITY:
            status = get_cache_info(attr, loc, value);
            break;
        case CPU_ATTR__NUMA_MEM_SIZE:
            //fall through
        case CPU_ATTR__HWTHREAD_NUMA_AFFINITY:
            status = os_cpu_get_attribute_at(attr, loc, value);
            break;
        default:
            status = CPU_ERROR;
    }

    return status;
}

int
name_id_arm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_ARM_1176:
            strcpy(name, "ARM1176");
            break;
        case NAMEID_ARM_CORTEX_A7:
            strcpy(name, "ARM Cortex A7");
            break;
        case NAMEID_ARM_CORTEX_A8:
            strcpy(name, "ARM Cortex A8");
            break;
        case NAMEID_ARM_CORTEX_A9:
            strcpy(name, "ARM Cortex A9");
            break;
        case NAMEID_ARM_CORTEX_A15:
            strcpy(name, "ARM Cortex A15");
            break;
        case NAMEID_ARM_CORTEX_A53:
            strcpy(name, "ARM Cortex A53");
            break;
        case NAMEID_ARM_CORTEX_A57:
            strcpy(name, "ARM Cortex A57");
            break;
        case NAMEID_ARM_CORTEX_A76:
            strcpy(name, "ARM Cortex A76");
            break;
        case NAMEID_ARM_NEOVERSE_N1:
            strcpy(name, "ARM Neoverse N1");
            break;
        case NAMEID_ARM_NEOVERSE_N2:
            strcpy(name, "ARM Neoverse N2");
            break;
        case NAMEID_ARM_NEOVERSE_V1:
            strcpy(name, "ARM Neoverse V1");
            break;
        case NAMEID_ARM_NEOVERSE_V2:
            strcpy(name, "ARM Neoverse V2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_broadcom_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_BROADCOM_THUNDERX2:
            strcpy(name, "Broadcom ThunderX2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_cavium_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_CAVIUM_THUNDERX2:
            strcpy(name, "Cavium ThunderX2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_fujitsu_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_FUJITSU_A64FX:
            strcpy(name, "Fujitsu A64FX");
            break;
        case NAMEID_FUJITSU_MONAKA:
            strcpy(name, "Fujitsu FUJITSU-MONAKA");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_hisilicon_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_HISILICON_KUNPENG:
            strcpy(name, "Hisilicon Kunpeng");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_apm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_APM_XGENE:
            strcpy(name, "Applied Micro X-Gene");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_qualcomm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_QUALCOMM_KRAIT:
            strcpy(name, "ARM Qualcomm Krait");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
get_cache_info( CPU_attr_e attr, int level, int *value )
{
    int impl, part;
    unsigned int implementer, partnum;
    static _sysdetect_cache_level_info_t *clevel_ptr;

    int status;
    status = arm_cpu_get_attribute(CPU_ATTR__VENDOR_ID, &impl);
    if( status != CPU_SUCCESS ) return CPU_ERROR;
    implementer = impl;

    status = arm_cpu_get_attribute(CPU_ATTR__CPUID_MODEL, &part);
    if( status != CPU_SUCCESS ) return CPU_ERROR;
    partnum = part;

    if (clevel_ptr) {
        return cpu_get_cache_info(attr, level, clevel_ptr, value);
    }

    switch ( implementer ) {
        case VENDOR_ARM_FUJITSU:
            if ( NAMEID_FUJITSU_A64FX == partnum ) { /* Fujitsu A64FX */
                clevel_ptr = fujitsu_a64fx_cache_info;
            } else {
                return CPU_ERROR;
            }
            break;
        case VENDOR_ARM_ARM:
            if ( NAMEID_ARM_NEOVERSE_V2 == partnum ) { /* ARM Neoverse V2 */
                clevel_ptr = arm_neoverse_v2_cache_info;
            } else {
                return CPU_ERROR;
            }
            break;
        default:
            return CPU_ERROR;
    }

    return cpu_get_cache_info(attr, level, clevel_ptr, value);
}
