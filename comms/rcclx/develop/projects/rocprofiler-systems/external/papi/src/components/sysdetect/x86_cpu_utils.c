#include <string.h>
#include <assert.h>
#include <limits.h>

#include "sysdetect.h"
#include "x86_cpu_utils.h"
#include "os_cpu_utils.h"

typedef struct {
    int smt_mask;
    int smt_width;
    int core_mask;
    int core_width;
    int pkg_mask;
    int pkg_width;
} apic_subid_mask_t;

typedef struct {
    int pkg;
    int core;
    int smt;
} apic_subid_t;

typedef struct {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
} cpuid_reg_t;

static _sysdetect_cache_level_info_t clevel[PAPI_MAX_MEM_HIERARCHY_LEVELS];

static int cpuid_get_vendor( char *vendor );
static int cpuid_get_name( char *name );
static int cpuid_get_attribute( CPU_attr_e attr, int *value );
static int cpuid_get_attribute_at( CPU_attr_e attr, int loc, int *value );
static int cpuid_get_topology_info( CPU_attr_e attr, int *value );
static int cpuid_get_cache_info( CPU_attr_e attr, int level, int *value );
static int intel_get_cache_info( CPU_attr_e attr, int level, int *value );
static int amd_get_cache_info( CPU_attr_e attr, int level, int *value );
static int cpuid_supports_leaves_4_11( void );
static int enum_cpu_resources( int num_mappings, apic_subid_mask_t *mask,
                               apic_subid_t *subids, int *sockets,
                               int *cores, int *threads );
static int cpuid_get_versioning_info( CPU_attr_e attr, int *value );
static int cpuid_parse_id_foreach_thread( unsigned int num_mappings,
                                          apic_subid_mask_t *mask,
                                          apic_subid_t *subid );
static int cpuid_parse_ids( int os_proc_count,
                            apic_subid_mask_t *mask,
                            apic_subid_t *subid );
static int cpuid_get_mask( apic_subid_mask_t *mask );
static int cpuid_get_leaf11_mask( apic_subid_mask_t *mask );
static int cpuid_get_leaf4_mask( apic_subid_mask_t *mask );
static unsigned int cpuid_get_x2apic_id( void );
static unsigned int cpuid_get_apic_id( void );
static unsigned int bit_width( unsigned int x );
static void cpuid( cpuid_reg_t *reg, const unsigned int func );
static void cpuid2( cpuid_reg_t *reg, const unsigned int func, const unsigned int subfunc );

static int cpuid_has_leaf4;  /* support legacy leaf1 and leaf4 interface */
static int cpuid_has_leaf11; /* support modern leaf11 interface */

int
x86_cpu_init( void )
{
    /*
     * In the future we might need to dynamically
     * allocate and free objects; init/finalize
     * functions are a good place for doing that.
     */
    return CPU_SUCCESS;
}

int
x86_cpu_finalize( void )
{
    return CPU_SUCCESS;
}

int
x86_cpu_get_vendor( char *vendor )
{
    return cpuid_get_vendor(vendor);
}

int
x86_cpu_get_name( char *name )
{
    return cpuid_get_name(name);
}

int
x86_cpu_get_attribute( CPU_attr_e attr, int *value )
{
    return cpuid_get_attribute(attr, value);
}

int
x86_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
    return cpuid_get_attribute_at(attr, loc, value);
}

int
cpuid_get_vendor( char *vendor )
{
    cpuid_reg_t reg;
    cpuid(&reg, 0); /* Highest function parameter and manufacturer ID */
    memcpy(vendor    , &reg.ebx, 4);
    memcpy(vendor + 4, &reg.edx, 4);
    memcpy(vendor + 8, &reg.ecx, 4);
    vendor[12] = '\0';
    return CPU_SUCCESS;
}

int
cpuid_get_name( char *name )
{
    cpuid_reg_t reg;
    cpuid(&reg, 0x80000000);
    if (reg.eax < 0x80000004) {
        /* Feature not implemented. Fallback! */
        return os_cpu_get_name(name);
    }

    cpuid(&reg, 0x80000002);
    memcpy(name     , &reg.eax, 4);
    memcpy(name + 4 , &reg.ebx, 4);
    memcpy(name + 8 , &reg.ecx, 4);
    memcpy(name + 12, &reg.edx, 4);

    cpuid(&reg, 0x80000003);
    memcpy(name + 16, &reg.eax, 4);
    memcpy(name + 20, &reg.ebx, 4);
    memcpy(name + 24, &reg.ecx, 4);
    memcpy(name + 28, &reg.edx, 4);

    cpuid(&reg, 0x80000004);
    memcpy(name + 32, &reg.eax, 4);
    memcpy(name + 36, &reg.ebx, 4);
    memcpy(name + 40, &reg.ecx, 4);
    memcpy(name + 44, &reg.edx, 4);

    name[48] = '\0';

    return CPU_SUCCESS;
}

int
cpuid_get_attribute( CPU_attr_e attr, int *value )
{
    int status = CPU_SUCCESS;

    switch(attr) {
        case CPU_ATTR__NUM_SOCKETS:
        case CPU_ATTR__NUM_NODES:
        case CPU_ATTR__NUM_THREADS:
        case CPU_ATTR__NUM_CORES:
            status = cpuid_get_topology_info(attr, value);
            break;
        case CPU_ATTR__CPUID_FAMILY:
        case CPU_ATTR__CPUID_MODEL:
        case CPU_ATTR__CPUID_STEPPING:
            status = cpuid_get_versioning_info(attr, value);
            break;
        case CPU_ATTR__CACHE_MAX_NUM_LEVELS:
            *value = PAPI_MAX_MEM_HIERARCHY_LEVELS;
            break;
        default:
            status = os_cpu_get_attribute(attr, value);
    }

    return status;
}

int
cpuid_get_attribute_at( CPU_attr_e attr, int loc, int *value )
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
            status = cpuid_get_cache_info(attr, loc, value);
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
cpuid_get_topology_info( CPU_attr_e attr, int *value )
{
    int status = CPU_SUCCESS;
    static int sockets, nodes, cores, threads;

    if (attr == CPU_ATTR__NUM_SOCKETS && sockets) {
        *value = sockets;
        return status;
    } else if (attr == CPU_ATTR__NUM_THREADS && threads) {
        *value = threads;
        return status;
    } else if (attr == CPU_ATTR__NUM_CORES && cores) {
        *value = cores;
        return status;
    } else if (attr == CPU_ATTR__NUM_NODES && nodes) {
        *value = nodes;
        return status;
    }

    /* Query for cpuid supported topology enumeration capabilities:
     * - cpuid in the first generation of Intel Xeon and Intel Pentium 4
     *   supporting hyper-threading (2002) provides information that allows
     *   to decompose the 8-bit wide APIC IDs into a two-level topology
     *   enumeration;
     * - with the introduction of dual-core Intel 64 processors in 2005,
     *   system topology enumeration using cpuid evolved into a three-level
     *   algorithm (to account for physical cores) on the 8-bit wide APIC ID;
     * - modern Intel 64 platforms with support for large number of logical
     *   processors use an extended 32-bit wide x2APIC ID. This is known as
     *   cpuid leaf11 interface. Legacy cpuid interface with limited 256
     *   APIC IDs, is referred to as leaf4. */
    if (!cpuid_supports_leaves_4_11()) {
        return os_cpu_get_attribute(attr, value);
    }

    /* Allocate SUBIDs' space for each logical processor */
    int os_proc_count = os_cpu_get_num_supported();
    apic_subid_t *subids = papi_malloc(os_proc_count * sizeof(*subids));
    if (!subids)
        return CPU_ERROR;

    /* Get masks for later SUBIDs extraction */
    apic_subid_mask_t mask = { 0 };
    if (cpuid_get_mask(&mask))
        goto fn_fail;

    /* For each logical processor get the unique APIC/x2APIC ID and use
     * use previously retrieved masks to extract package, core and smt
     * SUBIDs. */
    int num_mappings = cpuid_parse_ids(os_proc_count, &mask, subids);
    if (num_mappings == -1)
        goto fn_fail;

    /* Enumerate all cpu resources once and store them for later */
    status = enum_cpu_resources(num_mappings, &mask, subids, &sockets,
                                &cores, &threads);
    if (status != CPU_SUCCESS)
        goto fn_fail;

    if (attr == CPU_ATTR__NUM_SOCKETS && sockets) {
        *value = sockets;
    } else if (attr == CPU_ATTR__NUM_THREADS && threads) {
        *value = threads;
    } else if (attr == CPU_ATTR__NUM_CORES && cores) {
        *value = cores;
    } else if (attr == CPU_ATTR__NUM_NODES) {
        /* We can't read the number of numa nodes using cpuid */
        status = os_cpu_get_attribute(attr, &nodes);
        *value = nodes;
    }

    /* Parse subids and get package, core and smt counts */
    papi_free(subids);

  fn_exit:
    return status;
  fn_fail:
    papi_free(subids);
    status = CPU_ERROR;
    goto fn_exit;
}

int
cpuid_get_cache_info( CPU_attr_e attr, int level, int *value )
{
    int status = CPU_SUCCESS;
    char vendor[13] = { 0 };

    cpuid_get_vendor(vendor);

    if (!strcmp(vendor, "GenuineIntel")) {
        status = intel_get_cache_info(attr, level, value);
    } else if (!strcmp(vendor, "AuthenticAMD")) {
        status = amd_get_cache_info(attr, level, value);
    } else {
        status = CPU_ERROR;
    }

    return status;
}

int
intel_get_cache_info( CPU_attr_e attr, int level, int *value )
{
    static _sysdetect_cache_level_info_t *clevel_ptr;

    if (clevel_ptr) {
        return cpu_get_cache_info(attr, level, clevel_ptr, value);
    }

    if (!cpuid_supports_leaves_4_11()) {
        return os_cpu_get_attribute_at(attr, level, value);
    }

    clevel_ptr = clevel;

    cpuid_reg_t reg;
    int subleaf = 0;
    while(1) {
        /*
         * We query cache info only for the logical processor we are running on
         * and rely on the fact that the rest are all identical
         */
        cpuid2(&reg, 4, subleaf);

        /*
         * Decoded as per table 3-12 in Intel's Software Developer's Manual
         * Volume 2A
         */
        int type = reg.eax & 0x1f;
        if (type == 0)
            break;

        switch(type) {
            case 1:
                type = PAPI_MH_TYPE_DATA;
                break;
            case 2:
                type = PAPI_MH_TYPE_INST;
                break;
            case 3:
                type = PAPI_MH_TYPE_UNIFIED;
                break;
            default:
                type = PAPI_MH_TYPE_UNKNOWN;
        }

        int level       = (reg.eax >> 5) & 0x3;
        int fully_assoc = (reg.eax >> 9) & 0x1;
        int line_size   = (reg.ebx & 0xfff) + 1;
        int partitions  = ((reg.ebx >> 12) & 0x3ff) + 1;
        int ways        = ((reg.ebx >> 22) & 0x3ff) + 1;
        int sets        = (reg.ecx + 1);

        int *num_caches = &clevel[level-1].num_caches;
        clevel_ptr[level-1].cache[*num_caches].type = type;
        clevel_ptr[level-1].cache[*num_caches].size = (ways * partitions * sets * line_size);
        clevel_ptr[level-1].cache[*num_caches].line_size = line_size;
        clevel_ptr[level-1].cache[*num_caches].num_lines = (ways * partitions * sets);
        clevel_ptr[level-1].cache[*num_caches].associativity = (fully_assoc) ? SHRT_MAX : ways;
        ++(*num_caches);

        ++subleaf;
    }

    return cpu_get_cache_info(attr, level, clevel_ptr, value);
}

int
amd_get_cache_info( CPU_attr_e attr, int level, int *value )
{
    static _sysdetect_cache_level_info_t *clevel_ptr;

    if (clevel_ptr) {
        return cpu_get_cache_info(attr, level, clevel_ptr, value);
    }

    cpuid_reg_t reg;

    /* L1 Caches */
    cpuid(&reg, 0x80000005);

    unsigned char byt[16];
    memcpy(byt     , &reg.eax, 4);
    memcpy(byt + 4 , &reg.ebx, 4);
    memcpy(byt + 8 , &reg.ecx, 4);
    memcpy(byt + 12, &reg.edx, 4);

    clevel_ptr = clevel;
    clevel_ptr[0].cache[0].type = PAPI_MH_TYPE_DATA;
    clevel_ptr[0].cache[0].size = byt[11] << 10;
    clevel_ptr[0].cache[0].line_size = byt[8];
    clevel_ptr[0].cache[0].num_lines = clevel_ptr[0].cache[0].size / clevel_ptr[0].cache[0].line_size;
    clevel_ptr[0].cache[0].associativity = byt[10];

    clevel_ptr[0].cache[1].type = PAPI_MH_TYPE_INST;
    clevel_ptr[0].cache[1].size = byt[15] << 10;
    clevel_ptr[0].cache[1].line_size = byt[12];
    clevel_ptr[0].cache[1].num_lines = clevel_ptr[0].cache[1].size / clevel_ptr[0].cache[1].line_size;
    clevel_ptr[0].cache[1].associativity = byt[14];

    clevel_ptr[0].num_caches = 2;

    /* L2 and L3 caches */
    cpuid(&reg, 0x80000006);

    memcpy(byt     , &reg.eax, 4);
    memcpy(byt + 4 , &reg.ebx, 4);
    memcpy(byt + 8 , &reg.ecx, 4);
    memcpy(byt + 12, &reg.edx, 4);

    static short int assoc[16] = {
        0, 1, 2, -1, 4, -1, 8, -1, 16, -1, 32, 48, 64, 96, 128, SHRT_MAX
    };

    if (reg.ecx) {
        clevel_ptr[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
        clevel_ptr[1].cache[0].size = (int)((reg.ecx & 0xffff0000) >> 6);
        clevel_ptr[1].cache[0].line_size = byt[8];
        clevel_ptr[1].cache[0].num_lines = clevel_ptr[1].cache[0].size / clevel_ptr[1].cache[0].line_size;
        clevel_ptr[1].cache[0].associativity = assoc[(byt[9] & 0xf0) >> 4];
        clevel_ptr[1].num_caches = 1;
    }

    if (reg.edx) {
        clevel_ptr[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
        clevel_ptr[2].cache[0].size = (int)((reg.edx & 0xfffc0000) << 1);
        clevel_ptr[2].cache[0].line_size = byt[12];
        clevel_ptr[2].cache[0].num_lines = clevel_ptr[1].cache[0].size / clevel_ptr[1].cache[0].line_size;
        clevel_ptr[2].cache[0].associativity = assoc[(byt[13] & 0xf0) >> 4];
        clevel_ptr[2].num_caches = 1;
    }

    return cpu_get_cache_info(attr, level, clevel_ptr, value);
}

int
cpuid_supports_leaves_4_11( void )
{
    char vendor[13];
    cpuid_get_vendor(vendor);

    cpuid_reg_t reg;
    cpuid(&reg, 0);

    /* If leaf4 not supported or vendor is not Intel, fallback */
    int fallback = (reg.eax < 4 || strcmp(vendor, "GenuineIntel"));
    if (!fallback) {
        cpuid_has_leaf4 = 1;
        cpuid(&reg, 11);
        if (reg.ebx != 0)
            cpuid_has_leaf11 = 1;
    }

    return !fallback;
}

int
enum_cpu_resources( int num_mappings, apic_subid_mask_t *mask,
                    apic_subid_t *subids, int *sockets,
                    int *cores, int *threads )
{
    int status = CPU_SUCCESS;
    int max_num_pkgs    = (1 << mask->pkg_width);
    int max_num_cores   = (1 << mask->core_width);
    int max_num_threads = (1 << mask->smt_width);

    int *pkg_arr = papi_calloc(max_num_pkgs, sizeof(int));
    if (!pkg_arr)
        goto fn_fail_pkg;

    int *core_arr = papi_calloc(max_num_cores, sizeof(int));
    if (!core_arr)
        goto fn_fail_core;

    int *smt_arr = papi_calloc(max_num_threads, sizeof(int));
    if (!smt_arr)
        goto fn_fail_thread;

    int i;
    for (i = 0; i < num_mappings; ++i) {
        pkg_arr[subids[i].pkg] =
            core_arr[subids[i].core] =
                smt_arr[subids[i].smt] = 1;
    }

    i = 0, *sockets = 0;
    while (i < max_num_pkgs) {
        if (pkg_arr[i++] != 0)
            (*sockets)++;
    }

    i = 0, *cores = 0;
    while (i < max_num_cores) {
        if (core_arr[i++] != 0)
            (*cores)++;
    }

    i = 0, *threads = 0;
    while (i < max_num_threads) {
        if (smt_arr[i++] != 0)
            (*threads)++;
    }

    papi_free(pkg_arr);
    papi_free(core_arr);
    papi_free(smt_arr);

  fn_exit:
    return status;
  fn_fail_thread:
    papi_free(core_arr);
  fn_fail_core:
    papi_free(pkg_arr);
  fn_fail_pkg:
    status = CPU_ERROR;
    goto fn_exit;
}

int
cpuid_get_versioning_info( CPU_attr_e attr, int *value )
{
    static int family, model, stepping;

    if (attr == CPU_ATTR__CPUID_FAMILY && family) {
        *value = family;
        return CPU_SUCCESS;
    } else if (attr == CPU_ATTR__CPUID_MODEL && model) {
        *value = model;
        return CPU_SUCCESS;
    } else if (attr == CPU_ATTR__CPUID_STEPPING && stepping) {
        *value = stepping;
        return CPU_SUCCESS;
    }

    cpuid_reg_t reg;
    cpuid(&reg, 1);

    /* Query versioning info once and store results for later */
    family = (reg.eax >> 8) & 0x0000000f;
    model = (family == 6 || family == 15) ?
        ((reg.eax >> 4) & 0x0000000f) + ((reg.eax >> 12) & 0x000000f0) :
        ((reg.eax >> 4) & 0x0000000f);
    stepping = reg.eax & 0x0000000f;

    char vendor[13];
    cpuid_get_vendor(vendor);
    if (!strcmp(vendor, "AuthenticAMD") && family == 15) {
        /* Adjust family for AMD processors */
        family += (reg.eax >> 20) & 0x000000ff;
    }

    if (attr == CPU_ATTR__CPUID_FAMILY) {
        *value = family;
    } else if (attr == CPU_ATTR__CPUID_MODEL) {
        *value = model;
    } else {
        *value = stepping;
    }

    return CPU_SUCCESS;
}

int
cpuid_parse_id_foreach_thread( unsigned int num_mappings,
                               apic_subid_mask_t *mask,
                               apic_subid_t *subid )
{
    unsigned int apic_id = cpuid_get_apic_id();
    subid[num_mappings].pkg  = ((apic_id & mask->pkg_mask ) >> (mask->smt_width + mask->core_width));
    subid[num_mappings].core = ((apic_id & mask->core_mask) >> (mask->smt_width));
    subid[num_mappings].smt  = ((apic_id & mask->smt_mask ));
    return CPU_SUCCESS;
}

int
cpuid_parse_ids( int os_proc_count, apic_subid_mask_t *mask, apic_subid_t *subid )
{
    int i, ret = 0;
    int num_mappings = 0;

    /* save cpu affinity */
    os_cpu_store_affinity();

    for (i = 0; i < os_proc_count; ++i) {
        /* check if we are allowed to run on this logical processor */
        if (os_cpu_set_affinity(i)) {
            ret = -1;
            break;
        }

        /* now query id for the logical processor */
        cpuid_parse_id_foreach_thread(num_mappings, mask, subid);

        /* increment parsed ids */
        ret = ++num_mappings;
    }

    /* restore cpu affinity */
    os_cpu_load_affinity();

    return ret;
}

int
cpuid_get_mask( apic_subid_mask_t *mask )
{
    if (cpuid_has_leaf11) {
        return cpuid_get_leaf11_mask(mask);
    }

    return cpuid_get_leaf4_mask(mask);
}

int
cpuid_get_leaf11_mask( apic_subid_mask_t *mask )
{
    int status = CPU_SUCCESS;
    int core_reported = 0;
    int thread_reported = 0;
    int sub_leaf = 0, level_type, level_shift;
    unsigned int core_plus_smt_mask = 0;
    unsigned int core_plus_smt_width = 0;

    do {
        cpuid_reg_t reg;
        cpuid2(&reg, 11, sub_leaf);
        if (reg.ebx == 0)
            break;

        level_type  = (reg.ecx >> 8) & 0x000000ff;
        level_shift = reg.eax & 0x0000001f;

        /*
         * x2APIC ID layout (32 bits)
         * +---------+----------+---------+
         * |   pkg   |   core   |   smt   |
         * +---------+----------+---------+
         *                      <--------->
         *                       level type  = 1
         *                       level shift = smt width
         *           <-------------------->
         *            level type  = 2
         *            level shift = core + smt width
         *
         */
        switch (level_type) {
            case 1: /* level type is SMT, so the mask width is level shift */
                mask->smt_mask  = ~(0xFFFFFFFF << level_shift);
                mask->smt_width = level_shift;
                thread_reported = 1;
                break;
            case 2: /* level type is core, so the core + smt mask width is level shift */
                core_plus_smt_mask  = ~(0xFFFFFFFF << level_shift);
                core_plus_smt_width = level_shift;
                mask->pkg_mask      = 0xFFFFFFFF ^ core_plus_smt_mask;
                mask->pkg_width     = 8; /* use reasonably high value */
                core_reported       = 1;
                break;
            default:
                break;
        }
        ++sub_leaf;
    } while(1);

    if (thread_reported && core_reported) {
        mask->core_mask  = core_plus_smt_mask ^ mask->smt_mask;
        mask->core_width = core_plus_smt_width - mask->smt_width;
    } else if (!core_reported && thread_reported) {
        mask->core_mask  = 0;
        mask->core_width = 0;
        mask->pkg_mask   = 0xFFFFFFFF ^ mask->smt_mask;
        mask->pkg_width  = 8; /* use reasonably high value */
    } else {
        status = CPU_ERROR;
    }

    return status;
}

int
cpuid_get_leaf4_mask( apic_subid_mask_t *mask )
{
    cpuid_reg_t reg;
    cpuid(&reg, 1);
    unsigned int core_plus_smt_max_cnt = (reg.ebx >> 16) & 0x000000ff;

    cpuid(&reg, 4);
    unsigned int core_max_cnt = ((reg.eax >> 26) & 0x0000003f) + 1;

    unsigned int core_width = bit_width(core_max_cnt);
    unsigned int smt_width = bit_width(core_plus_smt_max_cnt) - core_width;

    mask->smt_mask   = ~(0xFFFFFFFF << smt_width);
    mask->smt_width  = smt_width;
    mask->core_mask  = ~(0xFFFFFFFF << bit_width(core_plus_smt_max_cnt)) ^ mask->smt_mask;
    mask->core_width = core_width;
    mask->pkg_mask   = 0xFFFFFFFF << bit_width(core_plus_smt_max_cnt);
    mask->pkg_width  = 8; /* use reasonably high value */

    return CPU_SUCCESS;
}

unsigned int
cpuid_get_x2apic_id( void )
{
    cpuid_reg_t reg;
    cpuid(&reg, 11);
    return reg.edx;
}

unsigned int
cpuid_get_apic_id( void )
{
    if (cpuid_has_leaf11) {
        return cpuid_get_x2apic_id();
    }

    cpuid_reg_t reg;
    cpuid(&reg, 1);
    return (reg.ebx >> 24) & 0x000000ff;
}

unsigned int
bit_width( unsigned int x )
{
    int count = 0;
    double y = (double) x;

    while ((y /= 2) > 1) {
        ++count;
    }

    return (y < 1) ? count + 1 : count;
}

void
cpuid2( cpuid_reg_t *reg, unsigned int func, unsigned int subfunc )
{
    __asm__ ("cpuid;"
             : "=a" (reg->eax), "=b" (reg->ebx), "=c" (reg->ecx), "=d" (reg->edx)
             : "a"  (func), "c" (subfunc));
}

void
cpuid( cpuid_reg_t *reg, unsigned int func )
{
    cpuid2(reg, func, 0);
}
