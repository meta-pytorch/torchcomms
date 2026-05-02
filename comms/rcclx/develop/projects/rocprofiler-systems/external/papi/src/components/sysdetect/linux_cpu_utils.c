#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>

#include "sysdetect.h"
#include "linux_cpu_utils.h"

#define VENDOR_UNKNOWN       -1
#define VENDOR_UNINITED      0
#define VENDOR_INTEL_X86     1
#define VENDOR_AMD           2
#define VENDOR_IBM           3
#define VENDOR_CRAY          4
#define VENDOR_MIPS          8
#define VENDOR_INTEL_IA64    9
#define VENDOR_ARM_ARM       65
#define VENDOR_ARM_BROADCOM  66
#define VENDOR_ARM_CAVIUM    67
#define VENDOR_ARM_FUJITSU   70
#define VENDOR_ARM_HISILICON 72
#define VENDOR_ARM_APM       80
#define VENDOR_ARM_QUALCOMM  81

#define _PATH_SYS_SYSTEM "/sys/devices/system/"
#define _PATH_SYS_CPU0   _PATH_SYS_SYSTEM "/cpu/cpu0"

static int get_topology_info( const char *key, int *value );
static int get_naming_info( const char *key, char *value );
static int get_versioning_info( const char *key, int *value );
static int get_cache_info( CPU_attr_e attr, int level, int *value );
static int get_cache_level( const char *dirname, int *value );
static int get_cache_type( const char *dirname, int *value );
static int get_cache_size( const char *dirname, int *value );
static int get_cache_line_size( const char *dirname, int *value );
static int get_cache_associativity( const char *dirname, int *value );
static int get_cache_partition_count( const char *dirname, int *value );
static int get_cache_set_count( const char *dirname, int *value );
static int get_mem_info( int node, int *value );
static int get_thread_affinity( int thread, int *value );
static int path_sibling( const char *path, ... );
static char *search_cpu_info( FILE *fp, const char *key );
static int path_exist( const char *path, ... );
static void decode_vendor_string( char *s, int *vendor );
static int get_vendor_id( void );

int
linux_cpu_get_vendor( char *vendor )
{
    const char *namekey_x86  = "vendor_id";
    const char *namekey_ia64 = "vendor";
    const char *namekey_ibm  = "platform";
    const char *namekey_mips = "system type";
    const char *namekey_arm  = "CPU implementer";
    const char *namekey_dum  = "none";

    const char *namekey_ptr = NULL;

    int vendor_id = get_vendor_id();

    if (vendor_id == VENDOR_INTEL_X86 || vendor_id == VENDOR_AMD) {
        namekey_ptr = namekey_x86;
    } else if (vendor_id == VENDOR_INTEL_IA64) {
        namekey_ptr = namekey_ia64;
    } else if (vendor_id == VENDOR_IBM) {
        namekey_ptr = namekey_ibm;
    } else if (vendor_id == VENDOR_MIPS) {
        namekey_ptr = namekey_mips;
    } else if (vendor_id == VENDOR_ARM_ARM       ||
               vendor_id == VENDOR_ARM_BROADCOM  ||
               vendor_id == VENDOR_ARM_CAVIUM    ||
               vendor_id == VENDOR_ARM_FUJITSU   ||
               vendor_id == VENDOR_ARM_HISILICON ||
               vendor_id == VENDOR_ARM_APM       ||
               vendor_id == VENDOR_ARM_QUALCOMM) {
        namekey_ptr = namekey_arm;
    } else {
        namekey_ptr = namekey_dum;
    }

    return get_naming_info(namekey_ptr, vendor);
}

int
linux_cpu_get_name( char *name )
{
    const char *namekey_x86 = "model name";
    const char *namekey_ibm = "model";
    const char *namekey_arm = "model name";
    const char *namekey_dum = "none";

    const char *namekey_ptr = NULL;

    int vendor_id = get_vendor_id();

    if (vendor_id == VENDOR_INTEL_X86 || vendor_id == VENDOR_AMD) {
        namekey_ptr = namekey_x86;
    } else if (vendor_id == VENDOR_IBM) {
        namekey_ptr = namekey_ibm;
    } else if (vendor_id == VENDOR_ARM_ARM       ||
               vendor_id == VENDOR_ARM_BROADCOM  ||
               vendor_id == VENDOR_ARM_CAVIUM    ||
               vendor_id == VENDOR_ARM_FUJITSU   ||
               vendor_id == VENDOR_ARM_HISILICON ||
               vendor_id == VENDOR_ARM_APM       ||
               vendor_id == VENDOR_ARM_QUALCOMM) {
        namekey_ptr = namekey_arm;
    } else {
        namekey_ptr = namekey_dum;
    }

    return get_naming_info(namekey_ptr, name);
}

int
linux_cpu_get_attribute( CPU_attr_e attr, int *value )
{
    int status = CPU_SUCCESS;

#define TOPOKEY_NUM_KEY 4
#define VERKEY_NUM_KEY 4

    int topo_idx = TOPOKEY_NUM_KEY;
    int ver_idx = VERKEY_NUM_KEY;

    const char *topokey[TOPOKEY_NUM_KEY] = {
        "sockets",
        "nodes",
        "threads",
        "cores",
    };

    const char *verkey_x86[VERKEY_NUM_KEY] = {
        "cpu family",       /* cpuid_family */
        "model",            /* cpuid_model */
        "stepping",         /* cpuid_stepping */
        "vendor_id",        /* vendor id */
    };

    const char *verkey_ibm[VERKEY_NUM_KEY] = {
        "none",             /* cpuid_family */
        "none",             /* cpuid_model */
        "revision",         /* cpuid_stepping */
        "vendor_id",        /* vendor id */
    };

    const char *verkey_arm[VERKEY_NUM_KEY] = {
        "CPU architecture", /* cpuid_family */
        "CPU part",         /* cpuid_model */
        "CPU variant",      /* cpuid_stepping */
        "CPU implementer",  /* vendor id */
    };

    const char *verkey_dum[VERKEY_NUM_KEY] = {
        "none",
        "none",
        "none",
        "none",
    };

    const char **verkey_ptr = NULL;

    int vendor_id = get_vendor_id();

    if (vendor_id == VENDOR_INTEL_X86 || vendor_id == VENDOR_AMD) {
        verkey_ptr = verkey_x86;
    } else if (vendor_id == VENDOR_IBM) {
        verkey_ptr = verkey_ibm;
    } else if (vendor_id == VENDOR_ARM_ARM       ||
               vendor_id == VENDOR_ARM_BROADCOM  ||
               vendor_id == VENDOR_ARM_CAVIUM    ||
               vendor_id == VENDOR_ARM_FUJITSU   ||
               vendor_id == VENDOR_ARM_HISILICON ||
               vendor_id == VENDOR_ARM_APM       ||
               vendor_id == VENDOR_ARM_QUALCOMM) {
        verkey_ptr = verkey_arm;
    } else {
        verkey_ptr = verkey_dum;
    }

    switch(attr) {
        case CPU_ATTR__NUM_SOCKETS:
            --topo_idx;
            // fall through
        case CPU_ATTR__NUM_NODES:
            --topo_idx;
            // fall through
        case CPU_ATTR__NUM_THREADS:
            --topo_idx;
            // fall through
        case CPU_ATTR__NUM_CORES:
            --topo_idx;
            status = get_topology_info(topokey[topo_idx], value);
            break;
        case CPU_ATTR__CPUID_FAMILY:
            --ver_idx;
            // fall through
        case CPU_ATTR__CPUID_MODEL:
            --ver_idx;
            // fall through
        case CPU_ATTR__CPUID_STEPPING:
            --ver_idx;
            // fall through
        case CPU_ATTR__VENDOR_ID:
            --ver_idx;
            status = get_versioning_info(verkey_ptr[ver_idx], value);
            break;
        case CPU_ATTR__CACHE_MAX_NUM_LEVELS:
            *value = PAPI_MAX_MEM_HIERARCHY_LEVELS;
            break;
        default:
            status = CPU_ERROR;
    }

    return status;
}

int
linux_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
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
            status = get_mem_info(loc, value);
            break;
        case CPU_ATTR__HWTHREAD_NUMA_AFFINITY:
            status = get_thread_affinity(loc, value);
            break;
        default:
            status = CPU_ERROR;
    }

    return status;
}

int
linux_cpu_set_affinity( int cpu )
{
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

int
linux_cpu_get_num_supported( void )
{
    return sysconf(_SC_NPROCESSORS_CONF);
}

static cpu_set_t saved_affinity;

int
linux_cpu_store_affinity( void )
{
    if (!CPU_COUNT(&saved_affinity))
        return sched_getaffinity(0, sizeof(cpu_set_t), &saved_affinity);

    return CPU_SUCCESS;
}

int
linux_cpu_load_affinity( void )
{
    return sched_setaffinity(0, sizeof(cpu_set_t), &saved_affinity);
}

int
get_topology_info( const char *key, int *val )
{
    int status = CPU_SUCCESS;
    static int sockets, nodes, threads, cores;

    if (!strcmp("sockets", key) && sockets) {
        *val = sockets;
        return status;
    } else if (!strcmp("nodes", key) && nodes) {
        *val = nodes;
        return status;
    } else if (!strcmp("threads", key) && threads) {
        *val = threads;
        return status;
    } else if (!strcmp("cores", key) && cores) {
        *val = cores;
        return status;
    }

    /* Query topology information once and store results for later */
    int totalcpus = 0;

    while (path_exist(_PATH_SYS_SYSTEM "/cpu/cpu%d", totalcpus)) {
        ++totalcpus;
    }

    if (path_exist(_PATH_SYS_CPU0 "/topology/thread_siblings")) {
        threads = path_sibling(_PATH_SYS_CPU0 "/topology/thread_siblings");
    }

    if (path_exist(_PATH_SYS_CPU0 "/topology/core_siblings")) {
        cores = path_sibling(_PATH_SYS_CPU0 "/topology/core_siblings") / threads;
    }

    sockets = totalcpus / cores / threads;

    while (path_exist(_PATH_SYS_SYSTEM "/node/node%d", nodes))
        ++nodes;

    if (!strcmp("sockets", key)) {
        *val = sockets;
    } else if (!strcmp("nodes", key)) {
        *val = (nodes == 0) ? nodes = 1 : nodes;
    } else if (!strcmp("cores", key)) {
        *val = cores;
    } else if (!strcmp("threads", key)) {
        *val = threads;
    } else {
        status = CPU_ERROR;
    }

    return status;
}

int
get_naming_info( const char *key, char *val )
{
    if (!strcmp(key, "none")) {
        strcpy(val, "UNKNOWN");
        return CPU_SUCCESS;
    }

    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) {
        return CPU_ERROR;
    }

    char *str = search_cpu_info(fp, key);
    if (str) {
        strncpy(val, str, PAPI_MAX_STR_LEN);
        val[PAPI_MAX_STR_LEN - 1] = 0;
    }

    fclose(fp);

    return CPU_SUCCESS;
}

int
get_versioning_info( const char *key, int *val )
{
    if (!strcmp(key, "none")) {
        *val = -1;
        return CPU_SUCCESS;
    }

    if (!strcmp(key, "vendor_id") || !strcmp(key, "CPU implementer")) {
        *val = get_vendor_id();
        return CPU_SUCCESS;
    }

    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) {
        return CPU_ERROR;
    }

    char *str = search_cpu_info(fp, key);
    if (str) {
        /* FIXME: this is a ugly hack to handle old (prior to Linux 3.19) ARM64 */
        if (strcmp(key, "CPU architecture") == 0) {
            /* Prior version 3.19 'CPU architecture' is always 'AArch64'
             * so we convert it to '8', which is the value since 3.19. */
            if (strstr(str, "AArch64"))
                *val = 8;
            else
                *val = strtol(str, NULL, 10);

            /* Old Fallbacks if the above didn't work (e.g. Raspberry Pi) */
            if (*val < 0) {
                str = search_cpu_info(fp, "Processor");
                if (str) {
                    char *t = strchr(str, '(');
                    int tmp = *(t + 2) - '0';
                    *val = tmp;
                } else {
                    /* Try the model name and look inside of parens */
                    str = search_cpu_info(fp, "model name");
                    if (str) {
                        char *t = strchr(str, '(');
                        int tmp = *(t + 2) - '0';
                        *val = tmp;
                    }
                }
            }
        } else {
            sscanf(str, "%x", val);
        }
    }

    fclose(fp);

    return CPU_SUCCESS;
}

static _sysdetect_cache_level_info_t clevel[PAPI_MAX_MEM_HIERARCHY_LEVELS];

int
get_cache_info( CPU_attr_e attr, int level, int *val )
{
    int type = 0;
    int size, line_size, associativity, sets;
    DIR *dir;
    struct dirent *d;
    int max_level = 0;
    int *level_count, level_index;
    static _sysdetect_cache_level_info_t *L;

    if (L) {
        return cpu_get_cache_info(attr, level, L, val);
    }

    L = clevel;

    /* open Linux cache dir                 */
    /* assume all CPUs same as cpu0.        */
    /* Not necessarily a good assumption    */

    dir = opendir("/sys/devices/system/cpu/cpu0/cache");
    if (dir == NULL) {
        goto fn_fail;
    }

    while(1) {
        d = readdir(dir);
        if (d == NULL)
            break;

        if (strncmp(d->d_name, "index", 5))
            continue;

        if (get_cache_level(d->d_name, &level_index)) {
            goto fn_fail;
        }

        if (get_cache_type(d->d_name, &type)) {
            goto fn_fail;
        }
        level_count = &L[level_index].num_caches;
        L[level_index].cache[*level_count].type = type;

        if (get_cache_size(d->d_name, &size)) {
            goto fn_fail;
        }
        /* Linux reports in kB, PAPI expects in Bytes */
        L[level_index].cache[*level_count].size = size * 1024;

        if (get_cache_line_size(d->d_name, &line_size)) {
            goto fn_fail;
        }
        L[level_index].cache[*level_count].line_size = line_size;

        if (get_cache_associativity(d->d_name, &associativity)) {
            goto fn_fail;
        }
        L[level_index].cache[*level_count].associativity = associativity;

        int partitions;
        if (get_cache_partition_count(d->d_name, &partitions)) {
            goto fn_fail;
        }

        if (get_cache_set_count(d->d_name, &sets)) {
            goto fn_fail;
        }
        L[level_index].cache[*level_count].num_lines =
            (sets * associativity * partitions);

        if (((size * 1024) / line_size / associativity) != sets) {
            MEMDBG("Warning!  sets %d != expected %d\n",
                   sets, ((size * 1024) / line_size / associativity));
        }

        if (level > max_level) {
            max_level = level;
        }

        if (level >= PAPI_MAX_MEM_HIERARCHY_LEVELS) {
            MEMDBG("Exceeded maximum cache level %d\n",
                   PAPI_MAX_MEM_HIERARCHY_LEVELS);
            break;
        }

        ++(*level_count);
    }

    closedir(dir);
    return cpu_get_cache_info(attr, level, L, val);

  fn_fail:
    closedir(dir);
    return CPU_ERROR;
}

int
get_cache_level( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int level_index;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/level",
            dirname);

    FILE *fff = fopen(filename,"r");
    if (fff == NULL) {
        MEMDBG("Cannot open level.\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &level_index);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read cache level\n");
        return CPU_ERROR;
    }

    /* Index arrays from 0 */
    level_index -= 1;
    *value = level_index;

    return CPU_SUCCESS;
}

int
get_cache_type( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    char type_string[BUFSIZ];
    char buffer[BUFSIZ];
    int type = PAPI_MH_TYPE_EMPTY;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/type",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open type\n");
        return CPU_ERROR;
    }

    char *result = fgets(buffer, BUFSIZ, fff);
    fclose(fff);
    if (result == NULL) {
        MEMDBG("Could not read cache type\n");
        return CPU_ERROR;
    }

    sscanf(buffer,"%s",type_string);

    if (!strcmp(type_string, "Data")) {
        type = PAPI_MH_TYPE_DATA;
    }

    if (!strcmp(type_string, "Instruction")) {
        type = PAPI_MH_TYPE_INST;
    }

    if (!strcmp(type_string, "Unified")) {
        type = PAPI_MH_TYPE_UNIFIED;
    }

    *value = type;

    return CPU_SUCCESS;
}

int
get_cache_size( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int size;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/size",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open size\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &size);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read cache size\n");
        return CPU_ERROR;
    }

    *value = size;

    return CPU_SUCCESS;
}

int
get_cache_line_size( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int line_size;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/coherency_line_size",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open linesize\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &line_size);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read cache line-size\n");
        return CPU_ERROR;
    }

    *value = line_size;

    return CPU_SUCCESS;
}

int
get_cache_associativity( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int associativity;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/ways_of_associativity",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open associativity\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &associativity);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read cache associativity\n");
        return CPU_ERROR;
    }

    *value = associativity;

    return CPU_SUCCESS;
}

int
get_cache_partition_count( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int partitions;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/physical_line_partition",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open partitions\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &partitions);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read partitions count\n");
        return CPU_ERROR;
    }

    *value = partitions;

    return CPU_SUCCESS;
}

int
get_cache_set_count( const char *dirname, int *value )
{
    char filename[BUFSIZ];
    int sets;

    sprintf(filename, "/sys/devices/system/cpu/cpu0/cache/%s/number_of_sets",
            dirname);

    FILE *fff = fopen(filename, "r");
    if (fff == NULL) {
        MEMDBG("Cannot open sets\n");
        return CPU_ERROR;
    }

    int result = fscanf(fff, "%d", &sets);
    fclose(fff);
    if (result != 1) {
        MEMDBG("Could not read cache sets\n");
        return CPU_ERROR;
    }

    *value = sets;

    return CPU_SUCCESS;
}

int
get_mem_info( int node, int *val )
{
    if (path_exist(_PATH_SYS_SYSTEM "/node/node%d", node)) {
        char filename[PAPI_MAX_STR_LEN];
        sprintf(filename, _PATH_SYS_SYSTEM "/node/node%d/meminfo", node);
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            return CPU_ERROR;
        }

        char search_str[PAPI_MIN_STR_LEN];
        sprintf(search_str, "Node %d MemTotal", node);
        char *str = search_cpu_info(fp, search_str);
        if (str) {
            sprintf(search_str, "%s", str);
            int len = strlen(search_str);
            search_str[len-3] = '\0'; /* Remove trailing "KB" */
            *val = atoi(search_str);
        }

        fclose(fp);
    }

    return CPU_SUCCESS;
}

int
get_thread_affinity( int thread, int *val )
{
    if (!path_exist(_PATH_SYS_SYSTEM "/cpu/cpu0/node0")) {
        *val = 0;
        return CPU_SUCCESS;
    }

    // If gaps exist in the core numbering, the caller of this
    // fucntion will likely inquire about cpu-ids that do not
    // exist in the system (i.e., the gaps).
    if( !path_exist(_PATH_SYS_SYSTEM "/cpu/cpu%d", thread) ){
        *val = -1;
        return CPU_ERROR;
    }

    int i = 0;
    while (!path_exist(_PATH_SYS_SYSTEM "/cpu/cpu%d/node%d", thread, i)) {
        ++i;
    }
    *val = i;

    return CPU_SUCCESS;
}

static char pathbuf[PATH_MAX] = "/";

FILE *
xfopen( const char *path, const char *mode )
{
    FILE *fd = fopen(path, mode);
    return fd;
}

FILE *
path_vfopen( const char *mode, const char *path, va_list ap )
{
    vsnprintf( pathbuf, sizeof ( pathbuf ), path, ap );
    return xfopen( pathbuf, mode );
}

int
path_sibling( const char *path, ... )
{
    int c;
    long n;
    int result = CPU_SUCCESS;
    char s[2];
    FILE *fp;
    va_list ap;
    va_start( ap, path );
    fp = path_vfopen( "r", path, ap );
    va_end( ap );

    while ((c = fgetc(fp)) != EOF) {
        if (isxdigit(c)) {
            s[0] = (char) c;
            s[1] = '\0';
            for (n = strtol(s, NULL, 16); n > 0; n /= 2) {
                if (n % 2)
                    result++;
            }
        }
    }

    fclose(fp);
    return result;
}

char *
search_cpu_info( FILE * f, const char *search_str )
{
    static char line[PAPI_HUGE_STR_LEN] = "";
    char *s, *start = NULL;

    rewind(f);

    while (fgets(line, PAPI_HUGE_STR_LEN,f) != NULL) {
        s=strstr(line, search_str);
        if (s != NULL) {
            /* skip all characters in line up to the colon */
            /* and then spaces */
            s=strchr(s, ':');
            if (s == NULL) break;
            s++;
            while (isspace(*s)) {
                s++;
            }
            start = s;
            /* Find and clear newline */
            s=strrchr(start, '\n');
            if (s != NULL) *s = 0;
            break;
        }
    }

    return start;
}

int
path_exist( const char *path, ... )
{
    va_list ap;
    va_start(ap, path);
    vsnprintf(pathbuf, sizeof ( pathbuf ), path, ap);
    va_end(ap);
    return access(pathbuf, F_OK) == 0;
}

void
decode_vendor_string( char *s, int *vendor )
{
    if (strcasecmp(s, "GenuineIntel") == 0)
        *vendor = VENDOR_INTEL_X86;
    else if ((strcasecmp(s, "AMD") == 0) ||
             (strcasecmp(s, "AuthenticAMD") == 0 ))
        *vendor = VENDOR_AMD;
    else if (strcasecmp(s, "IBM") == 0)
        *vendor = VENDOR_IBM;
    else if (strcasecmp(s, "Cray") == 0)
        *vendor = VENDOR_CRAY;
    else if (strcasecmp(s, "ARM_ARM") == 0)
        *vendor = VENDOR_ARM_ARM;
    else if (strcasecmp(s, "ARM_BROADCOM") == 0)
        *vendor = VENDOR_ARM_BROADCOM;
    else if (strcasecmp(s, "ARM_CAVIUM") == 0)
        *vendor = VENDOR_ARM_CAVIUM;
    else if (strcasecmp(s, "ARM_FUJITSU") == 0)
        *vendor = VENDOR_ARM_FUJITSU;
    else if (strcasecmp(s, "ARM_HISILICON") == 0)
        *vendor = VENDOR_ARM_HISILICON;
    else if (strcasecmp(s, "ARM_APM") == 0)
        *vendor = VENDOR_ARM_APM;
    else if (strcasecmp(s, "ARM_QUALCOMM") == 0)
        *vendor = VENDOR_ARM_QUALCOMM;
    else if (strcasecmp(s, "MIPS") == 0)
        *vendor = VENDOR_MIPS;
    else if (strcasecmp(s, "SiCortex") == 0)
        *vendor = VENDOR_MIPS;
    else
        *vendor = VENDOR_UNKNOWN;
}

int
get_vendor_id( void )
{
    static int vendor_id; // VENDOR_UNINITED;

    if (vendor_id != VENDOR_UNINITED)
        return vendor_id;

    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) {
        return CPU_ERROR;
    }

    char vendor_string[PAPI_MAX_STR_LEN] = "";
    char *s = search_cpu_info(fp, "vendor_id");
    if (s) {
        strncpy(vendor_string, s, PAPI_MAX_STR_LEN);
        vendor_string[PAPI_MAX_STR_LEN - 1] = 0;
    } else {
        s = search_cpu_info(fp, "vendor");
        if (s) {
            strncpy(vendor_string, s, PAPI_MAX_STR_LEN);
            vendor_string[PAPI_MAX_STR_LEN - 1] = 0;
        } else {
            s = search_cpu_info(fp, "system type");
            if (s) {
                strncpy(vendor_string, s, PAPI_MAX_STR_LEN);
                vendor_string[PAPI_MAX_STR_LEN - 1] = 0;
            } else {
                s = search_cpu_info(fp, "platform");
                if (s) {
                    if (strcasecmp(s, "pSeries") == 0 ||
                        strcasecmp(s, "PowerNV") == 0 ||
                        strcasecmp(s, "PowerMac") == 0) {
                        strcpy(vendor_string, "IBM");
                    }
                } else {
                    s = search_cpu_info(fp, "CPU implementer");
                    if (s) {
                        int tmp;
                        sscanf(s, "%x", &tmp);
                        switch(tmp) {
                            case VENDOR_ARM_ARM:
                                strcpy(vendor_string, "ARM_ARM");
                                break;
                            case VENDOR_ARM_BROADCOM:
                                strcpy(vendor_string, "ARM_BROADCOM");
                                break;
                            case VENDOR_ARM_CAVIUM:
                                strcpy(vendor_string, "ARM_CAVIUM");
                                break;
                            case VENDOR_ARM_FUJITSU:
                                strcpy(vendor_string, "ARM_FUJITSU");
                                break;
                            case VENDOR_ARM_HISILICON:
                                strcpy(vendor_string, "ARM_HISILICON");
                                break;
                            case VENDOR_ARM_APM:
                                strcpy(vendor_string, "ARM_APM");
                                break;
                            case VENDOR_ARM_QUALCOMM:
                                strcpy(vendor_string, "ARM_QUALCOMM");
                                break;
                            default:
                                strcpy(vendor_string, "UNKNOWN");
                        }
                    }
                }
            }
        }
    }

    if (strlen(vendor_string)) {
        decode_vendor_string(vendor_string, &vendor_id);
    }

    fclose(fp);
    return vendor_id;
}
