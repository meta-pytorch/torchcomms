/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@eecs.utk.edu
*          cache and TLB info exported to a separate file
*          which is not OS or driver dependent
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu
*          Merge all of the various copies of linux-related
*          memory detection info this file.
*/

#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h" /* papi_calloc() */

#include "x86_cpuid_info.h"

#include "linux-lock.h"

/* 2.6.19 has this:
VmPeak:     4588 kB
VmSize:     4584 kB
VmLck:         0 kB
VmHWM:      1548 kB
VmRSS:      1548 kB
VmData:      312 kB
VmStk:        88 kB
VmExe:       684 kB
VmLib:      1360 kB
VmPTE:        20 kB
*/

int
generic_get_memory_info( PAPI_hw_info_t *hw_info );

int
_linux_get_dmem_info( PAPI_dmem_info_t * d )
{
    char fn[PATH_MAX], tmp[PATH_MAX];
    FILE *f;
    int ret;
    long long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat =
        0, dum = 0, lib = 0, hwm = 0;

    sprintf( fn, "/proc/%ld/status", ( long ) getpid(  ) );
    f = fopen( fn, "r" );
    if ( f == NULL ) {
        PAPIERROR( "fopen(%s): %s\n", fn, strerror( errno ) );
        return PAPI_ESYS;
    }
    while ( 1 ) {
        if ( fgets( tmp, PATH_MAX, f ) == NULL )
            break;
        if ( strspn( tmp, "VmSize:" ) == strlen( "VmSize:" ) ) {
            sscanf( tmp + strlen( "VmSize:" ), "%lld", &sz );
            d->size = sz;
            continue;
        }
        if ( strspn( tmp, "VmHWM:" ) == strlen( "VmHWM:" ) ) {
            sscanf( tmp + strlen( "VmHWM:" ), "%lld", &hwm );
            d->high_water_mark = hwm;
            continue;
        }
        if ( strspn( tmp, "VmLck:" ) == strlen( "VmLck:" ) ) {
            sscanf( tmp + strlen( "VmLck:" ), "%lld", &lck );
            d->locked = lck;
            continue;
        }
        if ( strspn( tmp, "VmRSS:" ) == strlen( "VmRSS:" ) ) {
            sscanf( tmp + strlen( "VmRSS:" ), "%lld", &res );
            d->resident = res;
            continue;
        }
        if ( strspn( tmp, "VmData:" ) == strlen( "VmData:" ) ) {
            sscanf( tmp + strlen( "VmData:" ), "%lld", &dat );
            d->heap = dat;
            continue;
        }
        if ( strspn( tmp, "VmStk:" ) == strlen( "VmStk:" ) ) {
            sscanf( tmp + strlen( "VmStk:" ), "%lld", &stk );
            d->stack = stk;
            continue;
        }
        if ( strspn( tmp, "VmExe:" ) == strlen( "VmExe:" ) ) {
            sscanf( tmp + strlen( "VmExe:" ), "%lld", &txt );
            d->text = txt;
            continue;
        }
        if ( strspn( tmp, "VmLib:" ) == strlen( "VmLib:" ) ) {
            sscanf( tmp + strlen( "VmLib:" ), "%lld", &lib );
            d->library = lib;
            continue;
        }
    }
    fclose( f );

    sprintf( fn, "/proc/%ld/statm", ( long ) getpid(  ) );
    f = fopen( fn, "r" );
    if ( f == NULL ) {
        PAPIERROR( "fopen(%s): %s\n", fn, strerror( errno ) );
        return PAPI_ESYS;
    }
    ret =
        fscanf( f, "%lld %lld %lld %lld %lld %lld %lld", &dum, &dum, &shr, &dum,
                &dum, &dat, &dum );
    if ( ret != 7 ) {
        PAPIERROR( "fscanf(7 items): %d\n", ret );
        fclose(f);
        return PAPI_ESYS;
    }
    d->pagesize = getpagesize(  );
    d->shared = ( shr * d->pagesize ) / 1024;
    fclose( f );

    return PAPI_OK;
}

/*
 * Architecture-specific cache detection code 
 */


#if defined(__i386__)||defined(__x86_64__)
static int
x86_get_memory_info( PAPI_hw_info_t * hw_info )
{
    int retval = PAPI_OK;

    switch ( hw_info->vendor ) {
    case PAPI_VENDOR_AMD:
    case PAPI_VENDOR_INTEL:
        retval = _x86_cache_info( &hw_info->mem_hierarchy );
        break;
    default:
        PAPIERROR( "Unknown vendor in memory information call for x86." );
        return PAPI_ENOIMPL;
    }
    return retval;
}
#endif

#if defined(__ia64__)
static int
get_number( char *buf )
{
    char numbers[] = "0123456789";
    int num;
    char *tmp, *end;

    tmp = strpbrk( buf, numbers );
    if ( tmp != NULL ) {
        end = tmp;
        while ( isdigit( *end ) )
            end++;
        *end = '\0';
        num = atoi( tmp );
        return num;
    }

    PAPIERROR( "Number could not be parsed from %s", buf );
    return -1;
}

static void
fline( FILE * fp, char *rline )
{
    char *tmp, *end, c;

    tmp = rline;
    end = &rline[1023];

    memset( rline, '\0', 1024 );

    do {
        if ( feof( fp ) )
            return;
        c = getc( fp );
    }
    while ( isspace( c ) || c == '\n' || c == '\r' );

    ungetc( c, fp );

    for ( ;; ) {
        if ( feof( fp ) ) {
            return;
        }
        c = getc( fp );
        if ( c == '\n' || c == '\r' )
            break;
        *tmp++ = c;
        if ( tmp == end ) {
            *tmp = '\0';
            return;
        }
    }
    return;
}

static int
ia64_get_memory_info( PAPI_hw_info_t * hw_info )
{
    int retval = 0;
    FILE *f;
    int clevel = 0, cindex = -1;
    char buf[1024];
    int num, i, j;
    PAPI_mh_info_t *meminfo = &hw_info->mem_hierarchy;
    PAPI_mh_level_t *L = hw_info->mem_hierarchy.level;

    f = fopen( "/proc/pal/cpu0/cache_info", "r" );

    if ( !f ) {
        PAPIERROR( "fopen(/proc/pal/cpu0/cache_info) returned < 0" );
        return PAPI_ESYS;
    }

    while ( !feof( f ) ) {
        fline( f, buf );
        if ( buf[0] == '\0' )
            break;
        if ( !strncmp( buf, "Data Cache", 10 ) ) {
            cindex = 1;
            clevel = get_number( buf );
            L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_DATA;
        } else if ( !strncmp( buf, "Instruction Cache", 17 ) ) {
            cindex = 0;
            clevel = get_number( buf );
            L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_INST;
        } else if ( !strncmp( buf, "Data/Instruction Cache", 22 ) ) {
            cindex = 0;
            clevel = get_number( buf );
            L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_UNIFIED;
        } else {
            if ( ( clevel == 0 || clevel > 3 ) && cindex >= 0 ) {
                PAPIERROR
                    ( "Cache type could not be recognized, please send /proc/pal/cpu0/cache_info" );
                return PAPI_EBUG;
            }

            if ( !strncmp( buf, "Size", 4 ) ) {
                num = get_number( buf );
                L[clevel - 1].cache[cindex].size = num;
            } else if ( !strncmp( buf, "Associativity", 13 ) ) {
                num = get_number( buf );
                L[clevel - 1].cache[cindex].associativity = num;
            } else if ( !strncmp( buf, "Line size", 9 ) ) {
                num = get_number( buf );
                L[clevel - 1].cache[cindex].line_size = num;
                L[clevel - 1].cache[cindex].num_lines =
                    L[clevel - 1].cache[cindex].size / num;
            }
        }
    }

    fclose( f );

    f = fopen( "/proc/pal/cpu0/vm_info", "r" );
    /* No errors on fopen as I am not sure this is always on the systems */
    if ( f != NULL ) {
        cindex = -1;
        clevel = 0;
        while ( !feof( f ) ) {
            fline( f, buf );
            if ( buf[0] == '\0' )
                break;
            if ( !strncmp( buf, "Data Translation", 16 ) ) {
                cindex = 1;
                clevel = get_number( buf );
                L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_DATA;
            } else if ( !strncmp( buf, "Instruction Translation", 23 ) ) {
                cindex = 0;
                clevel = get_number( buf );
                L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_INST;
            } else {
                if ( ( clevel == 0 || clevel > 2 ) && cindex >= 0 ) {
                    PAPIERROR
                        ( "TLB type could not be recognized, send /proc/pal/cpu0/vm_info" );
                    return PAPI_EBUG;
                }

                if ( !strncmp( buf, "Number of entries", 17 ) ) {
                    num = get_number( buf );
                    L[clevel - 1].tlb[cindex].num_entries = num;
                } else if ( !strncmp( buf, "Associativity", 13 ) ) {
                    num = get_number( buf );
                    L[clevel - 1].tlb[cindex].associativity = num;
                }
            }
        }
        fclose( f );
    }

    /* Compute and store the number of levels of hierarchy actually used */
    for ( i = 0; i < PAPI_MH_MAX_LEVELS; i++ ) {
        for ( j = 0; j < 2; j++ ) {
            if ( L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
                 L[i].cache[j].type != PAPI_MH_TYPE_EMPTY )
                meminfo->levels = i + 1;
        }
    }
    return retval;
}
#endif

#if defined(__powerpc__)

PAPI_mh_info_t sys_mem_info[] = {
    {2,                         // 970 begin
     {
      {                         // level 1 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_UNIFIED, 1024, 4, 0}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_INST, 65536, 128, 512, 1}
        ,
        {PAPI_MH_TYPE_DATA, 32768, 128, 256, 2}
        }
       }
      ,
      {                         // level 2 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_UNIFIED, 524288, 128, 4096, 8}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
       }
      ,
      }
     }
    ,                         // 970 end
    {3,
     {
      {                         // level 1 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_UNIFIED, 1024, 4, 0}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_INST, 65536, 128, 512, 2}
        ,
        {PAPI_MH_TYPE_DATA, 32768, 128, 256, 4}
        }
       }
      ,
      {                         // level 2 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_UNIFIED, 1966080, 128, 15360, 10}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
       }
      ,
      {                         // level 3 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_UNIFIED, 37748736, 256, 147456, 12}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
       }
      ,
      }
     }
    ,                         // POWER5 end
    {3,
     {
      {                         // level 1 begins
       {                     // tlb's begin
        /// POWER6 has an ERAT (Effective to Real Address
        /// Translation) instead of a TLB.  For the purposes of this
        /// data, we will treat it like a TLB.
        {PAPI_MH_TYPE_INST, 128, 2, 0}
        ,
        {PAPI_MH_TYPE_DATA, 128, 128, 0}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_INST, 65536, 128, 512, 4}
        ,
        {PAPI_MH_TYPE_DATA, 65536, 128, 512, 8}
        }
       }
      ,
      {                         // level 2 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        {PAPI_MH_TYPE_UNIFIED, 4194304, 128, 16384, 8}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
       }
      ,
      {                         // level 3 begins
       {                     // tlb's begin
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1}
        }
       ,
       {                     // caches begin
        /// POWER6 has a 2 slice L3 cache.  Each slice is 16MB, so
        /// combined they are 32MB and usable by each core.  For
        /// this reason, we will treat it as a single 32MB cache.
        {PAPI_MH_TYPE_UNIFIED, 33554432, 128, 262144, 16}
        ,
        {PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
        }
       }
      ,
      }
     }
    ,                         // POWER6 end
    {3,
     {
      [0] = { // level 1 begins
        .tlb = {
        /// POWER7 has an ERAT (Effective to Real Address
        /// Translation) instead of a TLB.  For the purposes of this
        /// data, we will treat it like a TLB.
        [0] = { .type = PAPI_MH_TYPE_INST,
                .num_entries = 64, .page_size = 0, .associativity = 2 }
        ,
        [1] = { .type = PAPI_MH_TYPE_DATA,
                .num_entries = 64, .page_size = 0,
                .associativity = SHRT_MAX }
        }
        ,
        .cache = { // level 1 caches begin
        [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU,
                .size = 32768, .line_size = 128, .num_lines = 64,
                .associativity = 4 }
        ,
        [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU,
                .size = 32768, .line_size = 128, .num_lines = 32,
                .associativity = 8 }
        }
       }
      ,
      [1] = { // level 2 begins
        .tlb = {
        [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                .page_size = -1, .associativity = -1 }
        ,
        [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                .page_size = -1, .associativity = -1 }
        }
        ,
        .cache = {
        [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                .size = 524288, .line_size = 128, .num_lines = 256,
                .associativity = 8 }
        ,
        [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                .num_lines = -1, .associativity = -1 }
        }
       }
      ,
      [2] = { // level 3 begins
        .tlb = {
        [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                .page_size = -1, .associativity = -1 }
        ,
        [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                .page_size = -1, .associativity = -1 }
        }
        ,
        .cache = {
        [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                .size = 4194304, .line_size = 128, .num_lines = 4096,
                .associativity = 8 }
        ,
        [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                .num_lines = -1, .associativity = -1 }
        }
       }
      ,
      }
     },                         // POWER7 end
        {3,
         {
          [0] = { // level 1 begins
            .tlb = {
            /// POWER8 has an ERAT (Effective to Real Address
            /// Translation) instead of a TLB.  For the purposes of this
            /// data, we will treat it like a TLB.
            [0] = { .type = PAPI_MH_TYPE_INST,
                    .num_entries = 72, .page_size = 0,
                    .associativity = SHRT_MAX }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA,
                    .num_entries = 48, .page_size = 0,
                    .associativity = SHRT_MAX }
            }
            ,
            .cache = { // level 1 caches begin
            [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 32768, .line_size = 128, .num_lines = 64,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU,
                    .size = 65536, .line_size = 128, .num_lines = 512,
                    .associativity = 8 }
            }
           }
          ,
          [1] = { // level 2 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED, .num_entries = 2048,
                    .page_size = 0, .associativity = 4 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 262144, .line_size = 128, .num_lines = 256,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          [2] = { // level 3 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 8388608, .line_size = 128, .num_lines = 65536,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          }
         },                         // POWER8 end
        {3,
         {
          [0] = { // level 1 begins
            .tlb = {
            /// POWER9 has an ERAT (Effective to Real Address
            /// Translation) instead of a TLB.  For the purposes of this
            /// data, we will treat it like a TLB.
            [0] = { .type = PAPI_MH_TYPE_INST,
                    .num_entries = 64, .page_size = 0,
                    .associativity = SHRT_MAX }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA,
                    .num_entries = 64, .page_size = 0,
                    .associativity = SHRT_MAX }
            }
            ,
            .cache = { // level 1 caches begin
            [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 32768, .line_size = 128, .num_lines = 256,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU,
                    .size = 32768, .line_size = 128, .num_lines = 256,
                    .associativity = 8 }
            }
           }
          ,
          [1] = { // level 2 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED, .num_entries = 1024,
                    .page_size = 0, .associativity = 4 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 524288, .line_size = 128, .num_lines = 4096,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          [2] = { // level 3 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 10485760, .line_size = 128, .num_lines = 81920,
                    .associativity = 20 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          }
         },
        {3,
         {
          [0] = { // level 1 begins
            .tlb = {
            /// POWER10 has an ERAT (Effective to Real Address
            /// Translation) instead of a TLB.  For the purposes of this
            /// data, we will treat it like a TLB.
            [0] = { .type = PAPI_MH_TYPE_INST,
                    .num_entries = 64, .page_size = 0,
                    .associativity = SHRT_MAX }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA,
                    .num_entries = 64, .page_size = 0,
                    .associativity = SHRT_MAX }
            }
            ,
            .cache = { // level 1 caches begin
            [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 49152, .line_size = 128, .num_lines = 384,
                    .associativity = 6 }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU,
                    .size = 32768, .line_size = 128, .num_lines = 256,
                    .associativity = 8 }
            }
           }
          ,
          [1] = { // level 2 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED, .num_entries = 4096,
                    .page_size = 0, .associativity = 4 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 1048576, .line_size = 128, .num_lines = 8192,
                    .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          [2] = { // level 3 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                    .page_size = -1, .associativity = -1 }
            }
            ,
            .cache = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_PSEUDO_LRU,
                    .size = 4194304, .line_size = 128, .num_lines = 32768,
                    .associativity = 16 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                    .num_lines = -1, .associativity = -1 }
            }
           }
          ,
          }
         }                         // POWER10 end
};

#define SPRN_PVR 0x11F         /* Processor Version Register */
#define PVR_PROCESSOR_SHIFT 16
static unsigned int
mfpvr( void )
{
    unsigned long pvr;

    asm( "mfspr          %0,%1": "=r"( pvr ):"i"( SPRN_PVR ) );
    return pvr;

}

int
ppc64_get_memory_info( PAPI_hw_info_t * hw_info )
{
    unsigned int pvr = mfpvr(  ) >> PVR_PROCESSOR_SHIFT;

    int index;
    switch ( pvr ) {
    case 0x39:                 /* PPC970 */
    case 0x3C:                 /* PPC970FX */
    case 0x44:                 /* PPC970MP */
    case 0x45:                 /* PPC970GX */
        index = 0;
        break;
    case 0x3A:                 /* POWER5 */
    case 0x3B:                 /* POWER5+ */
        index = 1;
        break;
    case 0x3E:                 /* POWER6 */
        index = 2;
        break;
    case 0x3F:                 /* POWER7 */
        index = 3;
        break;
    case 0x4b:                 /* POWER8 */
        index = 4;
        break;
    case 0x4e:                 /* POWER9 */
        index = 5;
        break;
    case 0x80:                 /* POWER10 */
        index = 6;
        break;
    default:
        index = -1;
        hw_info->mem_hierarchy.levels = 0;
        break;
    }

    if ( index != -1 ) {
        int cache_level;
        PAPI_mh_info_t sys_mh_inf = sys_mem_info[index];
        PAPI_mh_info_t *mh_inf = &hw_info->mem_hierarchy;
        mh_inf->levels = sys_mh_inf.levels;
        PAPI_mh_level_t *level = mh_inf->level;
        PAPI_mh_level_t sys_mh_level;
        for ( cache_level = 0; cache_level < sys_mh_inf.levels; cache_level++ ) {
            sys_mh_level = sys_mh_inf.level[cache_level];
            int cache_idx;
            for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
                // process TLB info
                PAPI_mh_tlb_info_t curr_tlb = sys_mh_level.tlb[cache_idx];
                int type = curr_tlb.type;
                if ( type != PAPI_MH_TYPE_EMPTY ) {
                    level[cache_level].tlb[cache_idx].type = type;
                    level[cache_level].tlb[cache_idx].associativity =
                        curr_tlb.associativity;
                    level[cache_level].tlb[cache_idx].num_entries =
                        curr_tlb.num_entries;
                }
            }
            for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
                // process cache info
                PAPI_mh_cache_info_t curr_cache = sys_mh_level.cache[cache_idx];
                int type = curr_cache.type;
                if ( type != PAPI_MH_TYPE_EMPTY ) {
                    level[cache_level].cache[cache_idx].type = type;
                    level[cache_level].cache[cache_idx].associativity =
                        curr_cache.associativity;
                    level[cache_level].cache[cache_idx].size = curr_cache.size;
                    level[cache_level].cache[cache_idx].line_size =
                        curr_cache.line_size;
                    level[cache_level].cache[cache_idx].num_lines =
                        curr_cache.num_lines;
                }
            }
        }
    }
    return 0;
}
#endif



#if defined(__sparc__)
static int
sparc_sysfs_cpu_attr( char *name, char **result )
{
    const char *path_base = "/sys/devices/system/cpu/";
    char path_buf[PATH_MAX];
    char val_buf[32];
    DIR *sys_cpu;

    sys_cpu = opendir( path_base );
    if ( sys_cpu ) {
        struct dirent *cpu;

        while ( ( cpu = readdir( sys_cpu ) ) != NULL ) {
            int fd;

            if ( strncmp( "cpu", cpu->d_name, 3 ) )
                continue;
            strcpy( path_buf, path_base );
            strcat( path_buf, cpu->d_name );
            strcat( path_buf, "/" );
            strcat( path_buf, name );

            fd = open( path_buf, O_RDONLY );
            if ( fd < 0 )
                continue;

            if ( read( fd, val_buf, 32 ) < 0 )
                continue;
            close( fd );

            *result = strdup( val_buf );
            return 0;
        }
    }
    closedir( sys_cpu );
    return -1;
}

static int
sparc_cpu_attr( char *name, unsigned long long *val )
{
    char *buf;
    int r;

    r = sparc_sysfs_cpu_attr( name, &buf );
    if ( r == -1 )
        return -1;

    sscanf( buf, "%llu", val );

    free( buf );

    return 0;
}

static char *
search_cpu_info( FILE * f, char *search_str, char *line )
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrend\
     orf! */
  /* See the home page for the German version of PAPI. */

  char *s;

  while ( fgets( line, 256, f ) != NULL ) {
    if ( strstr( line, search_str ) != NULL ) {
      /* ignore all characters in line up to : */
      for ( s = line; *s && ( *s != ':' ); ++s );
      if ( *s )
        return s;
    }
  }
  return NULL;

  /* End stolen code */
}


static int
sparc_get_memory_info( PAPI_hw_info_t * hw_info )
{
    unsigned long long cache_size, cache_line_size;
    /* unsigned long long cycles_per_second; */
    char maxargs[PAPI_HUGE_STR_LEN];
    /* PAPI_mh_tlb_info_t *tlb; */
    PAPI_mh_level_t *level;
    char *s, *t;
    FILE *f;

    /* First, fix up the cpu vendor/model/etc. values */
    strcpy( hw_info->vendor_string, "Sun" );
    hw_info->vendor = PAPI_VENDOR_SUN;

    f = fopen( "/proc/cpuinfo", "r" );
    if ( !f )
        return PAPI_ESYS;

    rewind( f );
    s = search_cpu_info( f, "cpu", maxargs );
    if ( !s ) {
        fclose( f );
        return PAPI_ESYS;
    }

    t = strchr( s + 2, '\n' );
    if ( !t ) {
        fclose( f );
        return PAPI_ESYS;
    }

    *t = '\0';
    strcpy( hw_info->model_string, s + 2 );

    fclose( f );

    /*
    if ( sparc_sysfs_cpu_attr( "clock_tick", &s ) == -1 )
        return PAPI_ESYS;

    sscanf( s, "%llu", &cycles_per_second );
    free( s );

    hw_info->mhz = cycles_per_second / 1000000;
    hw_info->clock_mhz = hw_info->mhz;
    */

    /* Now fetch the cache info */
    hw_info->mem_hierarchy.levels = 3;

    level = &hw_info->mem_hierarchy.level[0];

    sparc_cpu_attr( "l1_icache_size", &cache_size );
    sparc_cpu_attr( "l1_icache_line_size", &cache_line_size );
    level[0].cache[0].type = PAPI_MH_TYPE_INST;
    level[0].cache[0].size = cache_size;
    level[0].cache[0].line_size = cache_line_size;
    level[0].cache[0].num_lines = cache_size / cache_line_size;
    level[0].cache[0].associativity = 1;

    sparc_cpu_attr( "l1_dcache_size", &cache_size );
    sparc_cpu_attr( "l1_dcache_line_size", &cache_line_size );
    level[0].cache[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT;
    level[0].cache[1].size = cache_size;
    level[0].cache[1].line_size = cache_line_size;
    level[0].cache[1].num_lines = cache_size / cache_line_size;
    level[0].cache[1].associativity = 1;

    sparc_cpu_attr( "l2_cache_size", &cache_size );
    sparc_cpu_attr( "l2_cache_line_size", &cache_line_size );
    level[1].cache[0].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WB;
    level[1].cache[0].size = cache_size;
    level[1].cache[0].line_size = cache_line_size;
    level[1].cache[0].num_lines = cache_size / cache_line_size;
    level[1].cache[0].associativity = 1;

#if 0
    tlb = &hw_info->mem_hierarchy.level[0].tlb[0];
    switch ( _perfmon2_pfm_pmu_type ) {
    case PFMLIB_SPARC_ULTRA12_PMU:
        tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[0].num_entries = 64;
        tlb[0].associativity = SHRT_MAX;
        tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[1].num_entries = 64;
        tlb[1].associativity = SHRT_MAX;
        break;

    case PFMLIB_SPARC_ULTRA3_PMU:
    case PFMLIB_SPARC_ULTRA3I_PMU:
    case PFMLIB_SPARC_ULTRA3PLUS_PMU:
    case PFMLIB_SPARC_ULTRA4PLUS_PMU:
        level[0].cache[0].associativity = 4;
        level[0].cache[1].associativity = 4;
        level[1].cache[0].associativity = 4;

        tlb[0].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[0].num_entries = 16;
        tlb[0].associativity = SHRT_MAX;
        tlb[1].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[1].num_entries = 16;
        tlb[1].associativity = SHRT_MAX;
        tlb[2].type = PAPI_MH_TYPE_DATA;
        tlb[2].num_entries = 1024;
        tlb[2].associativity = 2;
        tlb[3].type = PAPI_MH_TYPE_INST;
        tlb[3].num_entries = 128;
        tlb[3].associativity = 2;
        break;

    case PFMLIB_SPARC_NIAGARA1:
        level[0].cache[0].associativity = 4;
        level[0].cache[1].associativity = 4;
        level[1].cache[0].associativity = 12;

        tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[0].num_entries = 64;
        tlb[0].associativity = SHRT_MAX;
        tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[1].num_entries = 64;
        tlb[1].associativity = SHRT_MAX;
        break;

    case PFMLIB_SPARC_NIAGARA2:
        level[0].cache[0].associativity = 8;
        level[0].cache[1].associativity = 4;
        level[1].cache[0].associativity = 16;

        tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[0].num_entries = 64;
        tlb[0].associativity = SHRT_MAX;
        tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
        tlb[1].num_entries = 128;
        tlb[1].associativity = SHRT_MAX;
        break;
    }
#endif
    return 0;
}
#endif


#if defined(__aarch64__)

PAPI_mh_info_t sys_mem_info[] = {
    {2,                         // Fujitsu A64FX begin
     {
      [0] = { // level 1 begins
        .tlb = {
        [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_FIFO,
                .num_entries = 16, .page_size = 65536,
                .associativity = SHRT_MAX }
        ,
        [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_FIFO,
                .num_entries = 16, .page_size = 65536,
                .associativity = SHRT_MAX }
        }
        ,
        .cache = { // level 1 caches begin
        [0] = { .type = PAPI_MH_TYPE_INST,
                .size = 65536, .line_size = 256, .num_lines = 64,
                .associativity = 4 }
        ,
        [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WB,
                .size = 65536, .line_size = 256, .num_lines = 64,
                .associativity = 4 }
        }
       }
      ,
      [1] = { // level 2 begins
        .tlb = {
        [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_LRU, .num_entries = 1024,
                .page_size = 65536, .associativity = 4 }
        ,
        [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_LRU, .num_entries = 1024,
                .page_size = 65536, .associativity = 4 }
        }
        ,
        .cache = {
        [0] = { .type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WB,
                .size = 8388608, .line_size = 256, .num_lines = 2048,
                .associativity = 16 }
        ,
        [1] = { .type = PAPI_MH_TYPE_EMPTY, .size = -1, .line_size = -1,
                .num_lines = -1, .associativity = -1 }
        }
       }
      ,
      }
     },                        // Fujitsu A64FX end
    {3,                        // ARM Neoverse V2 begin
     {
      [0] = { // level 1 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_FIFO,
                            .num_entries = 48, .page_size = -1,
                            .associativity = SHRT_MAX }
            ,
            [1] = { .type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_FIFO,
                            .num_entries = 48, .page_size = -1,
                            .associativity = SHRT_MAX }
            }
       }
      ,
      [1] = { // level 2 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_UNIFIED, .num_entries = 2048,
                        .page_size = -1, .associativity = 8 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                            .page_size = -1, .associativity = -1 }
            }
       }
      ,
      [2] = { // level 3 begins
            .tlb = {
            [0] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                            .page_size = -1, .associativity = -1 }
            ,
            [1] = { .type = PAPI_MH_TYPE_EMPTY, .num_entries = -1,
                            .page_size = -1, .associativity = -1 }
            }
       }
      ,
      }
     }                         // ARM Neoverse V2 end
};

#define IMPLEMENTER_FUJITSU 0x46
#define PARTNUM_FUJITSU_A64FX 0x001

#define IMPLEMENTER_ARM 0x41
#define PARTNUM_ARM_NEOVERSE_V2 0xd4f

int
aarch64_get_memory_info( PAPI_hw_info_t * hw_info )
{
    unsigned int implementer, partnum;

    implementer = hw_info->vendor;
    partnum = hw_info->cpuid_model;

    int index = -1;
    switch ( implementer ) {
    case IMPLEMENTER_FUJITSU:
        switch ( partnum ) {
        case PARTNUM_FUJITSU_A64FX: /* Fujitsu A64FX */
            index = 0;
            break;
        default:
            generic_get_memory_info (hw_info);
            return 0;
        }
        break;
    case IMPLEMENTER_ARM:
        switch ( partnum ) {
        case PARTNUM_ARM_NEOVERSE_V2: /* ARM Neoverse V2 */
            index = 1;
            generic_get_memory_info (hw_info);
            break;
        default:
            generic_get_memory_info (hw_info);
            return 0;
        }
        break;
    default:
        generic_get_memory_info (hw_info);
        return 0;
    }

    if ( index != -1 ) {
        int cache_level;
        PAPI_mh_info_t sys_mh_inf = sys_mem_info[index];
        PAPI_mh_info_t *mh_inf = &hw_info->mem_hierarchy;
        mh_inf->levels = sys_mh_inf.levels;
        PAPI_mh_level_t *level = mh_inf->level;
        PAPI_mh_level_t sys_mh_level;
        for ( cache_level = 0; cache_level < sys_mh_inf.levels; cache_level++ ) {
            sys_mh_level = sys_mh_inf.level[cache_level];
            int cache_idx;
            for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
                // process TLB info
                PAPI_mh_tlb_info_t curr_tlb = sys_mh_level.tlb[cache_idx];
                int type = curr_tlb.type;
                if ( type != PAPI_MH_TYPE_EMPTY ) {
                    level[cache_level].tlb[cache_idx].type = type;
                    level[cache_level].tlb[cache_idx].associativity =
                        curr_tlb.associativity;
                    level[cache_level].tlb[cache_idx].num_entries =
                        curr_tlb.num_entries;
                }
            }
            for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
                // process cache info
                PAPI_mh_cache_info_t curr_cache = sys_mh_level.cache[cache_idx];
                int type = curr_cache.type;
                if ( type != PAPI_MH_TYPE_EMPTY ) {
                    level[cache_level].cache[cache_idx].type = type;
                    level[cache_level].cache[cache_idx].associativity =
                        curr_cache.associativity;
                    level[cache_level].cache[cache_idx].size = curr_cache.size;
                    level[cache_level].cache[cache_idx].line_size =
                        curr_cache.line_size;
                    level[cache_level].cache[cache_idx].num_lines =
                        curr_cache.num_lines;
                }
            }
        }
    }
    return 0;
}
#endif


/* Fallback Linux code to read the cache info from /sys */
int
generic_get_memory_info( PAPI_hw_info_t *hw_info )
{

    int type=0,level,result;
    int size,line_size,associativity,sets;
    int write_policy,allocation_policy;
    DIR *dir;
    FILE *fff;
    char filename[BUFSIZ],type_string[BUFSIZ];
    char *str_result;
    char write_policy_string[BUFSIZ],allocation_policy_string[BUFSIZ];
    struct dirent *d;
    int max_level=0;
    int level_count,level_index;

    PAPI_mh_level_t *L = hw_info->mem_hierarchy.level;

    /* open Linux cache dir            */
    /* assume all CPUs same as cpu0.    */
    /* Not necessarily a good assumption    */

    dir=opendir("/sys/devices/system/cpu/cpu0/cache");
    if (dir==NULL) {
        goto unrecoverable_error;
    }

    for (level_index=0; level_index < PAPI_MAX_MEM_HIERARCHY_LEVELS; ++level_index) {
        for (level_count = 0; level_count < PAPI_MH_MAX_LEVELS; ++level_count) {
            L[level_index].cache[level_count].type = PAPI_MH_TYPE_EMPTY;
        }
    }

    while(1) {
        d = readdir(dir);
        if (d==NULL) break;

        if (strncmp(d->d_name, "index", 5)) continue;

        MEMDBG("Found %s\n",d->d_name);

        /*************/
        /* Get level */
        /*************/
        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/level",
            d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open level.\n");
            goto unrecoverable_error;
        }

        result=fscanf(fff,"%d",&level);
        fclose(fff);
        if (result!=1) {
            MEMDBG("Could not read cache level\n");
            goto unrecoverable_error;
        }

        /* Index arrays from 0 */
        level_index=level-1;

        level_count = 0;
        while (L[level_index].cache[level_count].type != PAPI_MH_TYPE_EMPTY) {
            level_count++;
            if (level_count>=PAPI_MH_MAX_LEVELS) {
                break;
            }
        }

        if (level_count>=PAPI_MH_MAX_LEVELS) {
            MEMDBG("Exceeded maximum levels %d\n",
                PAPI_MH_MAX_LEVELS);
            break;
        }

        /************/
        /* Get type */
        /************/
        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/type",d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open type\n");
            goto unrecoverable_error;
        }
        str_result=fgets(type_string, BUFSIZ, fff);
        fclose(fff);
        if (str_result==NULL) {
            MEMDBG("Could not read cache type\n");
            goto unrecoverable_error;
        }
        char *cache_type = "Data";
        if (!strncmp(type_string,cache_type,strlen(cache_type))) {
            type=PAPI_MH_TYPE_DATA;
        }
        cache_type = "Instruction";
        if (!strncmp(type_string,cache_type,strlen(cache_type))) {
            type=PAPI_MH_TYPE_INST;
        }
        cache_type = "Unified";
        if (!strncmp(type_string,cache_type,strlen(cache_type))) {
            type=PAPI_MH_TYPE_UNIFIED;
        }

        /********************/
        /* Get write_policy */
        /********************/
        write_policy=0;
        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/write_policy",d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open write_policy\n");
            goto get_allocation_policy;
        }
        str_result=fgets(write_policy_string, BUFSIZ, fff);
        fclose(fff);
        if (str_result==NULL) {
            MEMDBG("Could not read cache write_policy\n");
            goto get_allocation_policy;
        }
        if (!strcmp(write_policy_string,"WriteThrough")) {
            write_policy=PAPI_MH_TYPE_WT;
        }
        if (!strcmp(write_policy_string,"WriteBack")) {
            write_policy=PAPI_MH_TYPE_WB;
        }

get_allocation_policy:

        /*************************/
        /* Get allocation_policy */
        /*************************/
        allocation_policy=0;
        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/allocation_policy",d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open allocation_policy\n");
            goto get_size;
        }
        str_result=fgets(allocation_policy_string, BUFSIZ, fff);
        fclose(fff);
        if (str_result==NULL) {
            MEMDBG("Could not read cache allocation_policy\n");
            goto get_size;
        }
        if (!strcmp(allocation_policy_string,"ReadAllocate")) {
            allocation_policy=PAPI_MH_TYPE_RD_ALLOC;
        }
        if (!strcmp(allocation_policy_string,"WriteAllocate")) {
            allocation_policy=PAPI_MH_TYPE_WR_ALLOC;
        }
        if (!strcmp(allocation_policy_string,"ReadWriteAllocate")) {
            allocation_policy=PAPI_MH_TYPE_RW_ALLOC;
        }

get_size:
        L[level_index].cache[level_count].type=type | write_policy | allocation_policy;

        /*************/
        /* Get Size  */
        /*************/
        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/size",d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open size\n");
            goto unrecoverable_error;
        }
        result=fscanf(fff,"%d",&size);
        fclose(fff);
        if (result!=1) {
            MEMDBG("Could not read cache size\n");
            goto unrecoverable_error;
        }

        /* Linux reports in kB, PAPI expects in Bytes */
        L[level_index].cache[level_count].size=size*1024;

        /*************/
        /* Line Size */
        /*************/
        L[level_index].cache[level_count].line_size=0;

        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/coherency_line_size",
            d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open linesize\n");
        }
        else {
            result=fscanf(fff,"%d",&line_size);
            fclose(fff);
            if (result!=1) {
                MEMDBG("Could not read cache line-size\n");
            } else {
                L[level_index].cache[level_count].line_size=line_size;
            }
        }


        /*********************/
        /* Get Associativity */
        /*********************/
        L[level_index].cache[level_count].associativity=0;

        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/ways_of_associativity",
            d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open associativity\n");
        }
        else {
            result=fscanf(fff,"%d",&associativity);
            fclose(fff);
            if (result!=1) {
                MEMDBG("Could not read cache associativity\n");
            }
            else {
                L[level_index].cache[level_count].associativity=associativity;
            }
        }

        /************/
        /* Get Sets */
        /************/
        L[level_index].cache[level_count].num_lines=0;

        sprintf(filename,
            "/sys/devices/system/cpu/cpu0/cache/%s/number_of_sets",
            d->d_name);
        fff=fopen(filename,"r");
        if (fff==NULL) {
            MEMDBG("Cannot open sets\n");
        }
        else {
            result=fscanf(fff,"%d",&sets);
            fclose(fff);

            if (result!=1) {
                MEMDBG("Could not read cache sets\n");
            }
            else {
                L[level_index].cache[level_count].num_lines=sets;
            }
        }

        /* Sanity check results */
        if ((line_size!=0) && (associativity!=0)) {
            if (((size*1024)/line_size/associativity)!=sets) {
                 MEMDBG("Warning!  sets %d != expected %d\n",
                    sets,((size*1024)/line_size/associativity));
            }
        }

        MEMDBG("\tL%d %s cache\n",level,type_string);
        MEMDBG("\t%d kilobytes\n",size);
        MEMDBG("\t%d byte linesize\n",line_size);
        MEMDBG("\t%d-way associative\n",associativity);
        MEMDBG("\t%d lines\n",sets);
        MEMDBG("\tUnknown inclusivity\n");
        MEMDBG("\tUnknown replacement algorithm\n");
        MEMDBG("\tUnknown if victim cache\n");

        if (level>max_level) max_level=level;

        if (level>PAPI_MAX_MEM_HIERARCHY_LEVELS) {
            MEMDBG("Exceeded maximum cache level %d\n",
                PAPI_MAX_MEM_HIERARCHY_LEVELS);
            break;
        }
    }

    closedir(dir);
    hw_info->mem_hierarchy.levels = max_level;

    return 0;

unrecoverable_error:

    /* Just say we have no cache */
    hw_info->mem_hierarchy.levels = 0;

    closedir(dir);
    return 0;
}


int
_linux_get_memory_info( PAPI_hw_info_t * hwinfo, int cpu_type )
{
    ( void ) cpu_type;         /*unused */
    int retval = PAPI_OK;

#if defined(__i386__)||defined(__x86_64__)
    x86_get_memory_info( hwinfo );
#elif defined(__ia64__)
    ia64_get_memory_info( hwinfo );
#elif defined(__powerpc__)
    ppc64_get_memory_info( hwinfo );
#elif defined(__sparc__)
    sparc_get_memory_info( hwinfo );
#elif defined(__arm__)
    #warning "WARNING! linux_get_memory_info() does nothing on ARM32!"
    generic_get_memory_info (hwinfo);
#elif defined(__aarch64__)
    aarch64_get_memory_info( hwinfo );
#else
    generic_get_memory_info (hwinfo);
#endif

    return retval;
}

int
_linux_update_shlib_info( papi_mdi_t *mdi )
{

    char fname[PAPI_HUGE_STR_LEN];
    unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
    char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[16];
    char mapname[PAPI_HUGE_STR_LEN], lastmapname[PAPI_HUGE_STR_LEN];
    unsigned long begin = 0, end = 0, size = 0, inode = 0, foo = 0;
    PAPI_address_map_t *tmp = NULL;
    FILE *f;

    memset( fname, 0x0, sizeof ( fname ) );
    memset( buf, 0x0, sizeof ( buf ) );
    memset( perm, 0x0, sizeof ( perm ) );
    memset( dev, 0x0, sizeof ( dev ) );
    memset( mapname, 0x0, sizeof ( mapname ) );
    memset( lastmapname, 0x0, sizeof ( lastmapname ) );

    sprintf( fname, "/proc/%ld/maps", ( long ) mdi->pid );
    f = fopen( fname, "r" );

    if ( !f ) {
        PAPIERROR( "fopen(%s) returned < 0", fname );
        return PAPI_OK;
    }

  again:
    while ( !feof( f ) ) {
        begin = end = size = inode = foo = 0;
        if ( fgets( buf, sizeof ( buf ), f ) == 0 )
            break;
        /* If mapname is null in the string to be scanned, we need to detect that */
        if ( strlen( mapname ) )
            strcpy( lastmapname, mapname );
        else
            lastmapname[0] = '\0';
        /* If mapname is null in the string to be scanned, we need to detect that */
        mapname[0] = '\0';
        sscanf( buf, "%lx-%lx %4s %lx %s %ld %s", &begin, &end, perm, &foo, dev,
                &inode, mapname );
        size = end - begin;

        /* the permission string looks like "rwxp", where each character can
         * be either the letter, or a hyphen.  The final character is either
         * p for private or s for shared. */

        if ( counting ) {
            if ( ( perm[2] == 'x' ) && ( perm[0] == 'r' ) && ( inode != 0 ) ) {
                if ( strcmp( mdi->exe_info.fullname, mapname )
                     == 0 ) {
                    mdi->exe_info.address_info.text_start =
                        ( vptr_t ) begin;
                    mdi->exe_info.address_info.text_end =
                        ( vptr_t ) ( begin + size );
                }
                t_index++;
            } else if ( ( perm[0] == 'r' ) && ( perm[1] == 'w' ) &&
                        ( inode != 0 )
                        &&
                        ( strcmp
                          ( mdi->exe_info.fullname,
                            mapname ) == 0 ) ) {
                mdi->exe_info.address_info.data_start =
                    ( vptr_t ) begin;
                mdi->exe_info.address_info.data_end =
                    ( vptr_t ) ( begin + size );
                d_index++;
            } else if ( ( perm[0] == 'r' ) && ( perm[1] == 'w' ) &&
                        ( inode == 0 )
                        &&
                        ( strcmp
                          ( mdi->exe_info.fullname,
                            lastmapname ) == 0 ) ) {
                mdi->exe_info.address_info.bss_start =
                    ( vptr_t ) begin;
                mdi->exe_info.address_info.bss_end =
                    ( vptr_t ) ( begin + size );
                b_index++;
            }
        } else if ( !counting ) {
            if ( ( perm[2] == 'x' ) && ( perm[0] == 'r' ) && ( inode != 0 ) ) {
                if ( strcmp( mdi->exe_info.fullname, mapname )
                     != 0 ) {
                    t_index++;
                    tmp[t_index - 1].text_start = ( vptr_t ) begin;
                    tmp[t_index - 1].text_end = ( vptr_t ) ( begin + size );
                    strncpy( tmp[t_index - 1].name, mapname, PAPI_HUGE_STR_LEN );
               tmp[t_index - 1].name[PAPI_HUGE_STR_LEN-1]=0;
                }
            } else if ( ( perm[0] == 'r' ) && ( perm[1] == 'w' ) &&
                        ( inode != 0 ) ) {
                if ( ( strcmp
                       ( mdi->exe_info.fullname,
                         mapname ) != 0 )
                     && ( t_index > 0 ) &&
                     ( tmp[t_index - 1].data_start == 0 ) ) {
                    tmp[t_index - 1].data_start = ( vptr_t ) begin;
                    tmp[t_index - 1].data_end = ( vptr_t ) ( begin + size );
                }
            } else if ( ( perm[0] == 'r' ) && ( perm[1] == 'w' ) &&
                        ( inode == 0 ) ) {
                if ( ( t_index > 0 ) && ( tmp[t_index - 1].bss_start == 0 ) ) {
                    tmp[t_index - 1].bss_start = ( vptr_t ) begin;
                    tmp[t_index - 1].bss_end = ( vptr_t ) ( begin + size );
                }
            }
        }
    }

    if ( counting ) {
        /* When we get here, we have counted the number of entries in the map
           for us to allocate */

        tmp =
            ( PAPI_address_map_t * ) papi_calloc( t_index,
                                                  sizeof
                                                  ( PAPI_address_map_t ) );
        if ( tmp == NULL ) {
            PAPIERROR( "Error allocating shared library address map" );
            fclose(f);
            return PAPI_ENOMEM;
        }
        t_index = 0;
        rewind( f );
        counting = 0;
        goto again;
    } else {
        if ( mdi->shlib_info.map )
            papi_free( mdi->shlib_info.map );
        mdi->shlib_info.map = tmp;
        mdi->shlib_info.count = t_index;

        fclose( f );
    }

    return PAPI_OK;
}
