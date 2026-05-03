/** file papi_hardware_avail.c
  * @page papi_hardware_avail
  * @brief papi_hardware_avail utility.
  * @section  NAME
  *     papi_hardware_avail - provides detailed information on the hardware available in the system.
  *
  * @section Synopsis
  *
  * @section Description
  *     papi_hardware_avail is a PAPI utility program that reports information
  *     about the hardware devices equipped in the system.
  *
  * @section Options
  *      <ul>
  *     <li>-h help message
  *      </ul>
  *
  * @section Bugs
  *     There are no known bugs in this utility.
  *     If you find a bug, it should be reported to the
  *     PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "print_header.h"

typedef struct command_flags
{
    int help;
} command_flags_t;

static void
print_help( char **argv )
{
    printf( "This is the PAPI hardware avail program.\n" );
    printf( "It provides availability of system's equipped hardware devices.\n" );
    printf( "Usage: %s [options]\n", argv[0] );
    printf( "Options:\n\n" );
    printf( "  --help, -h    print this help message\n" );
}

static void
parse_args( int argc, char **argv, command_flags_t * f )
{
    int i;

    /* Look for all currently defined commands */
    memset( f, 0, sizeof ( command_flags_t ) );
    for ( i = 1; i < argc; i++ ) {
        if ( !strcmp( argv[i], "-h" ) || !strcmp( argv[i], "--help" ) )
            f->help = 1;
        else
            printf( "%s is not supported\n", argv[i] );
    }

    /* if help requested, print and bail */
    if ( f->help ) {
        print_help( argv );
        exit( 1 );
    }

}

int
main( int argc, char **argv )
{
    int i;
    int retval;
    const PAPI_component_info_t *cmpinfo = NULL;
    command_flags_t flags;
    int numcmp;
    int sysdetect_avail = 0;

    /* Initialize before parsing the input arguments */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"Error!  PAPI_library_init\n");
        return retval;
    }

    parse_args( argc, argv, &flags );

    retval = PAPI_set_debug( PAPI_VERB_ECONT );
    if ( retval != PAPI_OK ) {
        fprintf(stderr,"Error!  PAPI_set_debug\n");
        return retval;
    }

    numcmp = PAPI_num_components( );
    for (i = 0; i < numcmp; i++) {
        cmpinfo = PAPI_get_component_info( i );
        if (strcmp("sysdetect", cmpinfo->name) == 0)
            sysdetect_avail = 1;
    }

    if (sysdetect_avail == 0) {
        fprintf(stderr, "Error! Sysdetect component not enabled\n");
        return 0;
    }

    printf( "\nDevice Summary -----------------------------------------------------------------\n" );
    void *handle;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__ALL;
    int id, vendor_id, dev_count;
    const char *vendor_name, *status;

    printf( "Vendor           DevCount \n" );
    while (PAPI_enum_dev_type(enum_modifier, &handle) == PAPI_OK) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__CHAR_NAME, &vendor_name);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, &dev_count);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__CHAR_STATUS, &status);

        printf( "%-18s (%d)\n", vendor_name, dev_count);
        printf( " \\-> Status: %s\n", status );
        printf( "\n" );
    }

    printf( "\nDevice Information -------------------------------------------------------------\n" );

    while (PAPI_enum_dev_type(enum_modifier, &handle) == PAPI_OK) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_PAPI_ID, &id);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID, &vendor_id);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__CHAR_NAME, &vendor_name);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, &dev_count);

        if ( id == PAPI_DEV_TYPE_ID__CPU && dev_count > 0 ) {
            unsigned int numas = 1;
            for ( i = 0; i < dev_count; ++i ) {
                const char *cpu_name;
                unsigned int family, model, stepping;
                unsigned int sockets, cores, threads;
                unsigned int l1i_size, l1d_size, l2u_size, l3u_size;
                unsigned int l1i_line_sz, l1d_line_sz, l2u_line_sz, l3u_line_sz;
                unsigned int l1i_line_cnt, l1d_line_cnt, l2u_line_cnt, l3u_line_cnt;
                unsigned int l1i_cache_ass, l1d_cache_ass, l2u_cache_ass, l3u_cache_ass;

                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_CHAR_NAME, &cpu_name);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_FAMILY, &family);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_MODEL, &model);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_STEPPING, &stepping);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT, &sockets);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT, &numas);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT, &cores);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT, &threads);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE, &l1i_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE, &l1d_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE, &l2u_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE, &l3u_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE, &l1i_line_sz);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE, &l1d_line_sz);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE, &l2u_line_sz);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE, &l3u_line_sz);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT, &l1i_line_cnt);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT, &l1d_line_cnt);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT, &l2u_line_cnt);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT, &l3u_line_cnt);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC, &l1i_cache_ass);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC, &l1d_cache_ass);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC, &l2u_cache_ass);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC, &l3u_cache_ass);

                printf( "Vendor                                : %s (%u,0x%x)\n",
                                                                 vendor_name,
                                                                 vendor_id,
                                                                 vendor_id );
                printf( "Id                                    : %u\n", i );
                printf( "Name                                  : %s\n", cpu_name );
                printf( "CPUID                                 : Family/Model/Stepping %u/%u/%u 0x%02x/0x%02x/0x%02x\n",
                                                                 family, model, stepping, family, model, stepping );
                printf( "Sockets                               : %u\n", sockets );
                printf( "Numa regions                          : %u\n", numas );
                printf( "Cores per socket                      : %u\n", cores );
                printf( "Cores per NUMA region                 : %u\n", threads / numas );
                printf( "SMT threads per core                  : %u\n", threads / sockets / cores );

                if (l1i_size > 0) {
                    printf( "L1i Cache                             : Size/LineSize/Lines/Assoc %uKB/%uB/%u/%u\n",
                            l1i_size >> 10, l1i_line_sz, l1i_line_cnt, l1i_cache_ass);
                    printf( "L1d Cache                             : Size/LineSize/Lines/Assoc %uKB/%uB/%u/%u\n",
                            l1d_size >> 10, l1d_line_sz, l1d_line_cnt, l1d_cache_ass);
                }

                if (l2u_size > 0) {
                    printf( "L2 Cache                              : Size/LineSize/Lines/Assoc %uKB/%uB/%u/%u\n",
                            l2u_size >> 10, l2u_line_sz, l2u_line_cnt, l2u_cache_ass );
                }

                if (l3u_size > 0) {
                    printf( "L3 Cache                              : Size/LineSize/Lines/Assoc %uKB/%uB/%u/%u\n",
                            l3u_size >> 10, l3u_line_sz, l3u_line_cnt, l3u_cache_ass );
                }

#define MAX_NUMA_NODES  (16)
#define MAX_CPU_THREADS (512)
                unsigned int j;
                unsigned int affinity[MAX_CPU_THREADS];
                unsigned int numa_threads_count[MAX_NUMA_NODES] = { 0 };
                unsigned int numa_threads[MAX_NUMA_NODES][MAX_CPU_THREADS];
                for (j = 0; j < threads; ++j) {
                    PAPI_get_dev_attr(handle, j, PAPI_DEV_ATTR__CPU_UINT_THR_NUMA_AFFINITY, &affinity[j]);
                    if( affinity[j] >= 0 )
                        numa_threads[affinity[j]][numa_threads_count[affinity[j]]++] = j;
                }

                for ( j = 0; j < numas; ++j ) {
                    unsigned int k, memsize;
                    PAPI_get_dev_attr(handle, j, PAPI_DEV_ATTR__CPU_UINT_NUMA_MEM_SIZE, &memsize);
                    printf( "Numa Node %u Memory                    : %uMB\n", j, memsize );
                    printf( "Numa Node %u Threads                   : ", j );
                    for (k = 0; k < numa_threads_count[j]; ++k) {
                        printf( "%u ", numa_threads[j][k] );
                    }
                    printf( "\n" );
                }
                printf( "\n" );
            }
        }

        if ( id == PAPI_DEV_TYPE_ID__CUDA && dev_count > 0 ) {
            printf( "Vendor                                : %s\n", vendor_name );

            for ( i = 0; i < dev_count; ++i ) {
                unsigned long uid;
                unsigned int warp_size, thread_per_block, block_per_sm;
                unsigned int shm_per_block, shm_per_sm;
                unsigned int blk_dim_x, blk_dim_y, blk_dim_z;
                unsigned int grd_dim_x, grd_dim_y, grd_dim_z;
                unsigned int sm_count, multi_kernel, map_host_mem, async_memcpy;
                unsigned int unif_addr, managed_mem;
                unsigned int cc_major, cc_minor;
                const char *dev_name;

                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_ULONG_UID, &uid);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME, &dev_name);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE, &warp_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_THR_PER_BLK, &thread_per_block);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_BLK_PER_SM, &block_per_sm);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_BLK, &shm_per_block);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_SM, &shm_per_sm);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_X, &blk_dim_x);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Y, &blk_dim_y);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Z, &blk_dim_z);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_X, &grd_dim_x);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Y, &grd_dim_y);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Z, &grd_dim_z);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_SM_COUNT, &sm_count);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_MULTI_KERNEL, &multi_kernel);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_MAP_HOST_MEM, &map_host_mem);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_MEMCPY_OVERLAP, &async_memcpy);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_UNIFIED_ADDR, &unif_addr);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_MANAGED_MEM, &managed_mem);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR, &cc_major);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR, &cc_minor);

                printf( "Id                                    : %d\n", i );
                printf( "UID                                   : %lu\n", uid );
                printf( "Name                                  : %s\n", dev_name );
                printf( "Warp size                             : %u\n", warp_size );
                printf( "Max threads per block                 : %u\n", thread_per_block );
                printf( "Max blocks per multiprocessor         : %u\n", block_per_sm );
                printf( "Max shared memory per block           : %u\n", shm_per_block );
                printf( "Max shared memory per multiprocessor  : %u\n", shm_per_sm );
                printf( "Max block dim x                       : %u\n", blk_dim_x );
                printf( "Max block dim y                       : %u\n", blk_dim_y );
                printf( "Max block dim z                       : %u\n", blk_dim_z );
                printf( "Max grid dim x                        : %u\n", grd_dim_x );
                printf( "Max grid dim y                        : %u\n", grd_dim_y );
                printf( "Max grid dim z                        : %u\n", grd_dim_z );
                printf( "Multiprocessor count                  : %u\n", sm_count );
                printf( "Multiple kernels per context          : %s\n", multi_kernel ? "yes" : "no" );
                printf( "Can map host memory                   : %s\n", map_host_mem ? "yes" : "no");
                printf( "Can overlap compute and data transfer : %s\n", async_memcpy ? "yes" : "no" );
                printf( "Has unified addressing                : %s\n", unif_addr ? "yes" : "no" );
                printf( "Has managed memory                    : %s\n", managed_mem ? "yes" : "no" );
                printf( "Compute capability                    : %u.%u\n", cc_major, cc_minor );
                printf( "\n" );
            }
        }

        if ( id == PAPI_DEV_TYPE_ID__ROCM && dev_count > 0 ) {
            printf( "Vendor                                : %s\n", vendor_name );

            unsigned long uid;
            const char *dev_name;
            unsigned int wf_size, simd_per_cu, wg_size;
            unsigned int wf_per_cu, shm_per_wg, wg_dim_x, wg_dim_y, wg_dim_z;
            unsigned int grd_dim_x, grd_dim_y, grd_dim_z;
            unsigned int cu_count;
            unsigned int cc_major, cc_minor;

            for ( i = 0; i < dev_count; ++i ) {
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_ULONG_UID, &uid);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_CHAR_DEVICE_NAME, &dev_name);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WAVEFRONT_SIZE, &wf_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_SIMD_PER_CU, &simd_per_cu);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WORKGROUP_SIZE, &wg_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WAVE_PER_CU, &wf_per_cu);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_SHM_PER_WG, &shm_per_wg);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_X, &wg_dim_x);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Y, &wg_dim_y);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Z, &wg_dim_z);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_X, &grd_dim_x);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Y, &grd_dim_y);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Z, &grd_dim_z);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_CU_COUNT, &cu_count);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MAJOR, &cc_major);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MINOR, &cc_minor);

                printf( "Id                                    : %d\n", i );
                printf( "Name                                  : %s\n", dev_name );
                printf( "Wavefront size                        : %u\n", wf_size );
                printf( "SIMD per compute unit                 : %u\n", simd_per_cu );
                printf( "Max threads per workgroup             : %u\n", wg_size );
                printf( "Max waves per compute unit            : %u\n", wf_per_cu );
                printf( "Max shared memory per workgroup       : %u\n", shm_per_wg );
                printf( "Max workgroup dim x                   : %u\n", wg_dim_x );
                printf( "Max workgroup dim y                   : %u\n", wg_dim_y );
                printf( "Max workgroup dim z                   : %u\n", wg_dim_z );
                printf( "Max grid dim x                        : %u\n", grd_dim_x );
                printf( "Max grid dim y                        : %u\n", grd_dim_y );
                printf( "Max grid dim z                        : %u\n", grd_dim_z );
                printf( "Compute unit count                    : %u\n", cu_count );
                printf( "Compute capability                    : %u.%u\n", cc_major, cc_minor );
                printf( "\n" );
            }
        }
    }

    printf( "--------------------------------------------------------------------------------\n" );

    PAPI_shutdown();
    return 0;
}
