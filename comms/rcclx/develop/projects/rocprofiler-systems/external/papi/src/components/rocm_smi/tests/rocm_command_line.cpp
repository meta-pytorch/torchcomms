#define __HIP_PLATFORM_HCC__

/* file rocm_command_line.c
 * Nearly identical to "papi/src/utils/papi_command_line.c". 
 * This simply tries to add the events listed on the command line,
 * all into a single event set. It will then conduct a test using
 * the HIP interface to the AMD GPUs. It must be compiled with 
 * hipcc; see tests/ROCM_Makefile.
*/

/**
  *    @page papi_command_line
  * @brief executes PAPI preset or native events from the command line.
  *
  *    @section Synopsis
  *        papi_command_line < event > < event > ...
  *
  *    @section Description
  *        papi_command_line is a PAPI utility program that adds named events from the 
  *        command line to a PAPI EventSet and does some work with that EventSet. 
  *        This serves as a handy way to see if events can be counted together, 
  *        and if they give reasonable results for known work.
  *
  *    @section Options
  * <ul>
  *        <li>-u          Display output values as unsigned integers
  *        <li>-x          Display output values as hexadecimal
  *        <li>-h          Display help information about this utility.
  *    </ul>
  *
  *    @section Bugs
  *        There are no known bugs in this utility.
  *        If you find a bug, it should be reported to the
  *        PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include <hip/hip_runtime.h>

// Checks if HIP command (AMD) worked or not.
#define HIPCHECK(cmd)            \
{                                \
    hipError_t error  = cmd;     \
    if (error != hipSuccess) {   \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n",               \
            hipGetErrorString(error), error,__FILE__, __LINE__);    \
        exit(EXIT_FAILURE);      \
    }                            \
}

//-----------------------------------------------------------------------------
// HIP routine: Square each element in the array A and write to array C.
//-----------------------------------------------------------------------------
template <typename T>
__global__ void
vector_square(T *C_d, T *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

//-----------------------------------------------------------------------------
// conduct a test using HIP. Derived from AMD sample code 'square.cpp'.
// This just does GPU work; it does not deal with PAPI at all. That must be
// done externally.
//-----------------------------------------------------------------------------
void conductTest(int device) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    int thisDev, verbose=0;

    HIPCHECK(hipSetDevice(device));                      // Set device requested.
    HIPCHECK(hipGetDevice(&thisDev));                    // Double check.
    hipDeviceProp_t props;                         
    HIPCHECK(hipGetDeviceProperties(&props, thisDev));   // Get properties (for name).
    if (verbose) printf ("info: Requested Device=%i, running on device %i=%s\n", device, thisDev, props.name);

    if (verbose) printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);                        // standard malloc for host.
    HIPCHECK(A_h == NULL ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);                        // standard malloc for host.
    HIPCHECK(C_h == NULL ? hipErrorMemoryAllocation : hipSuccess );

    // Fill with Phi + i
    for (size_t i=0; i<N; i++) 
    {
        A_h[i] = 1.618f + i; 
    }

    if (verbose) printf ("info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    HIPCHECK(hipMalloc(&A_d, Nbytes));                   // HIP malloc for device.
    HIPCHECK(hipMalloc(&C_d, Nbytes));                   // ...


    if (verbose) printf ("info: copy Host2Device\n");
    HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));  // Copy (*dest, *source, Type).

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    if (verbose) printf ("info: launch 'vector_square' kernel\n");
    // operands:       kernelID         blocks        threads                dynamic-shared-memory
    //                 |                |             |                      |  stream
    //                 |                |             |                      |  |  kernel operands ...
    hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    if (verbose) printf ("info: copy Device2Host\n");
    HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));  // copy (*dest, *source, Type).

    if (verbose) printf ("info: check result\n");
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {                 // If value received is not square of value sent,
            HIPCHECK(hipErrorUnknown);                   // ... We have a problem!
        }
    }

   printf("info: hipLaunchKernelGGL completed, device %i.\n", device);
} // end conductTest.

static void
print_help( char **argv )
{
    printf( "Usage: %s [options] [EVENTNAMEs]\n", argv[0] );
    printf( "Options:\n\n" );
    printf( "General command options:\n" );
    printf( "\t-u          Display output values as unsigned integers\n" );
    printf( "\t-x          Display output values as hexadecimal\n" );
    printf( "\t-h          Print this help message\n" );
    printf( "\tEVENTNAMEs  Specify one or more preset or native events\n" );
    printf( "\n" );
    printf( "This utility performs work while measuring the specified events.\n" );
    printf( "All available AMD devices are given work.                       \n" );
    printf( "It can be useful for sanity checks on given events and sets of events.\n" );
}


int
main( int argc, char **argv )
{
    int retval;
    int num_events;
    long long *values;
    char *success;
    PAPI_event_info_t info;
    int EventSet = PAPI_NULL;
    int i, j, k, event, data_type = PAPI_DATATYPE_INT64;
    int u_format = 0;
    int hex_format = 0;

    printf( "\nThis utility lets you add events from the command line "
        "interface to see if they work.\n\n" );

    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if (retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"Error! PAPI_library_init\n");
        exit(retval );
    }

    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_create_eventset\n");
        exit(retval );
    }

    retval = PAPI_add_named_event(EventSet, "rocm_smi:::NUMDevices");   // Number of devices.
 
    if ( retval != PAPI_OK ) {
        printf("Failed adding rocm_smi:::NUMDevices, error='%s'.\n", PAPI_strerror(retval));
        printf("Perhaps no rocm_smi component is available.\n"); 
        printf("Use papi/src/utils/papi_component_avail to check.\n"); 
        exit(-1);
    }

    retval = PAPI_add_named_event(EventSet, "rocm_smi:::rsmi_version");   // Version of Library.
 
    if ( retval != PAPI_OK ) {
        printf("Failed adding rocm_smi:::rsmi_version, error='%s'.\n", PAPI_strerror(retval));
        printf("Perhaps no rocm_smi component is available.\n"); 
        printf("Use papi/src/utils/papi_component_avail to check.\n"); 
        exit(-1);
    }

    long long startupValues[2] = {0,0};
    retval = PAPI_start( EventSet );
    if (retval != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_start, retval=%i [%s].\n", retval, PAPI_strerror(retval) );
        exit( retval );
    }

    retval = PAPI_stop( EventSet, startupValues );
    if (retval != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_read, retval=%i [%s].\n", retval, PAPI_strerror(retval) );
        exit( retval );
    }
    
    PAPI_cleanup_eventset(EventSet);                        // get rid of this set.

    int NUMDevices, major, minor, patch;
    NUMDevices = startupValues[0];
    patch = startupValues[1] & 0x000000000000ffff;          // Extract patch from packed major:minor:patch.
    minor = (startupValues[1]>>4) & 0x000000000000ffff;     // Extract minor.
    major = (startupValues[1]>>8) & 0x000000000000ffff;     // Extract major.
    printf("%i AMD rocm_smi capable devices found. Library version %i:%i:%i.\n", 
        NUMDevices, major, minor, patch);
    
    values = ( long long * ) malloc( sizeof ( long long ) * ( size_t ) argc );  // create reading space.
    success = ( char * ) malloc( ( size_t ) argc );

    if ( success == NULL || values == NULL ) {
        fprintf(stderr,"Error allocating memory!\n");
        exit(1);
    }

    for ( num_events = 0, i = 1; i < argc; i++ ) {
        if ( !strcmp( argv[i], "-h" ) ) {
            print_help( argv );
            exit( 1 );
        } else if ( !strcmp( argv[i], "-u" ) ) {
            u_format = 1;
        } else if ( !strcmp( argv[i], "-x" ) ) {
            hex_format = 1;
        } else {
            if ( ( retval = PAPI_add_named_event( EventSet, argv[i] ) ) != PAPI_OK ) {
                printf( "Failed adding: %s\nbecause: %s\n", argv[i], 
                    PAPI_strerror(retval));
            } else {
                success[num_events++] = i;
                printf( "Successfully added: %s\n", argv[i] );
            }
        }
    }

    /* Automatically pass if no events, for run_tests.sh */
    if ( num_events == 0 ) {
        printf("No events specified!\n");
        printf("Specify events like rocm_smi:::device=0:mem_usage_VRAM rocm_smi:::device=0:pci_throughput_sent\n");
        printf("Use papi/src/utils/papi_native_avail for a list of all events; search for 'rocm_smi:::'.\n");
        return 0;
    }

   // ROCM Activity.
    printf( "\n" );

    retval = PAPI_start( EventSet );
    if (retval != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_start, retval=%i [%s].\n", retval, PAPI_strerror(retval) );
        exit( retval );
    }

        // ROCM skipped do_flops(), do_misses() in papi_command_line.c.

        for (k = 0; k < NUMDevices; k++ ) {           // ROCM loop through devices.
            conductTest(k);                           // Do some GPU work on device 'k'.
            sleep(1);                                 // .. sleep between reads to build up events.

            retval = PAPI_read( EventSet, values );
            if (retval != PAPI_OK ) {
                 fprintf(stderr,"Error! PAPI_read, retval=%i [%s].\n", retval, PAPI_strerror(retval) );
                exit( retval );
            }
        printf( "\n----------------------------------\n" );
        
            for ( j = 0; j < num_events; j++ ) {      // Back to original papi_command_line...
                i = success[j];
                if (! (u_format || hex_format) ) {
                    retval = PAPI_event_name_to_code( argv[i], &event );
                    if (retval == PAPI_OK) {
                        retval = PAPI_get_event_info(event, &info);
                        if (retval == PAPI_OK) data_type = info.data_type;
                        else data_type = PAPI_DATATYPE_INT64;
                    }
                    switch (data_type) {
                      case PAPI_DATATYPE_UINT64:
                        printf( "%s : \t%llu(u)", argv[i], (unsigned long long)values[j] );
                        break;
                      case PAPI_DATATYPE_FP64:
                        printf( "%s : \t%0.3f", argv[i], *((double *)(&values[j])) );
                        break;
                      case PAPI_DATATYPE_BIT64:
                        printf( "%s : \t%#llX", argv[i], values[j] );
                        break;
                      case PAPI_DATATYPE_INT64:
                      default:
                        printf( "%s : \t%lld", argv[i], values[j] );
                        break;
                    }
                    if (retval == PAPI_OK)  printf( " %s", info.units );
                    printf( "\n" );
                }
                if (u_format) printf( "%s : \t%llu(u)\n", argv[i], (unsigned long long)values[j] );
                if (hex_format) printf( "%s : \t%#llX\n", argv[i], values[j] );
            }
        } // end ROCM device loop.

        retval = PAPI_stop( EventSet, values );       // ROCM added stop and test.
        if (retval != PAPI_OK ) {
             fprintf(stderr,"Error! PAPI_stop, retval=%i [%s].\n", retval, PAPI_strerror(retval) );
            exit( retval );
        }

    return 0;
} // end main.
