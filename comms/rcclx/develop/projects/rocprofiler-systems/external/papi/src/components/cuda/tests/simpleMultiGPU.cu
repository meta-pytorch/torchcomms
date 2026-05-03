/* 
 * PAPI Multiple GPU example.  This example is taken from the NVIDIA
 * documentation (Copyright 1993-2013 NVIDIA Corporation) and has been
 * adapted to show the use of CUPTI and PAPI in collecting event
 * counters for multiple GPU contexts.  PAPI Team (2015)
 *
 * Update, July/2021, for CUPTI 11. This version is for the CUPTI 11
 * API, which PAPI uses for Nvidia GPUs with Compute Capability >=
 * 7.0. It will only work on cuda distributions of 10.0 or better.
 * Similar to legacy CUpti API, PAPI is informed of the CUcontexts
 * that will be used to execute kernels at the time of adding PAPI
 * events for that device; as shown below.
 */

/*
 * This software contains source code provided by NVIDIA Corporation
 *
 * According to the Nvidia EULA (compute 5.5 version)
 * http://developer.download.nvidia.com/compute/cuda/5_5/rel/docs/EULA.pdf
 *
 * Chapter 2. NVIDIA CORPORATION CUDA SAMPLES END USER LICENSE AGREEMENT
 * 2.1.1. Source Code
 * Developer shall have the right to modify and create derivative works with the Source
 * Code. Developer shall own any derivative works ("Derivatives") it creates to the Source
 * Code, provided that Developer uses the Materials in accordance with the terms and
 * conditions of this Agreement. Developer may distribute the Derivatives, provided that
 * all NVIDIA copyright notices and trademarks are propagated and used properly and
 * the Derivatives include the following statement: “This software contains source code
 * provided by NVIDIA Corporation.”
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 *
 *  CUDA Context notes for CUPTI_11: Although a cudaSetDevice() will create a
 *  primary context for the device that allows kernel execution; PAPI cannot
 *  use a primary context to control the Nvidia Performance Profiler.
 *  Applications must create a context using cuCtxCreate() that will execute
 *  the kernel, this must be done prior to the PAPI_add_events() invocation in
 *  the code below. When multiple GPUs are in use, each requires its own
 *  context, and that context should be active when PAPI_events are added for
 *  each device. This means using seperate PAPI_add_events() for each device,
 *  as we do here.
 */

// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda.h>
#include <timer.h>

#ifdef PAPI
#include "papi.h"
#include "papi_test.h"
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

// //////////////////////////////////////////////////////////////////////////////
// Data configuration
// //////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N = 48576 * 32;
#ifdef PAPI
const int MAX_NUM_EVENTS = 32;
#endif

#define CHECK_CU_ERROR(err, cufunc)                                     \
    if (err != CUDA_SUCCESS) { fprintf (stderr, "Error %d for CUDA Driver API function '%s'\n", err, cufunc); return -1; }

#define CHECK_CUDA_ERROR(err)                                           \
    if (err != cudaSuccess) { fprintf (stderr, "%s:%i Error %d for CUDA [%s]\n", __FILE__, __LINE__, err, cudaGetErrorString(err) ); return -1; }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
    if (err != CUPTI_SUCCESS) { const char *errStr; cuptiGetResultString(err, &errStr); \
       fprintf (stderr, "%s:%i Error %d [%s] for CUPTI API function '%s'\n", __FILE__, __LINE__, err, errStr, cuptifunc); return -1; }

#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}

// //////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA SDK sample describing
// reduction optimization strategies
// //////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel( float *d_Result, float *d_Input, int N )
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float sum = 0;

    for( int pos = tid; pos < N; pos += threadN )
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

// //////////////////////////////////////////////////////////////////////////////
// Program main
// //////////////////////////////////////////////////////////////////////////////
int main( int argc, char **argv )
{
    // Solver config
    TGPUplan plan[MAX_GPU_COUNT];
    // GPU reduction results
    float h_SumGPU[MAX_GPU_COUNT];
    float sumGPU;
    double sumCPU, diff;
    int i, j, gpuBase, num_gpus;

    const int BLOCK_N = 32;
    const int THREAD_N = 256;
    const int ACCUM_N = BLOCK_N * THREAD_N;

    CUcontext ctx[MAX_GPU_COUNT];
    CUcontext poppedCtx;

	char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    int quiet = 0;
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    PRINT( quiet, "Starting simpleMultiGPU\n" );

#ifdef PAPI
    int event_count = argc - 1;

    /* if no events passed at command line, just report test skipped. */
    if (event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    /* PAPI Initialization must occur before any context creation/manipulation. */
    /* This is to ensure PAPI can monitor CUpti library calls.                  */
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed\n" );
        exit(-1);
    }

    printf( "PAPI version: %d.%d.%d\n", PAPI_VERSION_MAJOR( PAPI_VERSION ), PAPI_VERSION_MINOR( PAPI_VERSION ), PAPI_VERSION_REVISION( PAPI_VERSION ) );
#endif 
    
    // Report on the available CUDA devices
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    int runtimeVersion = 0, driverVersion = 0;
    char deviceName[64];
    CUdevice device[MAX_GPU_COUNT];
    CHECK_CUDA_ERROR( cudaGetDeviceCount( &num_gpus ) );
    if( num_gpus > MAX_GPU_COUNT ) num_gpus = MAX_GPU_COUNT;
    PRINT( quiet, "CUDA-capable device count: %i\n", num_gpus );
    for ( i=0; i<num_gpus; i++ ) {
        CHECK_CU_ERROR( cuDeviceGet( &device[i], i ), "cuDeviceGet" );
        CHECK_CU_ERROR( cuDeviceGetName( deviceName, 64, device[i] ), "cuDeviceGetName" );
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMajor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device[i]), "cuDeviceGetAttribute");
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMinor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device[i]), "cuDeviceGetAttribute");
        cudaRuntimeGetVersion( &runtimeVersion );
        cudaDriverGetVersion( &driverVersion );
        PRINT( quiet, "CUDA Device %d: %s : computeCapability %d.%d runtimeVersion %d.%d driverVersion %d.%d\n",
                i, deviceName, computeCapabilityMajor, computeCapabilityMinor, runtimeVersion/1000, (runtimeVersion%100)/10, driverVersion/1000, (driverVersion%100)/10 );
        if ( computeCapabilityMajor < 2 ) {
            fprintf( stderr, "CUDA Device %d compute capability is too low... will not add any more GPUs\n", i );
            num_gpus = i;
            break;
        }
    }

    // create one context per device. This can be delayed
    // to as late as PAPI_start(), but they are needed to
    // create streams, alloc memory, etc.
    for (i = 0; i < num_gpus; i++) {
        CHECK_CU_ERROR( cuCtxCreate( &(ctx[i]), 0, device[i] ), "cuCtxCreate" ); // automatically pushes the new context on the stack.
        CHECK_CU_ERROR( cuCtxPopCurrent(&poppedCtx), "cuCtxPopCurrent" );        // ... so take it off.
    }

    PRINT( quiet, "Generating input data...\n" );

    // Subdividing input data across GPUs
    // Get data sizes for each GPU
    for( i = 0; i < num_gpus; i++ )
        plan[i].dataN = DATA_N / num_gpus;
    // Take into account "odd" data sizes
    for( i = 0; i < DATA_N % num_gpus; i++ )
        plan[i].dataN++;

    // Assign data ranges to GPUs
    gpuBase = 0;
    for( i = 0; i < num_gpus; i++ ) {
        plan[i].h_Sum = h_SumGPU + i; // point within h_SumGPU array
        gpuBase += plan[i].dataN;
    }

    // Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for( i = 0; i < num_gpus; i++ ) {
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CUDA_ERROR( cudaStreamCreate( &plan[i].stream ) );
        CHECK_CUDA_ERROR( cudaMalloc( ( void ** ) &plan[i].d_Data, plan[i].dataN * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMalloc( ( void ** ) &plan[i].d_Sum, ACCUM_N * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMallocHost( ( void ** ) &plan[i].h_Sum_from_device, ACCUM_N * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMallocHost( ( void ** ) &plan[i].h_Data, plan[i].dataN * sizeof( float ) ) );
        for( j = 0; j < plan[i].dataN; j++ ) {
            plan[i].h_Data[j] = ( float ) rand() / ( float ) RAND_MAX;
        }
        CHECK_CU_ERROR( cuCtxPopCurrent(&poppedCtx), "cuCtxPopCurrent" );
    }

#ifdef PAPI
    PRINT(quiet, "Setup PAPI counters internally (PAPI)\n");
    int EventSet = PAPI_NULL;
    int NUM_EVENTS = MAX_GPU_COUNT*MAX_NUM_EVENTS;
    long long values[NUM_EVENTS];
    int total_events;
    int ee;

    int cid = PAPI_get_component_index("cuda");
    if (cid < 0) {
        PAPI_shutdown();
        test_fail(__FILE__, __LINE__, "Failed to get index of cuda component.", PAPI_ECMP);
    }

    PRINT(quiet, "Found CUDA Component at id %d\n", cid);

    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed.", papi_errno);
    }

    papi_errno = PAPI_assign_eventset_component(EventSet, cid);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_assign_eventset_component failed.", papi_errno);
    }

    // In this example measure events from each GPU
    // Add events at a GPU specific level ... eg cuda:::device:2:elapsed_cycles_sm
    // Similar to legacy CUpti API, we must change the contexts to the appropriate device to
    // add events to inform PAPI of the context that will run the kernels.

    // Save current context, will restore after adding events.
    CUcontext userContext;
    CHECK_CU_ERROR(cuCtxGetCurrent(&userContext), "cuCtxGetCurrent");

    char *EventName[NUM_EVENTS];
    char tmpEventName[64];
    total_events = 0;
    for( i = 0; i < num_gpus; i++ ) {
        CHECK_CU_ERROR(cuCtxSetCurrent(ctx[i]), "cuCtxSetCurrent");
        for ( ee=0; ee < event_count; ee++ ) {
            // Create a device specific event.
            snprintf( tmpEventName, 64, "%s:device=%d", argv[ee+1], i );
            papi_errno = PAPI_add_named_event( EventSet, tmpEventName );
            if (papi_errno==PAPI_OK) {
                PRINT( quiet, "Add event success: '%s' GPU %i\n", tmpEventName, i );
                EventName[total_events] = (char *)calloc( 64, sizeof(char) );
                if (EventName[total_events] == NULL) {
                    test_fail(__FILE__, __LINE__, "Failed to allocate string.\n", 0);
                }
                snprintf( EventName[total_events], 64, "%s", tmpEventName );
                total_events++;
            } else {
                fprintf( stderr, "Add event failure: '%s' GPU %i error=%s\n", tmpEventName, i, PAPI_strerror(papi_errno));
                test_skip(__FILE__, __LINE__, "", 0);
            }
        }
    }

    // Restore user context.

    CHECK_CU_ERROR(cuCtxSetCurrent(userContext), "cuCtxSetCurrent");

    // Invoke PAPI_start().
    papi_errno = PAPI_start( EventSet );
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__, __LINE__, "PAPI_start failed", papi_errno);
    }
#endif

    // Start timing and compute on GPU(s)
    PRINT( quiet, "Computing with %d GPUs...\n", num_gpus );
    StartTimer();

    // Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < num_gpus; i++) {
        // Pushing a context implicitly sets the device for which it was created.
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        // Copy input data from CPU
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof( float ), cudaMemcpyHostToDevice, plan[i].stream ) );
        // Perform GPU computations
        reduceKernel <<< BLOCK_N, THREAD_N, 0, plan[i].stream >>> ( plan[i].d_Sum, plan[i].d_Data, plan[i].dataN );
        if ( cudaGetLastError() != cudaSuccess ) { printf( "reduceKernel() execution failed (GPU %d).\n", i ); exit(EXIT_FAILURE); }
        // Read back GPU results
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof( float ), cudaMemcpyDeviceToHost, plan[i].stream ) );
        // Popping a context can change the device to match the previous context.
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    // Process GPU results
    PRINT( quiet, "Process GPU results on %d GPUs...\n", num_gpus );
    for( i = 0; i < num_gpus; i++ ) {
        float sum;
        // Pushing a context implicitly sets the device for which it was created.
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        // Wait for all operations to finish
        cudaStreamSynchronize( plan[i].stream );
        // Finalize GPU reduction for current subvector
        sum = 0;
        for( j = 0; j < ACCUM_N; j++ ) {
            sum += plan[i].h_Sum_from_device[j];
        }
        *( plan[i].h_Sum ) = ( float ) sum;
        // Popping a context can change the device to match the previous context.
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    double gpuTime = GetTimer();


#ifdef PAPI
    for ( i=0; i<num_gpus; i++ ) {
        // Pushing a context implicitly sets the device for which it was created.
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
        // Popping a context may change the current device to match the new current context.
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    papi_errno = PAPI_stop( EventSet, values );                                         // Stop (will read values).
    if( papi_errno != PAPI_OK )  fprintf( stderr, "PAPI_stop failed\n" );
    for( i = 0; i < total_events; i++ )
        PRINT( quiet, "PAPI counterValue %12lld \t\t --> %s \n", values[i], EventName[i] );

    papi_errno = PAPI_cleanup_eventset( EventSet );
    if( papi_errno != PAPI_OK )  fprintf( stderr, "PAPI_cleanup_eventset failed\n" );
    papi_errno = PAPI_destroy_eventset( &EventSet );
    if( papi_errno != PAPI_OK ) fprintf( stderr, "PAPI_destroy_eventset failed\n" );
    PAPI_shutdown();
#endif

    sumGPU = 0;
    for( i = 0; i < num_gpus; i++ ) {
        sumGPU += h_SumGPU[i];
    }
    PRINT( quiet, "  GPU Processing time: %f (ms)\n", gpuTime );

    // Compute on Host CPU
    PRINT( quiet, "Computing the same result with Host CPU...\n" );
    StartTimer();
    sumCPU = 0;
    for( i = 0; i < num_gpus; i++ ) {
        for( j = 0; j < plan[i].dataN; j++ ) {
            sumCPU += plan[i].h_Data[j];
        }
    }
    double cpuTime = GetTimer();
    if (gpuTime > 0) {
        PRINT( quiet, "  CPU Processing time: %f (ms) (speedup %.2fX)\n", cpuTime, (cpuTime/gpuTime) );
    } else {
        PRINT( quiet, "  CPU Processing time: %f (ms)\n", cpuTime);
    }

    // Compare GPU and CPU results
    PRINT( quiet, "Comparing GPU and Host CPU results...\n" );
    diff = fabs( sumCPU - sumGPU ) / fabs( sumCPU );
    PRINT( quiet, "  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU );
    PRINT( quiet, "  Relative difference: %E \n", diff );

    // Cleanup and shutdown
    for( i = 0; i < num_gpus; i++ ) {
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Sum_from_device ) );
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Data ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Sum ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Data ) );
        // Shut down this GPU
        CHECK_CUDA_ERROR( cudaStreamDestroy( plan[i].stream ) );
        CHECK_CU_ERROR( cuCtxDestroy(ctx[i]), "cuCtxDestroy");
    }

#ifdef PAPI
    if ( diff < 1e-5 )
        test_pass(__FILE__);
    else
        test_fail(__FILE__, __LINE__, "Result of GPU calculation doesn't match CPU.", PAPI_EINVAL);
#endif
    return 0;
}
