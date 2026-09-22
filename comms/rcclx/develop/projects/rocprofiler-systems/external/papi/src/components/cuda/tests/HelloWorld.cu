/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    HelloWorld.cu
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	Anustuv Pal
 *			anustuv@icl.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * test case for Example component
 *
 *
 * @brief
 *  This file is a very simple HelloWorld C example which serves (together
 *	with its Makefile) as a guideline on how to add tests to components.
 *  The papi configure and papi Makefile will take care of the compilation
 *	of the component tests (if all tests are added to a directory named
 *	'tests' in the specific component dir).
 *	See components/README for more details.
 *
 *	The string "Hello World!" is mangled and then restored.
 *
 *  CUDA Context notes for CUPTI_11: Although a cudaSetDevice() will create a
 *  primary context for the device that allows kernel execution; PAPI cannot
 *  use a primary context to control the Nvidia Performance Profiler.
 *  Applications must create a context using cuCtxCreate() that will execute
 *  the kernel, this must be done prior to the PAPI_add_events() invocation in
 *  the code below. If multiple GPUs are in use, each requires its own context,
 *  and that context should be active when PAPI_events are added for each
 *  device.  Which means using Seperate PAPI_add_events() for each device. For
 *  an example see simpleMultiGPU.cu.
 *
 *  There are three points below where cuCtxCreate() is called, this code works
 *  if any one of them is used alone.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PAPI
#include "papi.h"
#include "papi_test.h"
#endif

#define STEP_BY_STEP_DEBUG 0 /* helps debug CUcontext issues. */
#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}

// Device kernel
__global__ void helloWorld(char* str)
{
        // determine where in the thread grid we are
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // unmangle output
        str[idx] += idx;
}

/** @class add_events_from_command_line
  * @brief Try and add each event provided on the command line by the user.
  *
  * @param EventSet
  *   A PAPI eventset.
  * @param totalEventCount
  *   Number of events from the command line.
  * @param **eventNamesFromCommandLine
  *   Events provided on the command line.
  * @param *numEventsSuccessfullyAdded
  *   Total number of successfully added events.
  * @param **eventsSuccessfullyAdded
  *   Events that we are able to add to the EventSet.
  * @param *numMultipassEvents
  *   Counter to see if a multiple pass event was provided on the command line.
*/
static void add_events_from_command_line(int EventSet, int totalEventCount, char **eventNamesFromCommandLine, int *numEventsSuccessfullyAdded, char **eventsSuccessfullyAdded, int *numMultipassEvents)
{
    int i;
    for (i = 0; i < totalEventCount; i++) {
        int papi_errno = PAPI_add_named_event(EventSet, eventNamesFromCommandLine[i]);
        if (papi_errno != PAPI_OK) {
            if (papi_errno != PAPI_EMULPASS) {
                fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", eventNamesFromCommandLine[i], papi_errno);
                test_skip(__FILE__, __LINE__, "", 0);
            }

            // Handle multiple pass events
            (*numMultipassEvents)++;
            continue;
        }

        // Handle successfully added events
        int strLen = snprintf(eventsSuccessfullyAdded[(*numEventsSuccessfullyAdded)], PAPI_MAX_STR_LEN, "%s", eventNamesFromCommandLine[i]);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write successfully added event.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
        (*numEventsSuccessfullyAdded)++;
    }

    return;
}

// Host function
int main(int argc, char** argv)
{
    int quiet = 0;
    CUcontext getCtx=NULL, sessionCtx=NULL;
    cudaError_t cudaError;
    CUresult cuError; (void) cuError;

    cuError = cuInit(0);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize the CUDA driver API.\n");
        exit(1);
    }

#ifdef PAPI
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    /* PAPI Initialization */
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__,__LINE__, "PAPI_library_init failed", 0 );
    }

    printf( "PAPI_VERSION     : %4d %6d %7d\n",
        PAPI_VERSION_MAJOR( PAPI_VERSION ),
        PAPI_VERSION_MINOR( PAPI_VERSION ),
        PAPI_VERSION_REVISION( PAPI_VERSION ) );

    int i;
    int EventSet = PAPI_NULL;
    int eventCount = argc - 1;

    /* if no events passed at command line, just report test skipped. */
    if (eventCount == 0) {
        fprintf(stderr, "No eventnames specified at command line.");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    long long *values = (long long *) calloc(eventCount, sizeof (long long));
    if (values == NULL) {
        test_fail(__FILE__, __LINE__, "Failed to allocate memory for values.\n", 0);
    }

    int *events = (int *) calloc(eventCount, sizeof (int));
    if (events == NULL) {
        test_fail(__FILE__, __LINE__, "Failed to allocate memory for events.\n", 0);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i before PAPI_create_eventset() getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    papi_errno = PAPI_create_eventset( &EventSet );
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__,__LINE__,"Cannot create eventset",papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_create_eventset() getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // If multiple GPUs/contexts were being used, you'd need to
    // create contexts for each device. See, for example,
    // simpleMultiGPU.cu.
    int flags = 0;
    CUdevice device = 0;
#if defined(CUDA_TOOLKIT_GE_13)
    cuError = cuCtxCreate(&sessionCtx, (CUctxCreateParams*)0, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version >= 13: %d\n", cuError);
        exit(1);
    }
#else
    cuError = cuCtxCreate(&sessionCtx, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version < 13: %d\n", cuError);
        exit(1);
    }
#endif

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cuCtxCreate(&sessionCtx), about to PAPI_start(), sessionCtx=%p, getCtx=%p.\n", __FILE__, __func__, __LINE__, sessionCtx, getCtx);
    }

    // Handle the events from the command line
    int numEventsSuccessfullyAdded = 0, numMultipassEvents = 0;
    char **eventsSuccessfullyAdded, **metricNames = argv + 1;
    eventsSuccessfullyAdded = (char **) malloc(eventCount * sizeof(char *));
    if (eventsSuccessfullyAdded == NULL) {
        fprintf(stderr, "Failed to allocate memory for successfully added events.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    for (i = 0; i < eventCount; i++) {
        eventsSuccessfullyAdded[i] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        if (eventsSuccessfullyAdded[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for command line argument.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    add_events_from_command_line(EventSet, eventCount, metricNames, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

    // Only multiple pass events were provided on the command line
    if (numEventsSuccessfullyAdded == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i before PAPI_start(), getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    papi_errno = PAPI_start( EventSet );
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__, __LINE__, "PAPI_start failed.", papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_start(), getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

#endif

    int j;

    // desired output
    char str[] = "Hello World!";

    // mangle contents of output
    // the null character is left intact for simplicity
    for(j = 0; j < 12; j++) {
        str[j] -= j;
    }

    PRINT(quiet, "mangled str=%s\n", str);

    // allocate memory on the device
    char *d_str;
    size_t size = sizeof(str);
    cudaMalloc((void**)&d_str, size);

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cudaMalloc() getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // copy the string to the device
    cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cudaMemcpy(ToDevice) getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // set the grid and block sizes
    dim3 dimGrid(2); // one block per word
    dim3 dimBlock(6); // one thread per character

    // invoke the kernel
    helloWorld<<< dimGrid, dimBlock >>>(d_str);

    cudaError = cudaGetLastError();
    if (STEP_BY_STEP_DEBUG) {
        fprintf(stderr, "%s:%s:%i Kernel Return Code: %s.\n", __FILE__, __func__, __LINE__, cudaGetErrorString(cudaError));
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i After Kernel Execution: getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // retrieve the results from the device
    cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cudaMemcpy(ToHost) getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // free up the allocated memory on the device
    cudaFree(d_str);

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cudaFree() getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }


#ifdef PAPI
    papi_errno = PAPI_read( EventSet, values );
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__, __LINE__, "PAPI_read failed", papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_read getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    for( i = 0; i < numEventsSuccessfullyAdded; i++ ) {
        PRINT( quiet, "read: %12lld \t=0X%016llX \t\t --> %s \n", values[i], values[i], eventsSuccessfullyAdded[i] );
    }

    papi_errno = cuCtxPopCurrent(&getCtx);
    if( papi_errno != CUDA_SUCCESS) {
        fprintf( stderr, "cuCtxPopCurrent failed, papi_errno=%d (%s)\n", papi_errno, PAPI_strerror(papi_errno) );
        exit(1);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cuCtxPopCurrent() getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    papi_errno = PAPI_stop( EventSet, values );
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__, __LINE__, "PAPI_stop failed", papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_stop getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    papi_errno = PAPI_cleanup_eventset(EventSet);
    if( papi_errno != PAPI_OK ) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset failed", papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_cleanup_eventset getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    papi_errno = PAPI_destroy_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset failed", papi_errno);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_destroy_eventset getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    for( i = 0; i < numEventsSuccessfullyAdded; i++ ) {
        PRINT( quiet, "stop: %12lld \t=0X%016llX \t\t --> %s \n", values[i], values[i], eventsSuccessfullyAdded[i] );
    }
#endif

    if (STEP_BY_STEP_DEBUG) {
        fprintf(stderr, "%s:%s:%i before cuCtxDestroy sessionCtx=%p.\n", __FILE__, __func__, __LINE__, sessionCtx);
    }

    // Test destroying the session Context.
    if (sessionCtx != NULL) {
        cuCtxDestroy(sessionCtx);
    }

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after cuCtxDestroy(%p) getCtx=%p.\n", __FILE__, __func__, __LINE__, sessionCtx, getCtx);
    }

    // Free allocated memory
    free(values);
    free(events);
    for (i = 0; i < eventCount; i++) {
        free(eventsSuccessfullyAdded[i]);
    }
    free(eventsSuccessfullyAdded);

#ifdef PAPI
    PAPI_shutdown();

    if (STEP_BY_STEP_DEBUG) {
        cuCtxGetCurrent(&getCtx);
        fprintf(stderr, "%s:%s:%i after PAPI_shutdown getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);
    }

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(quiet, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    test_pass(__FILE__);
#endif
    return 0;
}
