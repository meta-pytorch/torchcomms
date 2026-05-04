/**
 * @file    test_2thr_1gpu_not_allowed.cu
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gpu_work.h"

#ifdef PAPI
#include <papi.h>
#include <papi_test.h>

#define PAPI_CALL(apiFuncCall)                                          \
do {                                                                           \
    int _status = apiFuncCall;                                         \
    if (_status != PAPI_OK) {                                              \
        fprintf(stderr, "error: function %s failed.", #apiFuncCall);  \
        test_fail(__FILE__, __LINE__, "", _status);  \
    }                                                                          \
} while (0)
#endif

#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}
int quiet;

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define NUM_THREADS 2

int numGPUs;
int g_event_count;
char **g_evt_names;

typedef struct pthread_params_s {
    pthread_t tid;
    CUcontext cuCtx;
    int idx;
    int retval;
} pthread_params_t;

void *thread_gpu(void * ptinfo)
{
    pthread_params_t *tinfo = (pthread_params_t *) ptinfo;
    int idx = tinfo->idx;
    int gpuid = idx % numGPUs;
    unsigned long gettid = (unsigned long) pthread_self();

    DRIVER_API_CALL(cuCtxSetCurrent(tinfo->cuCtx));
    PRINT(quiet, "This is idx %d thread %lu - using GPU %d context %p!\n",
            idx, gettid, gpuid, tinfo->cuCtx);

#ifdef PAPI
    int papi_errno;
    int EventSet = PAPI_NULL;
    long long values[1];
    PAPI_CALL(PAPI_create_eventset(&EventSet));

    papi_errno = PAPI_add_named_event(EventSet, g_evt_names[idx]);
    if (papi_errno != PAPI_OK) {
        fprintf(stderr, "Failed to add event %s\n", g_evt_names[idx]);
        test_skip(__FILE__, __LINE__, "", 0);
    }

    papi_errno = PAPI_start(EventSet);
    if (papi_errno == PAPI_ECNFLCT) {
        PRINT(quiet, "Thread %d was not allowed to start profiling on same GPU.\n", tinfo->idx);
        tinfo->retval = papi_errno;
        return NULL;
    }
#endif

    VectorAddSubtract(5000000*(idx+1), quiet);  // gpu work

#ifdef PAPI
    PAPI_CALL(PAPI_stop(EventSet, values));

    PRINT(quiet, "User measured values in thread id %d.\n", idx);
    PRINT(quiet, "%s\t\t%lld\n", g_evt_names[idx], values[0]);
    tinfo->retval = PAPI_OK;

    PAPI_CALL(PAPI_cleanup_eventset(EventSet));
    PAPI_CALL(PAPI_destroy_eventset(&EventSet));
#endif
    return NULL;
}

int main(int argc, char **argv)
{
    quiet = 0;
#ifdef PAPI
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);
    g_event_count = argc - 1;
    /* if no events passed at command line, just report test skipped. */
    if (g_event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    g_evt_names = argv + 1;
#endif
    int rc, i;
    pthread_params_t data[NUM_THREADS];

    RUNTIME_API_CALL(cudaGetDeviceCount(&numGPUs));
    PRINT(quiet, "No. of GPUs = %d\n", numGPUs);
    PRINT(quiet, "No. of threads to launch = %d\n", NUM_THREADS);

#ifdef PAPI
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
    // Point PAPI to function that gets the thread id
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) pthread_self));
#endif
    // Launch the threads
    for(i = 0; i < NUM_THREADS; i++)
    {
        data[i].idx = i;
        DRIVER_API_CALL(cuCtxCreate(&(data[i].cuCtx), 0, 0));
        DRIVER_API_CALL(cuCtxPopCurrent(&(data[i].cuCtx)));

        rc = pthread_create(&data[i].tid, NULL, thread_gpu, &(data[i]));
        if(rc)
        {
            fprintf(stderr, "\n ERROR: return code from pthread_create is %d \n", rc);
            exit(1);
        }
        PRINT(quiet, "\n Main thread %lu. Created new thread (%lu) in iteration %d ...\n",
                (unsigned long)pthread_self(), (unsigned long) data[i].tid, i);
    }

    // Join all threads when complete
    for (i=0; i<NUM_THREADS; i++) {
        pthread_join(data[i].tid, NULL);
        PRINT(quiet, "IDX: %d: TID: %lu: Done! Joined main thread.\n", i, (unsigned long)data[i].tid);
    }

    // Destroy all CUDA contexts for all threads/GPUs
    for (i=0; i<NUM_THREADS; i++) {
        DRIVER_API_CALL(cuCtxDestroy(data[i].cuCtx));
    }

#ifdef PAPI
    PAPI_shutdown();
    // Check test pass/fail
    int retval = PAPI_OK;
    for (i=0; i<NUM_THREADS; i++) {
        retval += data[i].retval;
    }
    if (retval == PAPI_ECNFLCT)
        test_pass(__FILE__);
    else
        test_fail(__FILE__, __LINE__, "Test condition not satisfied.", 0);
#else
    fprintf(stderr, "Please compile with -DPAPI to test this feature.\n");
#endif
    return 0;
}
