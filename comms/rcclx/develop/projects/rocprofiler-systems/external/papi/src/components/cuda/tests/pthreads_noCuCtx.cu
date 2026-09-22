/**
 * @file    pthreads_noCuCtx.cu
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
#include "papi_test.h"

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

#define MAX_THREADS (32)

int numGPUs;
int g_event_count;
char **g_evt_names;

static volatile int global_thread_count = 0;
pthread_mutex_t global_mutex;
pthread_t tidarr[MAX_THREADS];
CUcontext cuCtx[MAX_THREADS];
pthread_mutex_t lock;

// Globals for multiple pass events
int numMultipassEvents = 0;

/** @class add_events_from_command_line
  * @brief Try and add each event provided on the command line by the user.
  *
  * @param EventSet
  *   A PAPI eventset.
  * @param totalEventCount
  *   Number of events from the command line.
  * @param **eventNamesFromCommandLine
  *   Events provided on the command line.
  * @param gpu_id
  *   NVIDIA device index.
  * @param *numEventsSuccessfullyAdded
  *   Total number of successfully added events.
  * @param **eventsSuccessfullyAdded
  *   Events that we are able to add to the EventSet.
  * @param *numMultipassEvents
  *   Counter to see if a multiple pass event was provided on the command line.
*/
static void add_events_from_command_line(int EventSet, int totalEventCount, char **eventNamesFromCommandLine, int gpu_id, int *numEventsSuccessfullyAdded, char **eventsSuccessfullyAdded, int *numMultipassEvents)
{
    int i;
    for (i = 0; i < totalEventCount; i++) {
        char tmpEventName[PAPI_MAX_STR_LEN];
        int strLen = snprintf(tmpEventName, PAPI_MAX_STR_LEN, "%s:device=%d", eventNamesFromCommandLine[i], gpu_id);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write event name with appended device qualifier.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }

        int papi_errno = PAPI_add_named_event(EventSet, tmpEventName);
        if (papi_errno != PAPI_OK) {
            if (papi_errno != PAPI_EMULPASS) {
                fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", tmpEventName, papi_errno);
                test_skip(__FILE__, __LINE__, "", 0);
            }

            // Handle multiple pass events
            (*numMultipassEvents)++;
            continue;
        }

        // Handle successfully added events
        strLen = snprintf(eventsSuccessfullyAdded[(*numEventsSuccessfullyAdded)], PAPI_MAX_STR_LEN, "%s", tmpEventName);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write successfully added event.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
        (*numEventsSuccessfullyAdded)++;
    }

    return;
}

void *thread_gpu(void * idx)
{
    int tid = *((int*) idx);
    unsigned long gettid = (unsigned long) pthread_self();

#ifdef PAPI
    int gpuid = tid % numGPUs;
    int i;
    int EventSet = PAPI_NULL;
    long long values[MAX_THREADS];
    PAPI_CALL(PAPI_create_eventset(&EventSet));

    RUNTIME_API_CALL(cudaSetDevice(gpuid));
    PRINT(quiet, "This is idx %d thread %lu - using GPU %d\n",
            tid, gettid, gpuid);

    int numEventsSuccessfullyAdded = 0;
    char **eventsSuccessfullyAdded;
    eventsSuccessfullyAdded = (char **) malloc(g_event_count * sizeof(char *));
    if (eventsSuccessfullyAdded == NULL) {
        fprintf(stderr, "Failed to allocate memory for successfully added events.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    for (i = 0; i < g_event_count; i++) {
        eventsSuccessfullyAdded[i] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        if (eventsSuccessfullyAdded[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for command line argument.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    pthread_mutex_lock(&global_mutex);

    add_events_from_command_line(EventSet, g_event_count, g_evt_names, gpuid, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

    // Only multiple pass events were provided on the command line
    if (numEventsSuccessfullyAdded == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    ++global_thread_count;
    pthread_mutex_unlock(&global_mutex);

    while(global_thread_count < numGPUs);

    PAPI_CALL(PAPI_start(EventSet));
#endif

    VectorAddSubtract(50000*(tid+1), quiet);  // gpu work

#ifdef PAPI
    PAPI_CALL(PAPI_stop(EventSet, values));

    PRINT(quiet, "User measured values in thread id %d.\n", tid);
    for (i = 0; i < numEventsSuccessfullyAdded; i++) {
        PRINT(quiet, "%s\t\t%lld\n", eventsSuccessfullyAdded[i], values[i]);
    }

    // Free allocated memory
    for (i = 0; i < g_event_count; i++) {
        free(eventsSuccessfullyAdded[i]);
    }
    free(eventsSuccessfullyAdded);

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
    int tid[MAX_THREADS];

    RUNTIME_API_CALL(cudaGetDeviceCount(&numGPUs));
    PRINT(quiet, "No. of GPUs = %d\n", numGPUs);
    if (numGPUs < 1) {
        fprintf(stderr, "No GPUs found on system.\n");
#ifdef PAPI
        test_skip(__FILE__, __LINE__, "", 0);
#endif
        return 0;
    }
    if (numGPUs > MAX_THREADS)
        numGPUs = MAX_THREADS;
    PRINT(quiet, "No. of threads to launch = %d\n", numGPUs);

#ifdef PAPI
    pthread_mutex_init(&global_mutex, NULL);
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
    // Point PAPI to function that gets the thread id
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) pthread_self));
#endif

    // Launch the threads
    for(i = 0; i < numGPUs; i++)
    {
        tid[i] = i;
        RUNTIME_API_CALL(cudaSetDevice(tid[i] % numGPUs));
        RUNTIME_API_CALL(cudaFree(NULL));

        rc = pthread_create(&tidarr[i], NULL, thread_gpu, &(tid[i]));
        if(rc)
        {
            fprintf(stderr, "\n ERROR: return code from pthread_create is %d \n", rc);
            exit(1);
        }
        PRINT(quiet, "\n Main thread %lu. Created new thread (%lu) in iteration %d ...\n",
                (unsigned long)pthread_self(), (unsigned long)tidarr[i], i);
    }

    // Join all threads when complete
    for (i = 0; i < numGPUs; i++) {
        pthread_join(tidarr[i], NULL);
        PRINT(quiet, "IDX: %d: TID: %lu: Done! Joined main thread.\n", i, (unsigned long)tidarr[i]);
    }

#ifdef PAPI
    PAPI_shutdown();

    PRINT(quiet, "Main thread exit!\n");

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(quiet, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    test_pass(__FILE__);
#endif
    return 0;
}
