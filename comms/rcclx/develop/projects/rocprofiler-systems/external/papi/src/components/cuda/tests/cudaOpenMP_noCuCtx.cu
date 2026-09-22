/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Multi-GPU sample using OpenMP for threading on the CPU side
 * needs a compiler that supports OpenMP 2.0
 */

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

#include "gpu_work.h"
#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe

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


/** @class add_events_from_command_line
  * @brief Try and add each event provided on the command line by the user.
  *
  * @param EventSet
  *   A PAPI eventset.
  * @param totalEventCount
  *   Number of events from the command line.
  * @param gpu_id
  *   NVIDIA device index.
  * @param eventNamesFromCommandLine
  *   Events provided on the command line.
  * @param *numEventsSuccessfullyAdded
  *   Total number of successfully added events.
  * @param **eventsSuccessfullyAdded
  *   Events that we are able to add to the EventSet.
  * @param *numMultipassEvents
  *   Counter to see if a multiple pass event was provided on the command line.
*/
static void add_events_from_command_line(int EventSet, int totalEventCount, int gpu_id, char **eventNamesFromCommandLine, int *numEventsSuccessfullyAdded, char **eventsSuccessfullyAdded, int *numMultipassEvents)
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

int main(int argc, char *argv[])
{
    quiet = 0;
#ifdef PAPI
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    int event_count = argc - 1;
    /* if no events passed at command line, just report test skipped. */
    if (event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
#endif

    int num_gpus = 0, i;

    RUNTIME_API_CALL(cudaGetDeviceCount(&num_gpus));  // determine the number of CUDA capable GPUs

    if (num_gpus < 1) {
        fprintf(stderr, "no CUDA capable devices were detected\n");
#ifdef PAPI
        test_skip(__FILE__, __LINE__, "", 0);
#endif
        return 0;
    }
    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    PRINT(quiet, "number of host CPUs:\t%d\n", omp_get_num_procs());
    PRINT(quiet, "number of CUDA devices:\t%d\n", num_gpus);

    for (i = 0; i < num_gpus; i++) {
        cudaDeviceProp dprop;
        RUNTIME_API_CALL(cudaGetDeviceProperties(&dprop, i));
        PRINT(quiet, "   %d: %s\n", i, dprop.name);

        RUNTIME_API_CALL(cudaSetDevice(i));
        RUNTIME_API_CALL(cudaFree(NULL));
    }

    int num_threads = (num_gpus > MAX_THREADS) ? MAX_THREADS : num_gpus;

    PRINT(quiet, "---------------------------\n");
#ifdef PAPI
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num));
#endif

    omp_lock_t lock;
    omp_init_lock(&lock);

    omp_set_num_threads(num_threads);  // create as many CPU threads as there are CUDA devices
    int numMultipassEvents = 0;
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_id = cpu_thread_id % num_gpus;
        RUNTIME_API_CALL(cudaSetDevice(gpu_id));

#ifdef PAPI
        int EventSet = PAPI_NULL;
        long long values[MAX_THREADS];
        int j, errno;
        PAPI_CALL(PAPI_create_eventset(&EventSet));

        PRINT(quiet, "CPU thread %d (of %d) uses CUDA device %d @ eventset %d\n", cpu_thread_id, num_cpu_threads, gpu_id, EventSet);

        int numEventsSuccessfullyAdded = 0;
        char **eventsSuccessfullyAdded, **metricNames = argv + 1;
        eventsSuccessfullyAdded = (char **) malloc(event_count * sizeof(char *));
        if (eventsSuccessfullyAdded == NULL) {
            fprintf(stderr, "Failed to allocate memory for successfully added events.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
        for (i = 0; i < event_count; i++) {
            eventsSuccessfullyAdded[i] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
            if (eventsSuccessfullyAdded[i] == NULL) {
                fprintf(stderr, "Failed to allocate memory for command line argument.\n");
                test_skip(__FILE__, __LINE__, "", 0);
            }
        }

        add_events_from_command_line(EventSet, event_count, gpu_id, metricNames, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

        // Only multiple pass events were provided on the command line
        if (numEventsSuccessfullyAdded == 0) {
            fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }

        PAPI_CALL(PAPI_start(EventSet));
#endif
        VectorAddSubtract(50000*(cpu_thread_id+1), quiet);  // gpu work
#ifdef PAPI
        PAPI_CALL(PAPI_stop(EventSet, values));

        PRINT(quiet, "User measured values.\n");
        for (j = 0; j < numEventsSuccessfullyAdded; j++) {
            PRINT(quiet, "%s\t\t%lld\n", eventsSuccessfullyAdded[j], values[j]);
        }

        // Free allocated memory
        for (i = 0; i < event_count; i++) {
            free(eventsSuccessfullyAdded[i]);
        }
        free(eventsSuccessfullyAdded);

        errno = PAPI_cleanup_eventset(EventSet);
        if (errno != PAPI_OK) {
            fprintf(stderr, "PAPI_cleanup_eventset(%d) failed with error %d", EventSet, errno);
            test_fail(__FILE__, __LINE__, "", errno);
        }
        PAPI_CALL(PAPI_destroy_eventset(&EventSet));
#endif
    }  // omp parallel region end


    if (cudaSuccess != cudaGetLastError())
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));

    omp_destroy_lock(&lock);
#ifdef PAPI
    PAPI_shutdown();

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(quiet, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    test_pass(__FILE__);
#endif
    return 0;
}
