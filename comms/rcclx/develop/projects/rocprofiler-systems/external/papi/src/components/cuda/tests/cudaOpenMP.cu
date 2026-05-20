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
    CUcontext ctx_arr[MAX_THREADS];

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
    }
    int num_threads = (num_gpus > MAX_THREADS) ? MAX_THREADS : num_gpus;
    // Create a gpu context for every thread
    for (i=0; i < num_threads; i++) {
        DRIVER_API_CALL(cuCtxCreate(&(ctx_arr[i]), 0, i % num_gpus));  // "% num_gpus" allows more CPU threads than GPU devices
        DRIVER_API_CALL(cuCtxPopCurrent(&(ctx_arr[i])));
    }

    PRINT(quiet, "---------------------------\n");
#ifdef PAPI
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if ( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num));
#endif

    omp_lock_t lock;
    omp_init_lock(&lock);

    PRINT(quiet, "Launching %d threads.\n", num_threads);
    omp_set_num_threads(num_threads);  // create as many CPU threads as there are CUDA devices
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        PRINT(quiet, "cpu_thread_id %u, num_cpu_threads %u, num_threads %d, num_gpus %d\n", cpu_thread_id, num_cpu_threads, num_threads, num_gpus);

        DRIVER_API_CALL(cuCtxPushCurrent(ctx_arr[cpu_thread_id]));
#ifdef PAPI
        int gpu_id = cpu_thread_id % num_gpus;
        int EventSet = PAPI_NULL;
        long long values[MAX_THREADS];
        int j, errno;
        PAPI_CALL(PAPI_create_eventset(&EventSet));
        PRINT(quiet, "CPU thread %d (of %d) uses CUDA device %d with context %p @ eventset %d\n", cpu_thread_id, num_cpu_threads, gpu_id, ctx_arr[cpu_thread_id], EventSet);
        char tmpEventName[64];
        for (j = 0; j < event_count; j++) {
            snprintf(tmpEventName, 64, "%s:device=%d", argv[j+1], gpu_id);
            PRINT(quiet, "Adding event name %s\n", tmpEventName);
            errno = PAPI_add_named_event( EventSet, tmpEventName );
            if (errno != PAPI_OK) {
                fprintf(stderr, "Error adding event %s\n", tmpEventName);
                test_skip(__FILE__, __LINE__, "", 0);
            }
        }
        PAPI_CALL(PAPI_start(EventSet));
#endif
        VectorAddSubtract(50000*(cpu_thread_id+1), quiet);  // gpu work
#ifdef PAPI
        PAPI_CALL(PAPI_stop(EventSet, values));

        PRINT(quiet, "User measured values.\n");
        for (j = 0; j < event_count; j++) {
            snprintf(tmpEventName, 64, "%s:device=%d", argv[j+1], gpu_id);
            PRINT(quiet, "%s\t\t%lld\n", tmpEventName, values[j]);
        }
        DRIVER_API_CALL(cuCtxPopCurrent(&(ctx_arr[gpu_id])));

        errno = PAPI_cleanup_eventset(EventSet);
        if (errno != PAPI_OK) {
            fprintf(stderr, "PAPI_cleanup_eventset(%d) failed with error %d", EventSet, errno);
            test_fail(__FILE__, __LINE__, "", errno);
        }
        PAPI_CALL(PAPI_destroy_eventset(&EventSet));
#endif
    }  // omp parallel region end

    for (i = 0; i < num_threads; i++) {
        DRIVER_API_CALL(cuCtxDestroy(ctx_arr[i]));
    }

    if (cudaSuccess != cudaGetLastError())
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));

    omp_destroy_lock(&lock);
#ifdef PAPI
    PAPI_shutdown();
    test_pass(__FILE__);
#endif
    return 0;
}
