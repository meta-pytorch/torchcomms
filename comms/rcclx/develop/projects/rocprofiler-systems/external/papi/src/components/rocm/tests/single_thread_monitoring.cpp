/**
 * @file   single_thread.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 * Single thread monitoring test. This test launches kernels on
 * each of the available GPUs using separate omp threads. The main
 * thread does the PAPI monitoring for all GPUs. In both sample
 * and intercept mode any work launched by the spawned threads
 * is associated to the main one. Thus, PAPI will associate the
 * counters to the main thread's eventset.
 */
#include "single_thread_monitoring.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int quiet;

static void *run(void *thread_num_arg)
{
    int thread_num = *(int *) thread_num_arg;

    hipStream_t stream;
    hipError_t hip_errno = hipSetDevice(thread_num);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipSetDevice", hip_errno);
    }

    hip_errno = hipStreamCreate(&stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamCreate", hip_errno);
    }

    void *handle;
    hip_do_matmul_init(&handle);
    hip_do_matmul_work(handle, stream);

    hip_errno = hipStreamSynchronize(stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamSynchronize", hip_errno);
    }

    hip_errno = hipStreamDestroy(stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamDestroy", hip_errno);
    }
    hip_do_matmul_cleanup(&handle);

    pthread_exit(NULL);
}


int single_thread(int argc, char *argv[])
{
    int papi_errno;
    int pass_with_warning = 0;
    hipError_t hip_errno;
    quiet = tests_quiet(argc, argv);
    int retcode;

    if (!quiet) {
        fprintf(stdout, "%s : multi GPU activity monitoring program.\n",
                argv[0]);
    }

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

#define NUM_EVENTS 4
    const char *events[NUM_EVENTS] = {
        "rocm:::SQ_INSTS_VALU",
        "rocm:::SQ_INSTS_SALU",
        "rocm:::SQ_WAVES",
        "rocm:::SQ_WAVES_RESTORED",
    };

    int dev_count;
    /* The first hip call causes the hsa runtime to be initialized
     * too (by calling hsa_init()). If hsa is already initialized
     * this will result in the increment of an internal reference
     * counter and won't alter the current configuration. */
    hip_errno = hipGetDeviceCount(&dev_count);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipGetDeviceCount", hip_errno);
    }

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }
    for (int i = 0; i < dev_count; ++i) {
        for (int j = 0; j < NUM_EVENTS; ++j) {
            char named_event[PAPI_MAX_STR_LEN];
            sprintf(named_event, "%s:device=%d", events[j], i);
            papi_errno = PAPI_add_named_event(eventset,
                                              (const char *) named_event);
            if (papi_errno != PAPI_OK && papi_errno != PAPI_ENOEVNT) {
                test_fail(__FILE__, __LINE__, "PAPI_add_named_event",
                          papi_errno);
            } else if (papi_errno == PAPI_ENOEVNT) {
                pass_with_warning = 1;
            }
        }
    }

#define MAX_DEV_COUNT (16)
    long long counters[NUM_EVENTS * MAX_DEV_COUNT] = { 0 };

    pthread_t *thread = (pthread_t *) malloc(dev_count * sizeof(*thread));
    if (NULL == thread) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    int *thread_num = (int *) malloc(dev_count * sizeof(*thread_num));
    if (NULL == thread_num) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    for (int i = 0; i < dev_count; ++i) {
        thread_num[i] = i;
        pthread_create(&thread[i], &attr, run, &thread_num[i]);
    }

    for (int i = 0; i < dev_count; ++i) {
        pthread_join(thread[i], NULL);
    }

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    free(thread);
    free(thread_num);

    for (int i = 0; i < dev_count; ++i) {
        for (int j = 0; j < NUM_EVENTS; ++j) {
            if (!quiet) {
                fprintf(stdout, "%s:device=%d : %lld\n", events[j], i,
                        counters[i * NUM_EVENTS + j]);
            }
        }
    }

    papi_errno = PAPI_cleanup_eventset(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    /* Query only device 0 and assume all devices are identical */
    int warp_size;
    hip_errno = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);
    if (hip_errno != hipSuccess) {
        test_fail(__FILE__, __LINE__, "hipDeviceGetAttribute", hip_errno);
    }

    /* compute expected number of waves need to multiply two square matrices of ROWS x COLS elements */
    long long expected_waves = (long long) ((ROWS * COLS) / warp_size);

    for (int i = 0; i < dev_count; ++i) {
        if (match_expected_counter(expected_waves,
                                   counters[i * NUM_EVENTS + 2] - counters[i * NUM_EVENTS + 3]) != 1) {
            if (pass_with_warning) {
                test_warn(__FILE__, __LINE__, "match_expected_counter", 1);
            } else {
                test_fail(__FILE__, __LINE__, "match_expected_counter", -1);
            }
        }
    }
    test_pass(__FILE__);

    PAPI_shutdown();

    return 0;
}
