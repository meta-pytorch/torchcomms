/**
 * @file   multi_thread.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 * Multi thread monitoring test. This test launches kernels on
 * each of the available GPUs using separate omp threads. Each
 * thread registers with PAPI and does its own monitoring through
 * PAPI.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "common.h"

#define PASS     0x0
#define PASSWARN 0x1
#define FAIL     0x2
#define HIPFAIL  0x4

int *testLINE = NULL;
int *status = NULL;
char **errMSG = NULL;
int *papi_errno = NULL;
hipError_t *hip_errno = NULL;

int quiet;

static void log_error(int testLINE_arg, int status_arg, const char *errMSG_arg, int tid) {

    testLINE[tid] = testLINE_arg;
    status[tid] = status_arg;

    int ret = snprintf(errMSG[tid], PAPI_MAX_STR_LEN, "%s", errMSG_arg);
    if ( ret < 0 || ret >= PAPI_MAX_STR_LEN ) {
        fprintf(stdout, "[%s, %d] WARNING: Could not copy string %s into buffer.\n", __FILE__, __LINE__, errMSG_arg);
    }

    return;
}

static void *run(void *thread_num_arg)
{
    int j;

#define NUM_EVENTS 4
    const char *events[NUM_EVENTS] = {
        "rocm:::SQ_INSTS_VALU",
        "rocm:::SQ_INSTS_SALU",
        "rocm:::SQ_WAVES",
        "rocm:::SQ_WAVES_RESTORED",
    };

    int eventset = PAPI_NULL;
    int thread_num = *(int *) thread_num_arg;

    /* Initialize global variables. */
    testLINE[thread_num] = 0;
    status[thread_num] = PASS;
    papi_errno[thread_num] = PAPI_OK;
    hip_errno[thread_num] = hipSuccess;

    papi_errno[thread_num] = PAPI_create_eventset(&eventset);
    if (papi_errno[thread_num] != PAPI_OK) {
        log_error(__LINE__, FAIL, "PAPI_create_eventset", thread_num);
        pthread_exit(NULL);
    }

    for (int j = 0; j < NUM_EVENTS; ++j) {
        char named_event[PAPI_MAX_STR_LEN];
        sprintf(named_event, "%s:device=%d", events[j], thread_num);
        papi_errno[thread_num] = PAPI_add_named_event(eventset, (const char*) named_event);
        if (papi_errno[thread_num] != PAPI_OK && papi_errno[thread_num] != PAPI_ENOEVNT) {
            log_error(__LINE__, FAIL, "PAPI_add_named_event", thread_num);
            pthread_exit(NULL);
        } else if (papi_errno[thread_num] == PAPI_ENOEVNT) {
            status[thread_num] = PASSWARN;
        }
    }

    papi_errno[thread_num] = PAPI_start(eventset);
    if (papi_errno[thread_num] != PAPI_OK) {
        log_error(__LINE__, FAIL, "PAPI_start", thread_num);
        pthread_exit(NULL);
    }

    hipStream_t stream;
    hip_errno[thread_num] = hipSetDevice(thread_num);
    if (hip_errno[thread_num] != hipSuccess) {
        log_error(__LINE__, HIPFAIL, "hipSetDevice", thread_num);
        pthread_exit(NULL);
    }

    hip_errno[thread_num] = hipStreamCreate(&stream);
    if (hip_errno[thread_num] != hipSuccess) {
        log_error(__LINE__, HIPFAIL, "hipStreamCreate", thread_num);
        pthread_exit(NULL);
    }

    void *handle;
    hip_do_matmul_init(&handle);

    hip_do_matmul_work(handle, stream);

    hip_errno[thread_num] = hipStreamSynchronize(stream);
    if (hip_errno[thread_num] != hipSuccess) {
        log_error(__LINE__, HIPFAIL, "hipStreamSynchronize", thread_num);
        pthread_exit(NULL);
    }

    hip_errno[thread_num] = hipStreamDestroy(stream);
    if (hip_errno[thread_num] != hipSuccess) {
        log_error(__LINE__, HIPFAIL, "hipStreamDestroy", thread_num);
        pthread_exit(NULL);
    }

    hip_do_matmul_cleanup(&handle);

    long long counters[NUM_EVENTS] = { 0 };
    papi_errno[thread_num] = PAPI_stop(eventset, counters);
    if (papi_errno[thread_num] != PAPI_OK) {
        log_error(__LINE__, FAIL, "PAPI_stop", thread_num);
        pthread_exit(NULL);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        if (!quiet) {
            fprintf(stdout, "[tid:%lu] %s:device=%d : %lld\n",
                    pthread_self(), events[i], thread_num,
                    counters[i]);
        }
    }

    papi_errno[thread_num] = PAPI_cleanup_eventset(eventset);
    if (papi_errno[thread_num] != PAPI_OK) {
        log_error(__LINE__, FAIL, "PAPI_cleanup_eventset", thread_num);
        pthread_exit(NULL);
    }

    papi_errno[thread_num] = PAPI_destroy_eventset(&eventset);
    if (papi_errno[thread_num] != PAPI_OK) {
        log_error(__LINE__, FAIL, "PAPI_destroy_eventset", thread_num);
        pthread_exit(NULL);
    }

    /* Query only device 0 and assume all devices are identical */
    int warp_size;
    hip_errno[thread_num] = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);
    if (hip_errno[thread_num] != hipSuccess) {
        log_error(__LINE__, HIPFAIL, "hipDeviceGetAttribute", thread_num);
        pthread_exit(NULL);
    }

    /* compute expected number of waves need to multiply two square matrices of ROWS x COLS elements */
    long long expected_waves = (long long) ((ROWS * COLS) / warp_size);

    if (match_expected_counter(expected_waves, counters[2] - counters[3]) != 1) {
        if (PASSWARN == status[thread_num]) {
            papi_errno[thread_num] = 1;
            log_error(__LINE__, PASSWARN, "match_expected_counter", thread_num);
            pthread_exit(NULL);
        } else {
            papi_errno[thread_num] = -1;
            log_error(__LINE__, FAIL, "match_expected_counter", thread_num);
            pthread_exit(NULL);
        }
    }

    pthread_exit(NULL);
}

int multi_thread(int argc, char *argv[])
{
    int papi_errno_main;
    int retcode;
    hipError_t hip_errno_main;
    quiet = tests_quiet(argc, argv);

    if (!quiet) {
        fprintf(stdout, "%s : multi GPU activity monitoring program.\n",
                argv[0]);
    }

    papi_errno_main = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno_main != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno_main);
    }

    papi_errno_main = PAPI_thread_init((unsigned long (*)(void)) pthread_self);
    if (papi_errno_main != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_thread_init", papi_errno_main);
    }

    int dev_count;
    /* The first hip call causes the hsa runtime to be initialized
     * too (by calling hsa_init()). If hsa is already initialized
     * this will result in the increment of an internal reference
     * counter and won't alter the current configuration. */
    hip_errno_main = hipGetDeviceCount(&dev_count);
    if (hip_errno_main != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipGetDeviceCount", hip_errno_main);
    }

    pthread_t *thread = (pthread_t *) malloc(dev_count * sizeof(*thread));
    if (NULL == thread) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    int *thread_num = (int *) malloc(dev_count * sizeof(*thread_num));
    if (NULL == thread_num) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    /* Allocate memory for global variables. */
    testLINE   = (int*)malloc(dev_count*sizeof(int));
    if (NULL == testLINE) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    status     = (int*)malloc(dev_count*sizeof(int));
    if (NULL == status) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    errMSG   = (char**)malloc(dev_count*sizeof(char*));
    if (NULL == errMSG) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }
    for (int i = 0; i < dev_count; ++i) {
        errMSG[i] = (char*)malloc(PAPI_MAX_STR_LEN*sizeof(char));
        if (NULL == errMSG[i]) {
            test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
        }
    }

    papi_errno = (int*)malloc(dev_count*sizeof(int));
    if (NULL == papi_errno) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    hip_errno  = (hipError_t*)malloc(dev_count*sizeof(hipError_t));
    if (NULL == hip_errno) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ENOMEM);
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < dev_count; ++i) {
        thread_num[i] = i;
        pthread_create(&thread[i], &attr, run, &thread_num[i]);
    }

    for (int i = 0; i < dev_count; ++i) {
        pthread_join(thread[i], NULL);
    }

    int status_main = PASS;
    int tid;
    for ( tid = 0; tid < dev_count; ++tid) {
        if (PASS != status[tid]) {
            status_main = status[tid];
            break;
        }
    }

    switch ( status_main ) {
        case PASSWARN:
            test_warn(__FILE__, testLINE[tid], errMSG[tid], papi_errno[tid]);
            break;
        case FAIL:
            test_fail(__FILE__, testLINE[tid], errMSG[tid], papi_errno[tid]);
            break;
        case HIPFAIL:
            hip_test_fail(__FILE__, testLINE[tid], errMSG[tid], hip_errno[tid]);
            break;
        default: // PASS
            break;
    }

    /* Free dynamically allocated memory. */
    free(testLINE);
    free(status);
    free(papi_errno);
    free(hip_errno);
    for (int i = 0; i < dev_count; ++i) {
        free(errMSG[i]);
    }
    free(errMSG);

    free(thread);
    free(thread_num);

    PAPI_shutdown();

    return 0;
}
