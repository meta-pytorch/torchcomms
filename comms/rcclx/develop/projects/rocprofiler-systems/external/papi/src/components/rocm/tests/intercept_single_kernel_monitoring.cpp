/**
 * @file   intercept_single_kernel_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 * Single GPU activity monitoring test. This test launches a kernel
 * two times with different event sets. The goal of this test is to
 * fail at the second launch if the ROCm component is operating in
 * intercept mode (Refer to ROC profiler intercept mode).
 */
#include "common.h"

int quiet;

int main(int argc, char *argv[])
{
    int papi_errno;
    int pass_with_warning = 0;
    hipError_t hip_errno;
    quiet = tests_quiet(argc, argv);

    setenv("ROCP_HSA_INTERCEPT", "1", 1);

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

#define NUM_EVENTS (4)
    const char *events[NUM_EVENTS] = {
        "rocm:::SQ_INSTS_VALU",
        "rocm:::SQ_INSTS_SALU",
        "rocm:::SQ_WAVES",
        "rocm:::SQ_WAVES_RESTORED",
    };

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        char named_event[PAPI_MAX_STR_LEN] = { 0 };
        sprintf(named_event, "%s:device=0", events[i]);
        papi_errno = PAPI_add_named_event(eventset, named_event);
        if (papi_errno != PAPI_OK && papi_errno != PAPI_ENOEVNT) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        } else if (papi_errno == PAPI_ENOEVNT) {
            pass_with_warning = 1;
        }
    }

    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    hipStream_t stream;
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

    long long counters[NUM_EVENTS] = { 0 };

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        if (!quiet) {
            fprintf(stdout, "%s:device=0 : %lld\n", events[i], counters[i]);
        }
    }

    /* now remove the all events and restart the kernel monitoring.
     * when using intercept mode the following code should cause
     * papi to return PAPI_ECNFLCT error, indicating the eventset
     * cannot be altered. this is a limitation of the rocprofiler
     * library (ver 4.5.0 at the time of writing) */
    papi_errno = PAPI_cleanup_eventset(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

#define NUM_NEW_EVENTS (2)
    const char *new_events[NUM_NEW_EVENTS] = {
        "rocm:::SQ_INSTS_VMEM_RD",
        "rocm:::SQ_INSTS_VMEM_WR",
    };

    for (int i = 0; i < NUM_NEW_EVENTS; ++i) {
        char named_event[PAPI_MAX_STR_LEN] = { 0 };
        sprintf(named_event, "%s:device=0", new_events[i]);
        papi_errno = PAPI_add_named_event(eventset, named_event);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
    }

#if 0
    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

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

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    for (int i = 0; i < NUM_NEW_EVENTS; ++i) {
        if (!quiet) {
            fprintf(stdout, "%s:device=0 : %lld\n", new_events[i], counters[i]);
        }
    }
#endif

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

    if (match_expected_counter(expected_waves, counters[2] - counters[3]) != 1) {
        if (pass_with_warning) {
            test_warn(__FILE__, __LINE__, "match_expected_counter", 1);
        } else {
            test_fail(__FILE__, __LINE__, "match_expected_counter", -1);
        }
    }

    PAPI_shutdown();

    test_pass(__FILE__);
    return 0;
}
