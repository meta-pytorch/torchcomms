/**
 * @file   sample_overflow_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 * Credit to:
 *         Anthony Danalis <adanalis@icl.utk.edu>
 *         Who originally wrote the test used
 *         here as template for the SDE component.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#define EV_THRESHOLD 1000

void setup_PAPI(int *event_set, int threshold);
void unset_PAPI(int event_set);

int remaining_handler_invocations = 10;
int quiet;

int main(int argc, char **argv)
{
    int papi_errno;
    hipError_t hip_errno;
    int event_set = PAPI_NULL;
    long long counter_values[1] = { 0 };
    quiet = tests_quiet(argc, argv);

    setenv("ROCP_HSA_INTERCEPT", "0", 1);

    setup_PAPI(&event_set, EV_THRESHOLD);

    void *handler;
    hip_do_matmul_init(&handler);

    hipStream_t stream;
    hip_errno = hipStreamCreate(&stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamCreate", hip_errno);
    }

    // --- Start PAPI
    papi_errno = PAPI_start(event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    int i;
    for (i = 0; i < 5; i++) {

        hip_do_matmul_work(handler, stream);

        // --- Read the event counters _and_ reset them
        papi_errno = PAPI_accum(event_set, counter_values);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_accum", papi_errno);
        }
        if (!quiet) {
            fprintf(stdout, "rocm:::SQ_WAVES:device=0 : %lld\n",
                    counter_values[0]);
            fflush(stdout);
        }

        counter_values[0] = 0;

        hip_do_matmul_work(handler, 0);

        // --- Read the event counters _and_ reset them
        papi_errno = PAPI_accum(event_set, counter_values);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_accum", papi_errno);
        }
        if (!quiet) {
            fprintf(stdout, "rocm:::SQ_WAVES:device=0 : %lld\n",
                    counter_values[0]);
            fflush(stdout);
        }
        counter_values[0] = 0;
    }

    // --- Stop PAPI
    papi_errno = PAPI_stop(event_set, counter_values);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    unset_PAPI(event_set);
    hip_do_matmul_cleanup(&handler);
    hip_errno = hipStreamDestroy(stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamDestroy", hip_errno);
    }

    if (remaining_handler_invocations <= 1) { // Let's allow for up to one missed signal, or race condition.
        test_pass(__FILE__);
    } else {
        test_fail(__FILE__, __LINE__,
                  "ROCm overflow handler was not invoked as expected!", 0);
    }

    // The following "return" is dead code, because both test_pass() and test_fail() call exit(),
    // however, we need it to prevent compiler warnings.
    return 0;
}

void overflow_handler(int event_set, void *address __attribute__((unused)),
                      long long overflow_vector,
                      void *context __attribute__((unused)))
{
    int papi_errno;
    char event_name[PAPI_MAX_STR_LEN];
    int *event_codes, event_index, number=1;

    papi_errno = PAPI_get_overflow_event_index(event_set, overflow_vector,
                                               &event_index, &number);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_get_overflow_event_index", papi_errno);
    }

    number = event_index + 1;
    event_codes = (int *) calloc(number, sizeof(int));

    papi_errno = PAPI_list_events(event_set, event_codes, &number);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_list_events", papi_errno);
    }

    papi_errno = PAPI_event_code_to_name(event_codes[event_index], event_name);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", papi_errno);
    }

    free(event_codes);

    if (!quiet) {
        fprintf(stdout, "Event \"%s\" at index: %d exceeded its threshold again.\n",
                event_name, event_index);
        fflush(stdout);
    }

    if (!strcmp(event_name, "rocm:::SQ_WAVES:device=0") || !event_index) {
        remaining_handler_invocations--;
    }

    return;
}

void setup_PAPI(int *event_set, int threshold)
{
    int papi_errno;
    int event_code;

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    papi_errno = PAPI_create_eventset(event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    papi_errno = PAPI_event_name_to_code("rocm:::SQ_WAVES:device=0", &event_code);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", papi_errno);
    }

    papi_errno = PAPI_add_event(*event_set, event_code);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_add_event", papi_errno);
    }

    papi_errno = PAPI_overflow(*event_set, event_code, threshold,
                               PAPI_OVERFLOW_FORCE_SW, overflow_handler);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_overflow", papi_errno);
    }
    return;
}

void unset_PAPI(int event_set)
{
    int papi_errno;
    int event_code;

    papi_errno = PAPI_event_name_to_code("rocm:::SQ_WAVES:device=0", &event_code);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", papi_errno);
    }

    papi_errno = PAPI_overflow(event_set, event_code, 0, PAPI_OVERFLOW_FORCE_SW,
                               overflow_handler);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_overflow", papi_errno);
    }
    return;
}
