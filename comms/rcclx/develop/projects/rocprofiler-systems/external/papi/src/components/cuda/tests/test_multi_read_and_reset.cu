/**
 * @file    test_multi_read_and_reset.cu
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <stdio.h>
#include "gpu_work.h"

#ifdef PAPI
#include <papi.h>
#include "papi_test.h"
#endif

#define COMP_NAME "cuda"
#define MAX_EVENT_COUNT (32)
#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}
int quiet;

int approx_equal(long long v1, long long v2)
{
    double err = fabs(v1 - v2) / v1;
    if (err < 0.1)
        return 1;
    return 0;
}

void multi_reset(int event_count, char **evt_names, long long *values)
{
    CUcontext ctx;
    int papi_errno, i;
    papi_errno = cuCtxCreate(&ctx, 0, 0);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
        exit(1);
    }

#ifdef PAPI
    int EventSet = PAPI_NULL;
    int j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }

    for (i=0; i < event_count; i++) {
        papi_errno = PAPI_add_named_event(EventSet, evt_names[i]);
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", evt_names[i]);
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    papi_errno = PAPI_start(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start error.", papi_errno);
    }
#endif

    for (i=0; i<10; i++) {
        VectorAddSubtract(100000, quiet);
#ifdef PAPI
        papi_errno = PAPI_read(EventSet, values);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read error.", papi_errno);
        }
        PRINT(quiet, "Measured values iter %d\n", i);
        for (j=0; j < event_count; j++) {
            PRINT(quiet, "%s\t\t%lld\n", evt_names[j], values[j]);
        }
        papi_errno = PAPI_reset(EventSet);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_reset error.", papi_errno);
        }
#endif
    }
#ifdef PAPI
    papi_errno = PAPI_stop(EventSet, values);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop error.", papi_errno);
    }

    papi_errno = PAPI_cleanup_eventset(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset error.", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset error.", papi_errno);
    }
#endif
    papi_errno = cuCtxDestroy(ctx);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cude error: failed to destroy context.\n");
        exit(1);
    }
}

void multi_read(int event_count, char **evt_names, long long *values)
{
    CUcontext ctx;
    int papi_errno, i;
    papi_errno = cuCtxCreate(&ctx, 0, 0);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
        exit(1);
    }

#ifdef PAPI
    int EventSet = PAPI_NULL, j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }

    for (i=0; i < event_count; i++) {
        papi_errno = PAPI_add_named_event(EventSet, evt_names[i]);
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", evt_names[i]);
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    papi_errno = PAPI_start(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start error.", papi_errno);
    }
#endif
    for (i=0; i<10; i++) {
        VectorAddSubtract(100000, quiet);
#ifdef PAPI
        papi_errno = PAPI_read(EventSet, values);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_start error.", papi_errno);
        }
        PRINT(quiet, "Measured values iter %d\n", i);
        for (j=0; j < event_count; j++) {
            PRINT(quiet, "%s\t\t%lld\n", evt_names[j], values[j]);
        }
    }
    papi_errno = PAPI_stop(EventSet, values);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop error.", papi_errno);
    }
    papi_errno = PAPI_cleanup_eventset(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset error.", papi_errno);
    }
    papi_errno = PAPI_destroy_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset error.", papi_errno);
#endif
    }

    papi_errno = cuCtxDestroy(ctx);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cude error: failed to destroy context.\n");
        exit(1);
    }
}

void single_read(int event_count, char **evt_names, long long *values)
{
    int papi_errno, i;
    CUcontext ctx;
    papi_errno = cuCtxCreate(&ctx, 0, 0);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
        exit(1);
    }
#ifdef PAPI
    int EventSet = PAPI_NULL, j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }
    for (i=0; i < event_count; i++) {
        papi_errno = PAPI_add_named_event(EventSet, evt_names[i]);
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", evt_names[i]);
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    papi_errno = PAPI_start(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start error.", papi_errno);
    }
#endif
    for (i=0; i<10; i++) {
        VectorAddSubtract(100000, quiet);
    }
#ifdef PAPI
    papi_errno = PAPI_stop(EventSet, values);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop error.", papi_errno);
    }
    PRINT(quiet, "Measured values from single read\n");
    for (j=0; j < event_count; j++) {
        PRINT(quiet, "%s\t\t%lld\n", evt_names[j], values[j]);
    }
    papi_errno = PAPI_cleanup_eventset(EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset error.", papi_errno);
    }
    papi_errno = PAPI_destroy_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset error.", papi_errno);
    }
#endif
    papi_errno = cuCtxDestroy(ctx);
    if (papi_errno != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to destroy cuda context.\n");
        exit(1);
    }
}

int main(int argc, char **argv)
{
    cuInit(0);

    quiet = 0;
#ifdef PAPI
    int papi_errno;
	char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    int event_count = argc - 1;

    /* if no events passed at command line, just report test skipped. */
    if (event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "Failed to initialize PAPI.", 0);
    }
    papi_errno = PAPI_get_component_index(COMP_NAME);
    if (papi_errno < 0) {
        test_fail(__FILE__, __LINE__, "Failed to get index of cuda component.", PAPI_ECMP);
    }
    long long values_multi_reset[MAX_EVENT_COUNT];
    long long values_multi_read[MAX_EVENT_COUNT];
    long long values_single_read[MAX_EVENT_COUNT];

    PRINT(quiet, "Running multi_reset.\n");
    multi_reset(event_count, argv + 1, values_multi_reset);
    PRINT(quiet, "\nRunning multi_read.\n");
    multi_read(event_count, argv + 1, values_multi_read);
    PRINT(quiet, "\nRunning single_read.\n");
    single_read(event_count, argv + 1, values_single_read);

    int i;
    PRINT(quiet, "Final measured values\nEvent_name\t\t\t\t\t\tMulti_read\tsingle_read\n");
    for (i=0; i < event_count; i++) {
        PRINT(quiet, "%s\t\t\t%lld\t\t%lld\n", argv[i+1], values_multi_read[i], values_single_read[i]);
        if ( !approx_equal(values_multi_read[i], values_single_read[i]) )
            test_warn(__FILE__, __LINE__, "Measured values from multi read and single read don't match.", PAPI_OK);
    }
    PAPI_shutdown();
    test_pass(__FILE__);
#else
    fprintf(stderr, "Please compile with -DPAPI to test this feature.\n");
#endif
    return 0;
}
