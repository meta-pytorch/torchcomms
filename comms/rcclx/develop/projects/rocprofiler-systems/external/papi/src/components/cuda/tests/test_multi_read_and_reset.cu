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

// Globals for successfully added and multiple pass events
int numEventsSuccessfullyAdded = 0, numMultipassEvents = 0;

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
        int strLen;
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
        strLen = snprintf(eventsSuccessfullyAdded[(*numEventsSuccessfullyAdded)], PAPI_MAX_STR_LEN, "%s", eventNamesFromCommandLine[i]);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write successfully added event.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
        (*numEventsSuccessfullyAdded)++;
    }

    return;
}

void multi_reset(int event_count, char **evt_names, long long *values)
{
    CUcontext ctx;
    int papi_errno, i;

    CUresult cuError;
    int flags = 0;
    CUdevice device = 0;
#if defined(CUDA_TOOLKIT_GE_13)
    cuError = cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version >= 13: %d\n", cuError);
        exit(1);
    }
#else
    cuError = cuCtxCreate(&ctx, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version < 13: %d\n", cuError);
        exit(1);
    }
#endif

#ifdef PAPI
    int EventSet = PAPI_NULL;
    int j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }

    // Handle the events from the command line
    numEventsSuccessfullyAdded = 0;
    numMultipassEvents = 0;
    char **eventsSuccessfullyAdded;
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
    add_events_from_command_line(EventSet, event_count, evt_names, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

    // Only multiple pass events were provided on the command line
    if (numEventsSuccessfullyAdded == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
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
        for (j=0; j < numEventsSuccessfullyAdded; j++) {
            PRINT(quiet, "%s\t\t%lld\n", eventsSuccessfullyAdded[j], values[j]);
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

    // Free allocated memory
    for (i = 0; i < event_count; i++) {
        free(eventsSuccessfullyAdded[i]);
    }
    free(eventsSuccessfullyAdded);
}

void multi_read(int event_count, char **evt_names, long long *values)
{
    CUcontext ctx;
    int papi_errno, i;

    CUresult cuError;
    int flags = 0;
    CUdevice device = 0;
#if defined(CUDA_TOOLKIT_GE_13)
    cuError = cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version >= 13: %d\n", cuError);
        exit(1);
    }
#else
    cuError = cuCtxCreate(&ctx, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version < 13: %d\n", cuError);
        exit(1);
    }
#endif

#ifdef PAPI
    int EventSet = PAPI_NULL, j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }

    // Handle the events from the command line
    numEventsSuccessfullyAdded = 0;
    numMultipassEvents = 0;
    char **eventsSuccessfullyAdded;
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
    add_events_from_command_line(EventSet, event_count, evt_names, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

    // Only multiple pass events were provided on the command line
    if (numEventsSuccessfullyAdded == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
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
        for (j=0; j < numEventsSuccessfullyAdded; j++) {
            PRINT(quiet, "%s\t\t%lld\n", eventsSuccessfullyAdded[j], values[j]);
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

    // Free allocated memory
    for (i = 0; i < event_count; i++) {
        free(eventsSuccessfullyAdded[i]);
    }
    free(eventsSuccessfullyAdded);
}

void single_read(int event_count, char **evt_names, long long *values, char ***addedEvents)
{
    int papi_errno, i;
    CUcontext ctx;

    CUresult cuError;
    int flags = 0;
    CUdevice device = 0;
#if defined(CUDA_TOOLKIT_GE_13)
    cuError = cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version >= 13: %d\n", cuError);
        exit(1);
    }
#else
    cuError = cuCtxCreate(&ctx, flags, device);
    if (cuError != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create Cuda context for a Cuda Toolkit version < 13: %d\n", cuError);
        exit(1);
    }
#endif

#ifdef PAPI
    int EventSet = PAPI_NULL, j;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "Failed to create eventset.", papi_errno);
    }

    // Handle the events from the command line
    numEventsSuccessfullyAdded = 0;
    numMultipassEvents = 0;
    char **eventsSuccessfullyAdded;
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
    add_events_from_command_line(EventSet, event_count, evt_names, &numEventsSuccessfullyAdded, eventsSuccessfullyAdded, &numMultipassEvents);

    // Only multiple pass events were provided on the command line
    if (numEventsSuccessfullyAdded == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
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
    for (j=0; j < numEventsSuccessfullyAdded; j++) {
        PRINT(quiet, "%s\t\t%lld\n", eventsSuccessfullyAdded[j], values[j]);
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

    *addedEvents = eventsSuccessfullyAdded;
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
    char **eventsSuccessfullyAdded = { 0 };
    single_read(event_count, argv + 1, values_single_read, &eventsSuccessfullyAdded);

    int i;
    PRINT(quiet, "Final measured values\nEvent_name\t\t\t\t\t\tMulti_read\tsingle_read\n");
    for (i=0; i < numEventsSuccessfullyAdded; i++) {
        PRINT(quiet, "%s\t\t\t%lld\t\t%lld\n", eventsSuccessfullyAdded[i], values_multi_read[i], values_single_read[i]);
        if ( !approx_equal(values_multi_read[i], values_single_read[i]) )
            test_warn(__FILE__, __LINE__, "Measured values from multi read and single read don't match.", PAPI_OK);
    }

    // Free allocated memory
    for (i = 0; i < event_count; i++) {
        free(eventsSuccessfullyAdded[i]);
    }
    free(eventsSuccessfullyAdded);

    PAPI_shutdown();

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(quiet, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    test_pass(__FILE__);
#else
    fprintf(stderr, "Please compile with -DPAPI to test this feature.\n");
#endif
    return 0;
}
