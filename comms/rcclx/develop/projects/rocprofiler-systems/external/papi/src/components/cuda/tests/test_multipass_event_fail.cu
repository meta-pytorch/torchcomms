/**
 * @file    test_multipass_event_fail.cu
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <stdio.h>

#ifdef PAPI
#include "papi.h"
#include "papi_test.h"

#define PASS 1
#define FAIL 0
#define MAX_EVENT_COUNT (32)
#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}
int quiet;

int test_PAPI_add_named_event(int *EventSet, int numEvents, char **EventName) {
    int i, papi_errno;
    PRINT(quiet, "LOG: %s: Entering.\n", __func__);
    for (i=0; i<numEvents; i++) {
        papi_errno = PAPI_add_named_event(*EventSet, EventName[i]);
        if (papi_errno == PAPI_ENOEVNT) {
            fprintf(stderr, "Event name %s does not exist.\n", EventName[i]);
            return FAIL;
        }
        if (papi_errno != PAPI_OK) {
            PRINT(quiet, "Error %d: Failed to add event %s\n", papi_errno, EventName[i]);
        }
    }
    if (papi_errno == PAPI_EMULPASS)
        return PASS;           // Test pass condition
    return FAIL;
}

int test_PAPI_add_event(int *EventSet, int numEvents, char **EventName) {
    int event, i, papi_errno;
    PRINT(quiet, "LOG: %s: Entering.\n", __func__);

    for (i=0; i<numEvents; i++) {
        papi_errno = PAPI_event_name_to_code(EventName[i], &event);
        if (papi_errno == PAPI_ENOEVNT) {
            fprintf(stderr, "Event name %s does not exist.\n", EventName[i]);
            return FAIL;
        }
        if (papi_errno != PAPI_OK) {
            PRINT(quiet, "Error %d: Error in name to code.\n", papi_errno);
            goto fail;
        }
        papi_errno = PAPI_add_event(*EventSet, event);
        if (papi_errno != PAPI_OK) {
            PRINT(quiet, "Error %d: Failed to add event %s\n", papi_errno, EventName[i]);
        }
    }
    if (papi_errno == PAPI_EMULPASS)
        return PASS;
fail:
    return FAIL;
}

int test_PAPI_add_events(int *EventSet, int numEvents, char **EventName) {
    int papi_errno, i;
    PRINT(quiet, "LOG: %s: Entering.\n", __func__);

    int events[MAX_EVENT_COUNT];

    for (i=0; i<numEvents; i++) {
        papi_errno = PAPI_event_name_to_code(EventName[i], &events[i]);
        if (papi_errno != PAPI_OK) {
            PRINT(quiet, "Error %d: Error in name to code.\n", papi_errno);
            goto fail;
        }
    }
    papi_errno = PAPI_add_events(*EventSet, events, numEvents);
    if (papi_errno != PAPI_OK) {
        PRINT(quiet, "Error %d: Failed to add %d events\n", papi_errno, numEvents);
    }
    if (papi_errno < numEvents)        // Returns index at which error occurred.
        return PASS;
fail:
    return FAIL;
}
#endif

int main(int argc, char **argv)
{
#ifdef PAPI
    int papi_errno, pass;
    int event_set;

    quiet = 0;
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    int event_count = argc - 1;

    /* if no events passed at command line, just report test skipped. */
    if (event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init() failed", 0);
    }

    papi_errno = PAPI_get_component_index("cuda");
    if (papi_errno < 0 ) {
        test_fail(__FILE__, __LINE__, "CUDA component not configured", 0);
    }

    event_set = PAPI_NULL;
    papi_errno = PAPI_create_eventset( &event_set );
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);
    }

    pass = test_PAPI_add_event(&event_set, argc-1, argv+1);
    papi_errno = PAPI_cleanup_eventset(event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset() failed!", 0);
    }

    papi_errno = PAPI_destroy_eventset(&event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset() failed!", 0);
    }

    event_set = PAPI_NULL;
    papi_errno = PAPI_create_eventset( &event_set );
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);
    }

    pass += test_PAPI_add_named_event(&event_set, argc-1, argv+1);
    papi_errno = PAPI_cleanup_eventset(event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset() failed!", 0);
    }

    papi_errno = PAPI_destroy_eventset(&event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset() failed!", 0);
    }

    event_set = PAPI_NULL;
    papi_errno = PAPI_create_eventset( &event_set );
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);
    }

    pass += test_PAPI_add_events(&event_set, argc-1, argv+1);
    papi_errno = PAPI_cleanup_eventset(event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset() failed!", 0);
    }

    papi_errno = PAPI_destroy_eventset(&event_set);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset() failed!", 0);
    }

    if (pass != 3)
        test_fail(__FILE__, __LINE__, "CUDA framework multipass event test failed.", 0);
    else
        test_pass(__FILE__);

    PAPI_shutdown();
#else
    fprintf(stderr, "Please compile with -DPAPI to test this feature.\n");
#endif
    return 0;
}