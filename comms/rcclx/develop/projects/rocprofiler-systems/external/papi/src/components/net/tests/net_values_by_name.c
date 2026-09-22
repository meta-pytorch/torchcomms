/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @brief
 *   For each net event that is available add it to an EventSet by its name 
 *   e.g. net:::lo:rx:byte. Then run through a start - stop workflow.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define PINGADDR   "127.0.0.1"

int main (int argc, char **argv)
{
    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /* PAPI Initialization */
    int retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
    }

    const char *componentName = "net";
    int cmpIdx = PAPI_get_component_index(componentName);
    if (cmpIdx < 0) {
        test_fail(__FILE__, __LINE__,"PAPI_get_component_index", cmpIdx);
    }

    int eventCode = 0 | PAPI_NATIVE_MASK;
    int modifier = PAPI_ENUM_FIRST;
    retval = PAPI_enum_cmp_event(&eventCode, modifier, cmpIdx);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", retval);
    }

    int EventSet = PAPI_NULL;
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
    }  

    int numEventsAdded = 0;
    char **eventNames = NULL;
    modifier = PAPI_ENUM_EVENTS;
    do {
        // Get an events info to use the symbol member variable
        PAPI_event_info_t evtInfo;
        retval = PAPI_get_event_info(eventCode, &evtInfo);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_get_event_info", retval);
        }

        retval = PAPI_add_named_event(EventSet, evtInfo.symbol);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", retval);
        }

        // Allocate necessary memory to store successfully added events
        eventNames = (char **) realloc(eventNames, (numEventsAdded + 1) * sizeof(char *));
        if (eventNames == NULL) {
            fprintf(stderr, "Failed to allocate memory for the array eventNames.\n");
            exit(1);
        }
        eventNames[numEventsAdded] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        if (eventNames[numEventsAdded] == NULL) {
            fprintf(stderr, "Failed to allocate memory for index %d of array eventNames.\n", numEventsAdded);
            exit(1);
        }

        // Store successfully added events to be output after start - stop workflow
        int strLen = snprintf(eventNames[numEventsAdded], PAPI_MAX_STR_LEN, "%s", evtInfo.symbol);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write the event %s to index %d.\n", evtInfo.symbol, numEventsAdded);
            exit(1);
        }

        // Incremenent total number of events successfully added
        numEventsAdded++;
    } while(PAPI_enum_cmp_event(&eventCode, modifier, cmpIdx) == PAPI_OK);

    retval = PAPI_start( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", retval);
    }

    /* generate some traffic
     * the operation should take more than one second in order
     * to guarantee that the network counters are updated */
    retval = system("ping -c 4 " PINGADDR " > /dev/null");
    if (retval < 0) {
        fprintf(stderr, "Unable to start ping.\n");
        exit(1);
	}

    long long *counterValues = (long long *) malloc(numEventsAdded * sizeof(long long));
    if (counterValues == NULL) {
        fprintf(stderr, "Failed to allocate memory for array counterValues.\n");
        exit(1);
    }

    retval = PAPI_stop( EventSet, counterValues );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
    }

    int i;
    if (!TESTS_QUIET) {
        printf("Net events by name:\n");
        for (i = 0; i < numEventsAdded; i++) {
            printf("%s has a counter value of %d\n", eventNames[i], counterValues[i]);
        }
    }

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
    }

    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);
    }

    // Free allocated memory
    for (i = 0; i < numEventsAdded; i++) {
        free(eventNames[i]);
    }
    free(eventNames);
    free(counterValues);

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
