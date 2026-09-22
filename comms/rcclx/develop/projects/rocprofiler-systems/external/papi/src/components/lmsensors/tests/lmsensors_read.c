/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Treece Burgess (tburgess@icl.utk.edu)
 *
 * Test case for the lmsensors component.
 * For GitHub CI and terminal use.
 *
 * Tested on Leconte at ICL in winter 2024 with an 
 * Intel(R) Xeon(R) CPU E5-2698.
 *
 * @brief
 *   Creating a PAPI EventSet with lmsensor events
 *   to add, start, read, and stop. 
 */

#include <stdio.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

/* maximum number of events we want to add to the PAPI EventSet */
#define NUM_EVENTS 3

int main(int argc, char **argv)
{
    int retval, i, event_cnt = 0, cidx;
    int EventCode, EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    char EventName[PAPI_MAX_STR_LEN], lm_events[NUM_EVENTS][PAPI_MAX_STR_LEN];

    /* determine if we quiet output */
    tests_quiet(argc, argv);

    /* initialize the PAPI library */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
    }

    /* get the lmsensors component index */
    cidx = PAPI_get_component_index("lmsensors");
    if (cidx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index, failed for lmsensors", cidx);
    }
    
    if (!TESTS_QUIET) { 
        printf("Component index for lmsensors: %d\n", cidx);
    }

    /* create a PAPI eventset */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
    }

    int modifier = PAPI_ENUM_FIRST;
    EventCode = PAPI_NATIVE_MASK;
    retval = PAPI_enum_cmp_event(&EventCode, modifier, cidx);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", retval);
    }   

    /* enumerate through the available lmsensors events and add a maximum of three */
    modifier = PAPI_ENUM_EVENTS;
    do {
        retval = PAPI_event_code_to_name(EventCode, EventName);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
        }

        retval = PAPI_add_named_event(EventSet, EventName);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", retval);
        }

        if (!TESTS_QUIET) {
            printf("Successfully added %s to the EventSet.\n", EventName);
        }

        /* store current event name and increment count */
        int strLen = snprintf(lm_events[event_cnt], PAPI_MAX_STR_LEN, "%s", EventName);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write event name: %s into array.\n", EventName);
            return PAPI_EBUF;
        }
        event_cnt++;
    } while( ( PAPI_enum_cmp_event(&EventCode, modifier, cidx) == PAPI_OK ) && ( event_cnt < NUM_EVENTS ) );

    /* start counting */
    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", retval);
    }

    /* read temp values */
    retval = PAPI_read(EventSet, values);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_read", retval);
    }

    /* stop counting and ignore values */
    retval = PAPI_stop(EventSet, NULL);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
    }

    /* for each event successfully added print their counter values */
    if (!TESTS_QUIET) {
        printf("Read counters from the EventSet:\n");
        for (i = 0; i < event_cnt; i++) {
            printf("Event: %s, Counter Value: %lld\n", lm_events[i], values[i]);
        }
    } 
   
    /* cleanup for PAPI */
    retval = PAPI_cleanup_eventset(EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
    }

    retval = PAPI_destroy_eventset(&EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destory_eventset", retval);
    }

    PAPI_shutdown();

    /* if we make it here everything ran succesfully */
    test_pass(__FILE__);

    return PAPI_OK;
}
