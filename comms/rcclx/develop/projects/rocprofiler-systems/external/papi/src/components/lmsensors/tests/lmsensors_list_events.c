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
 *   List the event code and event name for all 
 *   available lmsensor events on the current 
 *   machine.
 */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

int main(int argc, char **argv) 
{
    int retval, event_cnt = 0, EventCode, cidx;
    char EventName[PAPI_2MAX_STR_LEN];

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
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index failed for lmsensors", cidx);
    }

    if (!TESTS_QUIET) { 
        printf("Component index for lmsensors: %d\n", cidx);
    }   

    int modifier = PAPI_ENUM_FIRST;
    EventCode = PAPI_NATIVE_MASK;
    retval = PAPI_enum_cmp_event(&EventCode, modifier, cidx);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", retval);
    }

    /* enumerate through all lmsensor events found on the current machine */ 
    modifier = PAPI_ENUM_EVENTS;
    do {
        /* print output header  */
        if (event_cnt == 0 && !TESTS_QUIET) {
            printf("%s %s", "Event code", "Event name\n");
        }

        retval = PAPI_event_code_to_name(EventCode, EventName);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
        }

        if (!TESTS_QUIET) {
            printf("%d %s\n", EventCode, EventName);
        }

        /* increment lmsensors event count */
        event_cnt++;
    } while(PAPI_enum_cmp_event(&EventCode, modifier, cidx) == PAPI_OK);

    if (!TESTS_QUIET) {
        printf("Total number of events for lmsensors: %d\n", event_cnt);
    }

    PAPI_shutdown();

    /* if we make it here everything ran succesfully */
    test_pass(__FILE__);

    return PAPI_OK;
}
