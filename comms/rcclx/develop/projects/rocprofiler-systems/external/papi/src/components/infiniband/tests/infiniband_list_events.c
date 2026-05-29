/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Jose Pedro Oliveira
 *
 * test case for the linux-infiniband component
 * Adapted from its counterpart in the net component.
 *
 * @brief
 *   List all net events codes and names
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

int main (int argc, char **argv)
{
    int retval,cid,numcmp;
    int total_events=0;
    int code;
    char event_name[PAPI_MAX_STR_LEN];
    int r;
    const PAPI_component_info_t *cmpinfo = NULL;

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
    }

    if (!TESTS_QUIET) {
        printf("Listing all infiniband events\n");
    }

    numcmp = PAPI_num_components();

    for(cid=0; cid<numcmp; cid++) {

        if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
            test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n",-1);
        }

        if ( strstr(cmpinfo->name, "infiniband") == NULL) {
            continue;
        }

        if (!TESTS_QUIET) {
            printf("Component %d (%d) - %d events - %s\n",
                cid, cmpinfo->CmpIdx,
                cmpinfo->num_native_events, cmpinfo->name);
        }
        if (cmpinfo->disabled) {
            test_skip(__FILE__,__LINE__,"Component infiniband is disabled", 0);
            continue;
        }

        code = PAPI_NATIVE_MASK;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );
        while ( r == PAPI_OK ) {

            retval = PAPI_event_code_to_name( code, event_name );
            if ( retval != PAPI_OK ) {
                test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
            }

            if (!TESTS_QUIET) {
                printf("%#x %s\n", code, event_name);
            }

            total_events++;

            r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
        }

    }

    if (total_events==0) {
        test_skip(__FILE__,__LINE__,"No infiniband events found", 0);
    }

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
