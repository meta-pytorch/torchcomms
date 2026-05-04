/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Jose Pedro Oliveira
 *
 * test case for the linux-net component
 *
 * @brief
 *   Prints the value of every net event (by code)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define IFNAME     "lo"
#define PINGADDR   "127.0.0.1"

int main (int argc, char **argv)
{
    int retval,cid,numcmp;
    int EventSet = PAPI_NULL;
    long long value;
    int code;
    char event_name[PAPI_MAX_STR_LEN];
    int total_events=0;
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
        printf("Trying all net events\n");
    }

    numcmp = PAPI_num_components();

    for(cid=0; cid<numcmp; cid++) {

        if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
            test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n",-1);
        }

        if (!TESTS_QUIET) {
            printf("Component %d - %d events - %s\n", cid,
                cmpinfo->num_native_events, cmpinfo->name);
        }

        if ( strstr(cmpinfo->name, "net") == NULL) {
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
                printf("%#x %-24s = ", code, event_name);
            }

            EventSet = PAPI_NULL;

            retval = PAPI_create_eventset( &EventSet );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_create_eventset()", retval);
            }

            retval = PAPI_add_event( EventSet, code );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_add_event()", retval);
            }

            retval = PAPI_start( EventSet );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_start()", retval);
            }

            if (strcmp(event_name, IFNAME ".rx.packets") == 0) {
                /* generate some traffic
                 * the operation should take more than one second in order
                 * to guarantee that the network counters are updated */
                retval = system("ping -c 4 " PINGADDR " > /dev/null");
                if (retval < 0) {
					test_fail(__FILE__, __LINE__, "Unable to start ping", retval);
				}
            }

            retval = PAPI_stop( EventSet, &value );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_stop()", retval);
            }

            if (!TESTS_QUIET) printf("%lld\n", value);

            retval = PAPI_cleanup_eventset( EventSet );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset()", retval);
            }

            retval = PAPI_destroy_eventset( &EventSet );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset()", retval);
            }

            total_events++;

            r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
        }

    }

    if (total_events==0) {
        test_skip(__FILE__,__LINE__,"No net events found", 0);
    }

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
