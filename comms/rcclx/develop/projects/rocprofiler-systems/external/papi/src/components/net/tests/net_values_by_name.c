/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Jose Pedro Oliveira
 *
 * test case for the linux-net component
 *
 * @brief
 *   Prints the values of several net events specified by names
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

/*
#define IFNAME     "eth0"
*/
#define IFNAME     "lo"
#define PINGADDR   "127.0.0.1"

#define NUM_EVENTS 4

int main (int argc, char **argv)
{
    int i, retval;
    int EventSet = PAPI_NULL;
    char *event_name[NUM_EVENTS] = {
        IFNAME ":rx:bytes",
        IFNAME ":rx:packets",
        IFNAME ":tx:bytes",
        IFNAME ":tx:packets",
    };
    int event_code[NUM_EVENTS] = { 0, 0, 0, 0};
    long long event_value[NUM_EVENTS];
    int total_events=0;

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
    }

    if (!TESTS_QUIET) {
        printf("Net events by name\n");
    }

    /* Map names to codes */
    for ( i=0; i<NUM_EVENTS; i++ ) {
        retval = PAPI_event_name_to_code( event_name[i], &event_code[i]);
        if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
        }

        total_events++;
    }

    /* Create and populate the EventSet */
    EventSet = PAPI_NULL;

    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset()", retval);
    }

    retval = PAPI_add_events( EventSet, event_code, NUM_EVENTS);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_add_events()", retval);
    }

    retval = PAPI_start( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start()", retval);
    }

    /* generate some traffic
     * the operation should take more than one second in order
     * to guarantee that the network counters are updated */
    retval = system("ping -c 4 " PINGADDR " > /dev/null");
    if (retval < 0) {
		test_fail(__FILE__, __LINE__, "Unable to start ping", retval);
	}

    retval = PAPI_stop( EventSet, event_value );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start()", retval);
    }

    if (!TESTS_QUIET) {
        for ( i=0; i<NUM_EVENTS; i++ ) {
            printf("%#x %-24s = %lld\n",
                event_code[i], event_name[i], event_value[i]);
        }
    }

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset()", retval);
    }

    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset()", retval);
    }

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
