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
 *   Prints the value of every native event (by code)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

int main (int argc, char **argv)
{
    int retval,cid,numcmp;
    int EventSet = PAPI_NULL;
    long long *values = 0;
    int *codes = 0;
    char *names = 0;
    int code, i;
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
        printf("Trying all infiniband events\n");
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

        if ( strstr(cmpinfo->name, "infiniband") == NULL) {
            continue;
        }
        if (cmpinfo->disabled) {
            test_skip(__FILE__,__LINE__,"Component infiniband is disabled", 0);
            continue;
        }

        values = (long long*) malloc(sizeof(long long) * cmpinfo->num_native_events);
        codes = (int*) malloc(sizeof(int) * cmpinfo->num_native_events);
        names = (char*) malloc(PAPI_MAX_STR_LEN * cmpinfo->num_native_events);

        EventSet = PAPI_NULL;

        retval = PAPI_create_eventset( &EventSet );
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_create_eventset()", retval);
        }

        code = PAPI_NATIVE_MASK;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );
        i = 0;
        while ( r == PAPI_OK ) {

            retval = PAPI_event_code_to_name( code, &names[i*PAPI_MAX_STR_LEN] );
            if ( retval != PAPI_OK ) {
                test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
            }
            codes[i] = code;

            retval = PAPI_add_event( EventSet, code );
            if (retval != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_add_event()", retval);
            }

            total_events++;

            r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
            i += 1;
        }

        retval = PAPI_start( EventSet );
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_start()", retval);
        }

        /* XXX figure out a general method to  generate some traffic 
         * for infiniband
         * the operation should take more than one second in order
         * to guarantee that the network counters are updated */
        /* For now, just sleep for 10 seconds */
        sleep(10);

        retval = PAPI_stop( EventSet, values);
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_stop()", retval);
        }

        if (!TESTS_QUIET) {
           for (i=0 ; i<cmpinfo->num_native_events ; ++i)
               printf("%#x %-24s = %lld\n", codes[i], names+i*PAPI_MAX_STR_LEN, values[i]);
        }

        retval = PAPI_cleanup_eventset( EventSet );
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset()", retval);
        }

        retval = PAPI_destroy_eventset( &EventSet );
        if (retval != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset()", retval);
        }

        free(names);
        free(codes);
        free(values);
    }

    if (total_events==0) {
        test_skip(__FILE__,__LINE__,"No infiniband events found", 0);
    }

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
