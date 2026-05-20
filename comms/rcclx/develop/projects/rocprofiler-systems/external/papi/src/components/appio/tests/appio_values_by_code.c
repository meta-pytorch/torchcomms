/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Tushar Mohan
 * (adapted from code in linux-net)
 * test case for the appio component
 *
 * @brief
 *   Prints the value of every appio event (by code)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define MAX_EVENTS 48

int main (int argc, char **argv)
{
    int retval,cid,numcmp;
    int EventSet = PAPI_NULL;
    int code;
    char event_names[MAX_EVENTS][PAPI_MAX_STR_LEN];
    int event_codes[MAX_EVENTS];
    long long event_values[MAX_EVENTS];
    int total_events=0; /* events added so far */
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
        printf("Trying all appio events\n");
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

        if ( strstr(cmpinfo->name, "appio") == NULL) {
            continue;
        }

        code = PAPI_NATIVE_MASK;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );
        /* Create and populate the EventSet */
        EventSet = PAPI_NULL;

        retval = PAPI_create_eventset( &EventSet );
         if (retval != PAPI_OK) {
             test_fail(__FILE__, __LINE__, "PAPI_create_eventset()", retval);
         }

        while ( r == PAPI_OK ) {

            retval = PAPI_event_code_to_name( code, event_names[total_events] );
            if ( retval != PAPI_OK ) {
                test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
            }
            
            if (!TESTS_QUIET) {
              printf("Added event %s (code=%#x)\n", event_names[total_events], code);
            }
            event_codes[total_events++] = code;
            r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
        }

    }

    int fdin,fdout;
    const char* infile = "/etc/group";
    printf("This program will read %s and write it to /dev/null\n", infile);
    int bytes = 0;
    char buf[1024];

    retval = PAPI_add_events( EventSet, event_codes, total_events);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_add_events()", retval);
    }

    retval = PAPI_start( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start()", retval);
    }

    fdin=open(infile, O_RDONLY);
    if (fdin < 0) perror("Could not open file for reading: \n");
    fdout = open("/dev/null", O_WRONLY);
    if (fdout < 0) perror("Could not open /dev/null for writing: \n");

    while ((bytes = read(fdin, buf, 1024)) > 0) {
      write(fdout, buf, bytes);
    }

    retval = PAPI_stop( EventSet, event_values );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop()", retval);
    }
    close(fdin);
    close(fdout);

    int i;
    if (!TESTS_QUIET) {
        for ( i=0; i<total_events; i++ ) {
            printf("%#x %-24s = %lld\n",
                event_codes[i], event_names[i], event_values[i]);
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

    if (total_events==0) {
        test_skip(__FILE__,__LINE__,"No appio events found", 0);
    }

    test_pass( __FILE__ );

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset()", retval);
    }

    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset()", retval);
    }
    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
