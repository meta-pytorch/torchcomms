/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Tushar Mohan
 *
 * test case for the appio component
 * (adapted from test in linux-net component)
 *
 * @brief
 *   Prints the values of several (but not all) appio events specified by names
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"


#define NUM_EVENTS 11

int main (int argc, char **argv)
{
    int i, retval;
    int EventSet = PAPI_NULL;
    char *event_name[NUM_EVENTS] = {
        "READ_BYTES",
        "READ_CALLS",
        "READ_USEC",
        "READ_EOF",
        "READ_SHORT",
        "READ_ERR",
        "WRITE_BYTES",
        "WRITE_CALLS",
        "WRITE_USEC",
        "WRITE_ERR",
        "WRITE_SHORT"
    };
    int event_code[NUM_EVENTS] = { 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        printf("Appio events by name\n");
    }

    /* Map names to codes */
    for ( i=0; i<NUM_EVENTS; i++ ) {
        retval = PAPI_event_name_to_code( event_name[i], &event_code[i]);
        if ( retval != PAPI_OK ) {
            test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
        }

        total_events++;
    }

    int fdin,fdout;
    const char* infile = "/etc/group";
    if (!TESTS_QUIET) printf("This program will read %s and write it to /dev/null\n", infile);
    int bytes = 0;
    char buf[1024];

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

    fdin=open(infile, O_RDONLY);
    if (fdin < 0) perror("Could not open file for reading: \n");
    fdout=open("/dev/null", O_WRONLY);
    if (fdout < 0) perror("Could not open /dev/null for writing: \n");

    while ((bytes = read(fdin, buf, 1024)) > 0) {
      write(fdout, buf, bytes);
    }
    close(fdin);
    close(fdout);

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

    if (total_events==0) {
        test_skip(__FILE__,__LINE__,"No appio events found", 0);
    }

    test_pass( __FILE__ );

    return 0;
}

// vim:set ai ts=4 sw=4 sts=4 et:
