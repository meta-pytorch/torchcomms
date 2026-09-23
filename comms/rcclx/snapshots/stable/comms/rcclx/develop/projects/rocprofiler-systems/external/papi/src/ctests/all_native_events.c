/*
 * File:    all_native_events.c
 * Author:  Haihang You <you@cs.utk.edu>
 * Author:  Treece Burgess tburgess@icl.utk.edu (updated in November 2024 to add a flag to enable or disable Cuda events.)
 */

/* This test tries to add all native events from all components */

/* This file hardware info and performs the following test:
		- Start and stop all native events.
    This is a good preliminary way to validate native event tables.
	In its current form this test also stresses the number of
	events sets the library can handle outstanding.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

static int
check_event( int event_code, char *name, int quiet )
{
    int retval;
    long long values;
    int EventSet = PAPI_NULL;

    /* Possibly there was an older issue with the	*/
    /* REPLAY_EVENT:BR_MSP on Pentium4 ???		*/

    /* Create an eventset */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK ) {
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
    }

    /* Add the event */
    retval = PAPI_add_event( EventSet, event_code );
    if ( retval != PAPI_OK ) {
        if (!quiet) printf( "Error adding %s %d\n", name, retval );
            return retval;
    }

    /* Start the event */
    retval = PAPI_start( EventSet );
    if ( retval != PAPI_OK ) {
        PAPI_perror( "PAPI_start" );
    } else {
        retval = PAPI_stop( EventSet, &values );
        if ( retval != PAPI_OK ) {
            PAPI_perror( "PAPI_stop" );
            return retval;
        } else {
            if (!quiet) printf( "Added and Stopped %s successfully.\n", name );
        }
     }

     /* Cleanup the eventset */
     retval=PAPI_cleanup_eventset( EventSet );
     if (retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
     }

     /* Destroy the eventset */
     retval=PAPI_destroy_eventset( &EventSet );
     if (retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval);
     }

     return PAPI_OK;
}

static void
print_help(char **argv)
{
    printf( "This is the all_native_events program.\n" );
    printf( "For all components compiled in, it attempts to: add the component events, start, and stop.\n" );
    printf( "Usage: %s [options]\n", argv[0] );
    printf( "General command options:\n" );
    printf( "\t-h, --help                           Print the help message.\n" );
    printf( "\t--disable-cuda-events=<yes,no>       Optionally disable processing the Cuda native events. Default is no.\n" );
    printf( "\n" );
}

int
main( int argc, char **argv )
{

    int i, k, add_count = 0, err_count = 0;
    int retval;
    PAPI_event_info_t info, info1;
    const PAPI_hw_info_t *hwinfo = NULL;
    const PAPI_component_info_t* cmpinfo;
    char disableCudaEvts[PAPI_MIN_STR_LEN] = "no";
    int event_code;
    int numcmp, cid;
    int quiet;

    /* Set quiet variable */
    quiet=tests_quiet( argc, argv );

    /* parse command line flags */
    for (i = 0; i < argc; i++) {
        if (strncmp(argv[i], "--disable-cuda-events=", 22) == 0) {
            strncpy(disableCudaEvts, argv[i] + 22, PAPI_MIN_STR_LEN);
        }

        if (strncmp(argv[i], "--help", 6) == 0 ||
            strncmp(argv[i], "-h", 2) == 0) {
            print_help(argv);
            exit(-1);
        }
    }

    /* Init PAPI library */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
    }

    if (!quiet) {
        printf("Test case ALL_NATIVE_EVENTS: Available "
               "native events and hardware "
               "information.\n");
    }

    hwinfo=PAPI_get_hardware_info();
    if ( hwinfo == NULL ) {
        test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
    }

    numcmp = PAPI_num_components(  );

    int rocm_id = PAPI_get_component_index("rocm");
    int cuda_id = PAPI_get_component_index("cuda");
    /* Loop through all components */
    for( cid = 0; cid < numcmp; cid++ ) {
        if (cid == rocm_id) {
            /* skip rocm component due to a bug in rocprofiler that
             * crashes PAPI if multiple GPUs are present */
            continue;
        }

        /* optionally skip the Cuda native events, default is no  */
        if (strcmp(disableCudaEvts, "yes") == 0 && cid == cuda_id) {
            continue;
        }

        cmpinfo = PAPI_get_component_info( cid );
        if (cmpinfo  == NULL) {
            test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 2 );
        }

        /* Skip disabled components */
        if (cmpinfo->disabled != PAPI_OK && cmpinfo->disabled != PAPI_EDELAY_INIT) {
            if (!quiet) {
                printf( "Name:   %-23s %s\n",
                    cmpinfo->name ,cmpinfo->description);
                printf("   \\-> Disabled: %s\n",
                    cmpinfo->disabled_reason);
            }
            continue;
        }

        /* For platform independence, always ASK FOR the first event */
        /* Don't just assume it'll be the first numeric value */
        i = 0 | PAPI_NATIVE_MASK;
        retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );

        do {
            retval = PAPI_get_event_info( i, &info );
            event_code = ( int ) info.event_code;
            if ( check_event( event_code, info.symbol, quiet ) == PAPI_OK) {
                add_count++;
            }
            else {
                err_count++;
            }

            /* We used to skip OFFCORE and UNCORE events  */
            /* Why? */

            /* Enumerate all umasks */
            k = i;
            if ( PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid )==PAPI_OK ) {
                do {
                    retval = PAPI_get_event_info( k, &info1 );
                    event_code = ( int ) info1.event_code;
                    if ( check_event( event_code, info1.symbol, quiet ) == PAPI_OK ) {
                        add_count++;
                    }
                    else {
                        err_count++;
                    }
                } while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK );
            }
        } while ( PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid ) == PAPI_OK );

    }

    if (!quiet) {
        printf( "\n\nSuccessfully found and added %d events "
            "(in %d eventsets).\n",
            add_count , add_count);
    }

    if ( err_count ) {
        if (!quiet) printf( "Failed to add %d events.\n", err_count );
    }

    if ( add_count <= 0 ) {
        test_fail( __FILE__, __LINE__, "No events added", 1 );
    }

    test_pass( __FILE__ );

    return 0;
}
