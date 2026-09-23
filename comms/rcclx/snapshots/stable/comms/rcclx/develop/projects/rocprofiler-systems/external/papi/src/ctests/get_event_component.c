/*
 * File:    get_event_component.c
 * Author:  Vince Weaver vweaver1@eecs.utk.edu
 * Author:  Treece Burgess tburgess@icl.utk.edu (updated in November 2024 to add a flag to enable or disable Cuda events.)
 */

/*
  This test makes sure PAPI_get_event_component() works
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

static void
print_help(char **argv)
{
    printf( "This is the get_event_component program.\n" );
    printf( "For all components compiled in, it uses the function\n" );
    printf( "PAPI_get_event_component to get the appropriate component index for a native event.\n");
    printf( "Usage: %s [options]\n", argv[0] );
    printf( "General command options:\n" );
    printf( "\t-h, --help                           Print the help message.\n" );
    printf( "\t--disable-cuda-events=<yes,no>       Optionally disable processing the Cuda native events. Default is no.\n" );
    printf( "\n" );
}

int
main( int argc, char **argv )
{

    int i;
    int retval;
    PAPI_event_info_t info;
    int numcmp, cid, our_cid;
    const PAPI_component_info_t* cmpinfo;
    char disableCudaEvts[PAPI_MIN_STR_LEN] = "no";

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

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

    numcmp = PAPI_num_components(  );


    /* Loop through all components */
    for( cid = 0; cid < numcmp; cid++ )
    {
        cmpinfo = PAPI_get_component_info( cid );

        if (cmpinfo  == NULL)
        {
            test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 2 );
        }

        /* optionally skip the Cuda native events, default is no  */
        if (strcmp(cmpinfo->name, "cuda") == 0 &&
            strcmp(disableCudaEvts, "yes") == 0)
        {
            continue;
        }


        if (cmpinfo->disabled != PAPI_OK && cmpinfo->disabled != PAPI_EDELAY_INIT && !TESTS_QUIET) {
            printf( "Name:   %-23s %s\n", cmpinfo->name ,cmpinfo->description);
            printf("   \\-> Disabled: %s\n",cmpinfo->disabled_reason);
            continue;
        }


        i = 0 | PAPI_NATIVE_MASK;
        retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );
        if (retval!=PAPI_OK) continue;

        do {
            if (PAPI_get_event_info( i, &info ) != PAPI_OK) {
                if (!TESTS_QUIET) {
                    printf("Getting information about event: %#x failed\n", i);
                }
                continue;
            }
            our_cid=PAPI_get_event_component(i);

            if (our_cid!=cid) {
                if (!TESTS_QUIET) {
                    printf("%d %d %s\n",cid,our_cid,info.symbol);
                }
                test_fail( __FILE__, __LINE__, "component mismatch", 1 );
            }

            if (!TESTS_QUIET) {
                printf("%d %d %s\n",cid,our_cid,info.symbol);
            }

	  
        } while ( PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid ) == PAPI_OK );

    }

    test_pass( __FILE__ );

    return 0;
}
