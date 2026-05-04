/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    example_multiple_components.c
 * @author  Vince Weaver
 *	    vweaver1@eecs.utk.edu
 * test if multiple components can be used at once
 *
 *
 * @brief
 *  This tests to see if the CPU component and Example component
 *   can be used simultaneously.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 1

int main (int argc, char **argv)
{

        int retval;
	int EventSet1 = PAPI_NULL, EventSet2 = PAPI_NULL;
	long long values1[NUM_EVENTS];
	long long values2[NUM_EVENTS];
	const PAPI_component_info_t *cmpinfo = NULL;
	int numcmp,cid,example_cid=-1;
	int code;
	int quiet=0;

        /* Set TESTS_QUIET variable */
        quiet=tests_quiet( argc, argv );

	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
	}

	if (!quiet) {
	   printf( "Testing simultaneous component use with PAPI %d.%d.%d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	}

	/* Find our component */

	numcmp = PAPI_num_components();
	for( cid=0; cid<numcmp; cid++) {
	   if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	      test_fail(__FILE__, __LINE__,
                           "PAPI_get_component_info failed\n", 0);
	   }
	   if (!quiet) {
	      printf("\tComponent %d - %d events - %s\n", cid,
		     cmpinfo->num_native_events,
		     cmpinfo->name);
	   }
	   if (strstr(cmpinfo->name,"example")) {
	      /* FOUND! */
	      example_cid=cid;
	   }
	}


	if (example_cid<0) {
	   test_skip(__FILE__, __LINE__,
		     "Example component not found\n", 0);
	}

	if (!quiet) {
	  printf("\nFound Example Component at id %d\n",example_cid);
	}


	/* Create an eventset for the Example component */

	retval = PAPI_create_eventset( &EventSet1 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "PAPI_create_eventset() failed\n", retval );
	}

	retval = PAPI_event_name_to_code("EXAMPLE_CONSTANT", &code);
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "EXAMPLE_ZERO not found\n",retval );
	}

	retval = PAPI_add_event( EventSet1, code);
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "PAPI_add_events failed\n", retval );
	}


	/* Create an eventset for the CPU component */

	retval = PAPI_create_eventset( &EventSet2 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "PAPI_create_eventset() failed\n", retval );
	}

	retval = PAPI_event_name_to_code("PAPI_TOT_CYC", &code);
	if ( retval != PAPI_OK ) {
	  test_skip( __FILE__, __LINE__,
		      "PAPI_TOT_CYC not available\n",retval );
	}

	retval = PAPI_add_event( EventSet2, code);
	if ( retval != PAPI_OK ) {
	   test_skip( __FILE__, __LINE__,
		      "NO CPU component found\n", retval );
	}

	if (!quiet) printf("\nStarting EXAMPLE_CONSTANT and PAPI_TOT_CYC at the same time\n");

	/* Start CPU component event */
	retval = PAPI_start( EventSet2 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "PAPI_start failed\n",retval );
	}

	/* Start example component */
	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__,
		      "PAPI_start failed\n",retval );
	}




	/* Stop example component */
	retval = PAPI_stop( EventSet1, values1 );
	if ( retval != PAPI_OK ) {
	   test_fail(  __FILE__, __LINE__, "PAPI_stop failed\n", retval);
	}

	/* Stop CPU component */
	retval = PAPI_stop( EventSet2, values2 );
	if ( retval != PAPI_OK ) {
	   test_fail(  __FILE__, __LINE__, "PAPI_stop failed\n", retval);
	}

	if (!quiet) printf("Stopping EXAMPLE_CONSTANT and PAPI_TOT_CYC\n\n");


	if (!quiet) printf("Results from EXAMPLE_CONSTANT: %lld\n",values1[0]);

	if (values1[0]!=42) {
	   test_fail(  __FILE__, __LINE__, "Result should be 42!\n", 0);
	}

	if (!quiet) printf("Results from PAPI_TOT_CYC: %lld\n\n",values2[0]);

	if (values2[0]<1) {
	   test_fail(  __FILE__, __LINE__, "Result should greater than 0\n", 0);
	}

	/* Cleanup EventSets */
	retval = PAPI_cleanup_eventset(EventSet1);
	if (retval != PAPI_OK) {
	   test_fail(  __FILE__, __LINE__, "PAPI_cleanup_eventset!\n", retval);
	}

	retval = PAPI_cleanup_eventset(EventSet2);
	if (retval != PAPI_OK) {
	   test_fail(  __FILE__, __LINE__, "PAPI_cleanup_eventset!\n", retval);
	}

	/* Destroy EventSets */
	retval = PAPI_destroy_eventset(&EventSet2);
	if (retval != PAPI_OK) {
	   test_fail(  __FILE__, __LINE__, "PAPI_destroy_eventset!\n", retval);
	}

	test_pass( __FILE__ );

	return 0;
}

