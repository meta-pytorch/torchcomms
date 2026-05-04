/* destroy.c */

/* Test that PAPI_destroy_eventset() doesn't leak file descriptors */

/* Run create/add/start/stop/remove/destroy in a large loop */
/* and make sure PAPI handles it OK */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define	NUM_EVENTS	1
#define NUM_LOOPS	16384

int main( int argc, char **argv ) {

	int retval, i;
	int EventSet1 = PAPI_NULL;
	long long values[NUM_EVENTS];
	int quiet=0;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("Testing to make sure we can destroy eventsets\n");
	}

	for(i=0;i<NUM_LOOPS;i++) {

		/* Initialize the EventSet */
		retval=PAPI_create_eventset(&EventSet1);
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"PAPI_create_eventset", retval );
		}

		/* Add PAPI_TOT_CYC */
		retval=PAPI_add_named_event(EventSet1,"PAPI_TOT_CYC");
		if (retval!=PAPI_OK) {
			if (!quiet) {
				printf("Trouble adding PAPI_TOT_CYC: %s\n",
					PAPI_strerror(retval));
			}
			test_fail( __FILE__, __LINE__,
				"adding PAPI_TOT_CYC", retval );
		}

		/* Start PAPI */
		retval = PAPI_start( EventSet1 );
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
		}

		/* Stop PAPI */
		retval = PAPI_stop( EventSet1, values );
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		}

		/* Shutdown the EventSet */
		retval = PAPI_remove_named_event( EventSet1, "PAPI_TOT_CYC" );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"PAPI_remove_named_event", retval );
		}

		retval=PAPI_destroy_eventset( &EventSet1 );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"PAPI_destroy_eventset", retval );
		}

	}

	test_pass( __FILE__ );

	return 0;
}
