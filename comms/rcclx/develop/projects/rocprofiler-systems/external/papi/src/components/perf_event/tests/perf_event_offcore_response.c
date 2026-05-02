/*
 * This tests the use of offcore_response events
 */

#include <stdio.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "event_name_lib.h"

int main( int argc, char **argv ) {

	int quiet;

	char *offcore_event=NULL;
	char event_name[BUFSIZ];

	int retval;
	int EventSet1 = PAPI_NULL;

	long long total_values[1];

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_create_eventset(&EventSet1);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	/* Get a relevant event name */
	offcore_event=get_offcore_event(event_name, BUFSIZ);
	if (offcore_event==NULL) {
		if (!quiet) {
			printf("No test event available on this processor\n");
		}
		test_skip( __FILE__, __LINE__,
                	"PAPI does not support offcore on this processor",
			PAPI_ENOSUPP );
	}

	retval = PAPI_add_named_event(EventSet1, offcore_event);
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to add %s\n",offcore_event);
		}
		test_fail(__FILE__, __LINE__, "adding offcore event ",retval);
	}

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet1, total_values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("\t%s count = %lld\n",offcore_event,total_values[0]);
	}

	test_pass( __FILE__ );

	return 0;
}
