/*
 * This tests adding invalid events
 */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "event_name_lib.h"

int main( int argc, char **argv ) {

	int retval;

	int EventSet = PAPI_NULL;
	int quiet=0;
	char user_event[4096];
	long long values[1];

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (get_invalid_event_name(user_event,4096)==NULL) {
		if (!quiet) {
			printf("No sample invalid event defined for this architecture\n");
		}
		test_skip( __FILE__, __LINE__, "No event", 0);
	}

	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	retval = PAPI_add_named_event(EventSet, user_event);
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Correctly failed adding invalid event  %s %s\n",user_event,PAPI_strerror(retval));
		}
		test_pass(__FILE__);

	}

	PAPI_start(EventSet);

	PAPI_stop(EventSet,&values[0]);

	if (!quiet) {
		printf("Read result: %lld\n",values[0]);
	}

	test_fail( __FILE__, __LINE__,"Added comma separated event somehow",0);

	return 0;
}
