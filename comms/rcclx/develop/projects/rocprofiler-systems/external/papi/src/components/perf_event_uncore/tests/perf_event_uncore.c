/*
 * This file tests uncore events on perf_event kernels
 *
 * In this test we use the :cpu=0 way of attaching to the CPU
 * rather than the legacy PAPI way.
 */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "perf_event_uncore_lib.h"

int main( int argc, char **argv ) {

	int retval,quiet;
	int EventSet = PAPI_NULL;
	long long values[1];
    char *uncore_event_tmp=NULL;
	char uncore_event[BUFSIZ];
	char event_name[BUFSIZ];
	int uncore_cidx=-1;
	const PAPI_component_info_t *info;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if (!quiet) {
		printf("Testing the :cpu=0 way of attaching an uncore event to a core\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Find the uncore PMU */
	uncore_cidx=PAPI_get_component_index("perf_event_uncore");
	if (uncore_cidx<0) {
		if (!quiet) {
			printf("perf_event_uncore component not found\n");
		}
		test_skip(__FILE__,__LINE__,"perf_event_uncore component not found",0);
	}

	/* Check if component disabled */
	info=PAPI_get_component_info(uncore_cidx);
	if (info->disabled) {
		if (!quiet) {
			printf("perf_event_uncore component is disabled\n");
		}
		test_skip(__FILE__,__LINE__,"uncore component disabled",0);
	}

	/* Get a relevant event name */
	uncore_event_tmp=get_uncore_event(event_name, BUFSIZ);
	if (uncore_event_tmp==NULL) {
		if (!quiet) {
			printf("uncore event name not available\n");
		}
		test_skip( __FILE__, __LINE__,
			"PAPI does not support uncore on this processor",
			PAPI_ENOSUPP );
	}

	sprintf(uncore_event,"%s:cpu=0",uncore_event_tmp);

	/* Create an eventset */
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	/* Add our uncore event */
	retval = PAPI_add_named_event(EventSet, uncore_event);
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			printf("Error trying to use event %s\n", uncore_event);
		}
		test_fail(__FILE__, __LINE__, "adding uncore event",retval);
	}

	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our work code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("\tUsing event %s\n",uncore_event);
		printf("\t%s: %lld\n",uncore_event,values[0]);
	}

	test_pass( __FILE__ );

	return 0;
}
