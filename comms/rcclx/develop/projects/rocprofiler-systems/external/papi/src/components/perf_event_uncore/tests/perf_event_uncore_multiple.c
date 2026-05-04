/*
 * This file tests measuring uncore and non-uncore events at the same time
 *
 * Despite perf_event supporting this, PAPI had to do this with
 * separate event sets on separate components.
 *
 * PAPI does not allow two eventsets to be running simultaneously
 * on the same component, nor does it allow events in the same
 * event set to have different domains/granularities.
 */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "perf_event_uncore_lib.h"

int main( int argc, char **argv ) {

	int retval,quiet;
	int EventSet = PAPI_NULL;
	int EventSet2 = PAPI_NULL;
	long long values[1],values2[1];
	char *uncore_event=NULL;
	char event_name[BUFSIZ];
	int uncore_cidx=-1;
	const PAPI_component_info_t *info;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

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
			printf("perf_event_uncore component disabled\n");
		}
		test_skip(__FILE__,__LINE__,"uncore component disabled",0);
	}

	/* Get a relevant event name */
	uncore_event=get_uncore_event(event_name, BUFSIZ);
	if (uncore_event==NULL) {
		if (!quiet) {
			printf("Could not find an uncore event for this processor\n");
		}
		test_skip( __FILE__, __LINE__,
			"PAPI does not support uncore on this processor",
			PAPI_ENOSUPP );
	}

	/* Create an eventset */
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	/* Create another eventset */
	retval = PAPI_create_eventset(&EventSet2);
	if (retval != PAPI_OK) {
		test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	/* Set a component for the EventSet */
	retval = PAPI_assign_eventset_component(EventSet, uncore_cidx);

	/* we need to set to a certain cpu for uncore to work */

	PAPI_cpu_option_t cpu_opt;

	cpu_opt.eventset=EventSet;
	cpu_opt.cpu_num=0;

	retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
	if (retval != PAPI_OK) {
		if (!quiet) {
			printf("Could not PAPI_CPU_ATTACH\n");
		}
		test_skip( __FILE__, __LINE__,
			"this test; trying to PAPI_CPU_ATTACH; need to run as root",
			retval);
	}

	/* we need to set the granularity to system-wide for uncore to work */

	PAPI_granularity_option_t gran_opt;

	gran_opt.def_cidx=0;
	gran_opt.eventset=EventSet;
	gran_opt.granularity=PAPI_GRN_SYS;

	retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
	if (retval != PAPI_OK) {
		if (!quiet) {
			printf("Could not set PAPI_GRN_SYS\n");
		}
		test_skip( __FILE__, __LINE__,
			"this test; trying to set PAPI_GRN_SYS",
			retval);
	}

	/* we need to set domain to be as inclusive as possible */

	PAPI_domain_option_t domain_opt;

	domain_opt.def_cidx=0;
	domain_opt.eventset=EventSet;
	domain_opt.domain=PAPI_DOM_ALL;

	retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
	if (retval != PAPI_OK) {
		if (!quiet) {
			printf("Could not set PAPI_DOM_ALL\n");
		}
		test_skip( __FILE__, __LINE__,
			"this test; trying to set PAPI_DOM_ALL; need to run as root",
			retval);
	}

	/* Add our uncore event */
	retval = PAPI_add_named_event(EventSet, uncore_event);
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			printf("Error trying to use event %s\n", uncore_event);
		}
		test_fail(__FILE__, __LINE__, "adding uncore event ",retval);
	}

	/* Add PAPI_TOT_CYC */
	retval = PAPI_add_named_event(EventSet2, "PAPI_TOT_CYC");
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			printf("Error trying to add PAPI_TOT_CYC\n");
		}
		test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
	}


	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	retval = PAPI_start( EventSet2 );
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

	retval = PAPI_stop( EventSet2, values2 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf("Uncore and regular event test:\n");
		printf("Using uncore event %s\n",uncore_event);
		printf("\t%s: %lld\n",uncore_event,values[0]);
		printf("\t%s: %lld\n","PAPI_TOT_CYC",values2[0]);
	}

	test_pass( __FILE__ );

	return 0;
}
