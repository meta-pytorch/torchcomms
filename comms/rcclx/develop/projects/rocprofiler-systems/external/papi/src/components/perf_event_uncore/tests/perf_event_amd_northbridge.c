/*
 * This file tests uncore events on AMD fam15h Northbridge machines
 *  The Linux perf_event developers introduced fam15h Northbridge
 *   support in Linux 3.9 with an interfae similar to fam10h
 *   where the events were part of the core CPU
 *  They broke the ABI with Linux 3.10 and made fam15h NB a separate
 *   PMU, like the Intel uncore support.
 */

#include <stdio.h>
#include <string.h>

#include <sys/utsname.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int main( int argc, char **argv ) {

	int retval;
	int EventSet = PAPI_NULL;
	long long values[1];
	char event_name[BUFSIZ];
	int uncore_cidx=-1;
	const PAPI_hw_info_t *hwinfo;
	int quiet;
	struct utsname uname_info;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	uname(&uname_info);
	if (!quiet) printf("Found Linux %s\n",uname_info.release);

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Check for AMD machine */
	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 0 );
	}

	if (hwinfo->vendor != PAPI_VENDOR_AMD) {
		if (!quiet) printf("Test only for AMD machines\n");
		test_skip(__FILE__,__LINE__,"Test only for AMD processor",0);
	}

	if ( hwinfo->cpuid_family != 21) {
		if (!quiet) printf("Test only for fam15h AMD machines\n");
		test_skip(__FILE__,__LINE__,"Test only for fam15h AMD processor",0);
	}

	if (!strcmp(uname_info.release,"3.9")) {

		if (!quiet) printf("Detected 3.9 kernel, using perf_event\n");

		/* For kernel 3.9 use regular CPU component */

		/* Find the uncore PMU */
		uncore_cidx=PAPI_get_component_index("perf_event");
		if (uncore_cidx<0) {
			test_skip(__FILE__,__LINE__,"perf_event component not found",0);
		}

		/* Get a relevant event name */
		strncpy(event_name,"DRAM_ACCESSES:ALL", BUFSIZ);
	}
	else {

		/* 3.10 and later */

		if (!quiet) {
			printf("Detected > 3.9 kernel, using perf_event_uncore\n");
		}

		/* Find the uncore PMU */
		uncore_cidx=PAPI_get_component_index("perf_event_uncore");
		if (uncore_cidx<0) {
			test_skip(__FILE__,__LINE__,"perf_event_uncore component not found",0);
		}

		/* Get a relevant event name */
		/* This might change once libpfm4 gets new fam15h NB support */
		strncpy(event_name,"DRAM_ACCESSES:ALL", BUFSIZ);
	}

	/* Create an eventset */
	retval = PAPI_create_eventset(&EventSet);
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
		test_skip( __FILE__, __LINE__,
			"this test; trying to set PAPI_DOM_ALL; need to run as root",
			retval);
	}

	/* Add our uncore event */
	retval = PAPI_add_named_event(EventSet, event_name);
	if (retval != PAPI_OK) {
		if ( !quiet ) {
			fprintf(stderr,"Error trying to use event %s\n", event_name);
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
		printf("AMD fam15h Northbridge test:\n");
		printf("Using event %s\n",event_name);
		printf("\t%s: %lld\n",event_name,values[0]);
	}

	test_pass( __FILE__ );

	return 0;
}
