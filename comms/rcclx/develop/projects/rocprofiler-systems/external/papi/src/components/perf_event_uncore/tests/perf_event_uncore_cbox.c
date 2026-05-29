/*
 * This file tests cbox uncore events on Intel Processors
 */

#include <stdio.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "perf_event_uncore_lib.h"

#define EVENTS_TO_TRY 16
#define MAX_PACKAGES  4

int main( int argc, char **argv ) {

	int retval,i,j,quiet;
	int EventSet[EVENTS_TO_TRY][MAX_PACKAGES];
	long long values[EVENTS_TO_TRY][MAX_PACKAGES];
	char event_name[BUFSIZ];
	char uncore_base[BUFSIZ];
	char uncore_event[BUFSIZ];
	int uncore_cidx=-1;
	int max_cbox=0;
	int core_to_use=0;
	char *result;

	const PAPI_hw_info_t *hwinfo;

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

	/* Get hardware info */
	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		test_fail(__FILE__,__LINE__,"PAPI_get_hardware_info()",retval);
	}

	/* Get event to use */
	if (hwinfo->vendor == PAPI_VENDOR_INTEL) {
		result=get_uncore_cbox_event(event_name,uncore_base,BUFSIZ);

		if (result==NULL) {
			if (!quiet) {
				printf("No event available\n");
			}
			test_skip( __FILE__, __LINE__,
				"No event available", PAPI_ENOSUPP );
		}
	}
	else {
		if (!quiet) {
			printf("We only support Intel for now\n");
		}
		test_skip( __FILE__, __LINE__,
			"This test only supported Intel chips", PAPI_ENOSUPP );
	}

	if (!quiet) {
		printf("Trying for %d sockets\n",hwinfo->sockets);
		printf("threads %d cores %d ncpus %d\n", hwinfo->threads,hwinfo->cores,
			hwinfo->ncpu);
	}

	for(i=0;i < hwinfo->sockets; i++) {

		/* perf_event provides which to use in "cpumask"    */
		/* but libpfm4 doesn't report this back to us (yet) */
		core_to_use=i*hwinfo->threads*hwinfo->cores;
		if (!quiet) {
			printf("Using core %d for socket %d\n",core_to_use,i);
		}

		for(j=0;j<EVENTS_TO_TRY;j++) {

			/* Create an eventset */
			EventSet[j][i]=PAPI_NULL;
			retval = PAPI_create_eventset(&EventSet[j][i]);
			if (retval != PAPI_OK) {
				test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
			}

			/* Set a component for the EventSet */
			retval = PAPI_assign_eventset_component(EventSet[j][i], uncore_cidx);
			if (retval!=PAPI_OK) {
				test_fail(__FILE__, __LINE__, "PAPI_assign_eventset_component",retval);
			}

			/* we need to set to a certain cpu for uncore to work */

			PAPI_cpu_option_t cpu_opt;

			cpu_opt.eventset=EventSet[j][i];
			cpu_opt.cpu_num=core_to_use;

			retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
			if (retval != PAPI_OK) {
				if (!quiet) {
					printf("Problem with PAPI_CPU_ATTACH\n");
				}
				test_skip( __FILE__, __LINE__,
					"this test; trying to PAPI_CPU_ATTACH; need to run as root",
					retval);
			}

			/* Default Granularity should work */

			/* Default domain should work */

			/* Add our uncore event */
			retval = snprintf(uncore_event, BUFSIZ, "%s%d::%s",uncore_base,j,event_name);
            if( retval >= BUFSIZ ){
                fprintf(stderr,"Event full name \"%s%d::%s\" has been truncated to \"%s\"\n",uncore_base,j,event_name, uncore_event);
            }
			retval = PAPI_add_named_event(EventSet[j][i], uncore_event);
			if (retval != PAPI_OK) {
				max_cbox=j;
				break;
			}
			if (!quiet) {
				printf("Added %s for socket %d\n",uncore_event,i);
			}

		}
	}

	for(i=0;i < hwinfo->sockets; i++) {

		for(j=0;j<max_cbox;j++) {
			if (!quiet) {
				printf("Starting EventSet %d\n",EventSet[j][i]);
			}

			/* Start PAPI */
			retval = PAPI_start( EventSet[j][i] );
			if ( retval != PAPI_OK ) {
				printf("Error starting socket %d cbox %d\n",i,j);
				test_fail( __FILE__, __LINE__, "PAPI_start", retval );
			}
		}
	}

	/* our work code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	for(i=0;i < hwinfo->sockets; i++) {
		for(j=0;j<max_cbox;j++) {
			retval = PAPI_stop( EventSet[j][i],&values[j][i] );
			if ( retval != PAPI_OK ) {
				if (!quiet) {
					printf("Error stopping socket %d cbox %d\n",i,j);
				}
				test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
			}
		}
	}

	/* Print Results */
	if ( !quiet ) {
		for(i=0;i < hwinfo->sockets; i++) {
			printf("Socket %d\n",i);
			for(j=0;j<max_cbox;j++) {
				printf("\t%s%d::%s %lld\n",uncore_base,j,event_name,values[j][i]);
			}
		}
	}

	PAPI_shutdown();

	test_pass( __FILE__ );

	return 0;
}
