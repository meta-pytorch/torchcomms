/* This file attempts to test the PAPI_HW_INT				*/
/* performance counter (Total hardware interrupts).			*/

/* This assumes that interrupts will be happening in the background */
/* Including a regular timer tick of HZ.  This is not always true   */
/* but should be roughly true on your typical Linux system.         */

/* by Vince Weaver, <vincent.weaver@maine.edu>	*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include <time.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"


int main(int argc, char **argv) {

	int quiet;

	long long count;
	int retval;
	int eventset=PAPI_NULL;

	struct timespec before,after;
	unsigned long long seconds;
        unsigned long long ns;

	quiet=tests_quiet(argc,argv);

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("\nTesting PAPI_HW_INT\n");
	}

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset,"PAPI_HW_INT");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_HW_INT\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_HW_INT", retval );
        }

	/********************************/
	/* testing 3 seconds of runtime */
	/********************************/

	if (!quiet) {
		printf("\nRunning for 3 seconds\n");
	}

	clock_gettime(CLOCK_REALTIME,&before);

	PAPI_reset(eventset);
	PAPI_start(eventset);

	while(1) {
		clock_gettime(CLOCK_REALTIME,&after);

		seconds=after.tv_sec - before.tv_sec;
		ns = after.tv_nsec - before.tv_nsec;
		ns = (seconds*1000000000ULL)+ns;

		/* be done if 3 billion nanoseconds has passed */
		if (ns>3000000000ULL) break;
	}

	retval=PAPI_stop(eventset,&count);

	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "Problem stopping!", retval );
	}

	if (!quiet) {
		printf("\tMeasured interrupts = %lld\n",count);
		/* FIXME: find actua Hz on system */
		/* Or even, read /proc/interrupts */
		printf("\tAssuming HZ=250, expect roughly 750\n");
	}

	if (!quiet) printf("\n");

	if (count<10) {
		if (!quiet) printf("Too few interrupts!\n");
		test_fail( __FILE__, __LINE__, "Too few interrupts!", 1 );
	}

	test_pass( __FILE__ );

	return 0;
}
