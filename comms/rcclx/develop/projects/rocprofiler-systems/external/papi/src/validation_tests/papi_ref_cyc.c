/* This test exercises the PAPI_TOT_CYC and PAPI_REF_CYC counters.

	PAPI_TOT_CYC should measure the number of cycles required to do
	a fixed amount of work.
	It should be roughly constant for constant work, regardless of the
	speed state a core is in.

	PAPI_REF_CYC should measure the number of cycles at a constant
	reference clock rate, independent of the actual clock rate of the core.
*/

/*
	PAPI_REF_CYC has various issues on Intel chips:

	On older machines PAPI uses UNHALTED_REFERENCE_CYCLES but this
	means different things on different architectures

	+ On Core2/Atom this maps to the special Fixed Counter 2
		CPU_CLK_UNHALTED.REF
		This counts at the same rate as the TSC (PAPI_get_real_cyc())
		And also seems to match PAPI_TOT_CYC
		It is documented as having a fixed ratio to the
		CPU_CLK_UNHALTED.BUS (3c/1) event.

	+ On Nehalem/Westemere this also maps to Fixed Counter 2.
		Again, counts same rate as the TSC  and returns
		CPU_CLK_UNHALTED.REF_P (3c/1)
		times the "Maximum Non-Turbo Ratio"

	+ Same for Sandybridge/Ivybridge

	On newer HSW,BDW,SKL machines PAPI uses a different type of event
	CPU_CLK_THREAD_UNHALTED:REF_XCLK

	+ On Haswell machines this is just the reference clock
		(100MHz?)
	+ On Sandybridge this is off by a factor of 8x?
*/

/* NOTE:
	PAPI_get_virt_cyc() returns a lie!
	It's just virt_time() * max_theoretical_MHz
	so no point in checking that */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define NUM_FLOPS  20000000

static void work (int EventSet, int sleep_test, int quiet)
{
	int retval;
	long long values[2];
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	double cycles_error;
	int numflops = NUM_FLOPS;

	/* Gather before stats */
	elapsed_us = PAPI_get_real_usec(  );
	elapsed_cyc = PAPI_get_real_cyc(  );
	elapsed_virt_us = PAPI_get_virt_usec(  );
	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our test code */
	if (sleep_test) {
		sleep(2);
	}
	else {
		do_flops( numflops, 1 );
	}

	/* Stop PAPI */
	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/* Calculate total values */
	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;
	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;
	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	if (!quiet) {
		printf( "-------------------------------------------------------------------------\n" );
		if (sleep_test) printf("Sleeping for 2s\n");
		else printf( "Using %d iterations of c += a*b\n", numflops );
		printf( "-------------------------------------------------------------------------\n" );

		printf( "PAPI_TOT_CYC             : \t%10lld\n", values[0] );
		printf( "PAPI_REF_CYC             : \t%10lld\n", values[1] );
		printf( "Real usec                : \t%10lld\n", elapsed_us );
		printf( "Real cycles              : \t%10lld\n", elapsed_cyc );
		printf( "Virt usec                : \t%10lld\n", elapsed_virt_us );
		printf( "Virt cycles (estimate)   : \t%10lld\n", elapsed_virt_cyc );
		printf( "Estimated GHz            : \t%10.3lf\n", (double) elapsed_cyc/(double)elapsed_us/1000.0);

		printf( "-------------------------------------------------------------------------\n" );
	}


	if (sleep_test) {
		if (!quiet) {
		printf( "Verification: PAPI_REF_CYC should be much lower than real_usec\n");
		}
		if (values[1]>elapsed_us) {
			if (!quiet) printf("PAPI_REF_CYC too high!\n");
			test_fail( __FILE__, __LINE__, "PAPI_REF_CYC too high", 0 );
		}

	}
	else {
		/* PAPI_REF_CYC should be roughly the same as TSC when busy */
		/* on Intel chips */
		if (!quiet) {
		printf( "Verification: real_cyc should be roughly PAPI_REF_CYC\n");
		printf( "              real_usec should be roughly virt_usec (on otherwise idle system)\n");
		}

		cycles_error=100.0*
			((double)values[1]-((double)elapsed_cyc))
				/values[1];

		if ((cycles_error>10.0) || (cycles_error<-10.0)) {
			if (!quiet) printf("Error of %.2f%%\n",cycles_error);
			test_fail( __FILE__, __LINE__, "PAPI_REF_CYC validation", 0 );
		}

		cycles_error=100.0*
			((double)elapsed_us-(double)elapsed_virt_us)
				/(double)elapsed_us;

		if ((cycles_error>10.0) || (cycles_error<-10.0)) {
			if (!quiet) printf("Error of %.2f%%\n",cycles_error);
			test_warn( __FILE__, __LINE__, "real_us validation", 0 );
		}
	}
}


int
main( int argc, char **argv )
{
	int retval;
	int EventSet = PAPI_NULL;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Check the ref cycles event */
	retval = PAPI_query_named_event("PAPI_REF_CYC");
	if (PAPI_OK!=retval) {
		if (!quiet) printf("No PAPI_REF_CYC available\n");
		test_skip( __FILE__, __LINE__,
			"PAPI_REF_CYC is not defined on this platform.", 0);
	}

	/* create an eventset */
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* add core cycle event */
	retval = PAPI_add_named_event( EventSet, "PAPI_TOT_CYC");
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__,
				"PAPI_add_named_event: PAPI_TOT_CYC", retval );
	}

	/* add ref cycle event */
	retval = PAPI_add_named_event( EventSet, "PAPI_REF_CYC");
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__,
				"PAPI_add_events: PAPI_REF_CYC", retval );
	}

	if (!quiet) {
		printf("Test case sleeping: "
			"Look at TOT and REF cycles.\n");
	}

	work(EventSet, 1, quiet);
//	do_flops(10*numflops);

	if (!quiet) {
		printf( "\nTest case busy:\n" );
	}

	work(EventSet, 0, quiet);

	test_pass( __FILE__ );

	return 0;
}

