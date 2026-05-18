/*
 * Test example for multiplex functionality, originally
 * provided by Timothy Kaiser, SDSC. It was modified to fit the
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 *
 * This example verifies the accuracy of multiplexed events
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define REPEATS 5
#define MAXEVENTS 14
#define SLEEPTIME 100
#define MINCOUNTS 100000
#define MPX_TOLERANCE   0.20
#define NUM_FLOPS  20000000

void
check_values( int eventset, int *events, int nevents, long long *values,
			  long long *refvalues )
{
	double spread[MAXEVENTS];
	int i = nevents, j = 0;

	if ( !TESTS_QUIET ) {
		printf( "\nRelative accuracy:\n" );
		for ( j = 0; j < nevents; j++ )
			printf( "   Event %.2d", j + 1 );
		printf( "\n" );
	}

	for ( j = 0; j < nevents; j++ ) {
	  spread[j] = abs( (int) ( refvalues[j] - values[j] ) );
		if ( values[j] )
			spread[j] /= ( double ) values[j];
		if ( !TESTS_QUIET )
			printf( "%10.3g ", spread[j] );
		/* Make sure that NaN get counted as errors */
		if ( spread[j] < MPX_TOLERANCE ) {
			i--;
		}
		else if ( refvalues[j] < MINCOUNTS ) {	/* Neglect inprecise results with low counts */
			i--;
		}
                else {
                  char buff[BUFSIZ];
		if (!TESTS_QUIET) {
		  printf("reference = %lld,  value = %lld,  diff = %lld\n",
			 refvalues[j],values[j],refvalues[j] - values[j]  );
		}
		  sprintf(buff,"Error on %d, spread %lf > threshold %lf AND count %lld > minimum size threshold %d\n",j,spread[j],MPX_TOLERANCE,
			 refvalues[j],MINCOUNTS);

		  test_fail( __FILE__, __LINE__, buff, 1 );
		}
	}
	if (!TESTS_QUIET) printf( "\n\n" );
#if 0
	if ( !TESTS_QUIET ) {
		for ( j = 0; j < nevents; j++ ) {
			PAPI_get_event_info( events[j], &info );
			printf( "Event %.2d: ref=", j );
			printf( LLDFMT10, refvalues[j] );
			printf( ", diff/ref=%7.2g  -- %s\n", spread[j], info.short_descr );
			printf( "\n" );
		}
		printf( "\n" );
	}
#else
	( void ) eventset;
	( void ) events;
#endif


}

void
ref_measurements( int iters, int *eventset, int *events, int nevents,
				  long long *refvalues )
{
	PAPI_event_info_t info;
	int i, retval;
	double x = 1.1, y;
	long long t1, t2;

	if (!TESTS_QUIET) printf( "PAPI reference measurements:\n" );

	if ( ( retval = PAPI_create_eventset( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	for ( i = 0; i < nevents; i++ ) {
		if ( ( retval = PAPI_add_event( *eventset, events[i] ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

		x = 1.0;

		t1 = PAPI_get_real_usec(  );
		if ( ( retval = PAPI_start( *eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
		y = do_flops3( x, iters, 1 );
		if ( ( retval = PAPI_stop( *eventset, &refvalues[i] ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		t2 = PAPI_get_real_usec(  );

		if (!TESTS_QUIET) {
		   printf( "\tOperations= %.1f Mflop", y * 1e-6 );
		   printf( "\t(%g Mflop/s)\n\n", ( ( float ) y / ( t2 - t1 ) ) );
		}

		PAPI_get_event_info( events[i], &info );
		if (!TESTS_QUIET) {
			printf( "%20s = ", info.short_descr );
			printf( LLDFMT, refvalues[i] );
			printf( "\n" );
		}

		if ( ( retval = PAPI_cleanup_eventset( *eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );
	}
	if ( ( retval = PAPI_destroy_eventset( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );
	*eventset = PAPI_NULL;
}



int
main( int argc, char **argv )
{
	PAPI_event_info_t info;
	int i, j, retval;
	int iters = NUM_FLOPS;
	double x = 1.1, y;
	long long t1, t2;
	long long values[MAXEVENTS], refvalues[MAXEVENTS];
	int sleep_time = SLEEPTIME;
	int nevents = MAXEVENTS;
	int eventset = PAPI_NULL;
	int events[MAXEVENTS];
	int quiet;

	quiet = tests_quiet( argc, argv );

	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) ) {
		}
		else {
			sleep_time = atoi( argv[1] );
			if ( sleep_time <= 0 )
				sleep_time = SLEEPTIME;
		}
	}

	events[0] = PAPI_FP_INS;
	events[1] = PAPI_TOT_INS;
	events[2] = PAPI_INT_INS;
	events[3] = PAPI_TOT_CYC;
	events[4] = PAPI_STL_CCY;
	events[5] = PAPI_BR_INS;
	events[6] = PAPI_SR_INS;
	events[7] = PAPI_LD_INS;
	events[8] = PAPI_TOT_IIS;
	events[9] = PAPI_FAD_INS;
	events[10] = PAPI_BR_TKN;
	events[11] = PAPI_BR_MSP;
	events[12] = PAPI_L1_ICA;
	events[13] = PAPI_L1_DCA;

	for ( i = 0; i < MAXEVENTS; i++ ) {
		values[i] = 0;
	}


	if ( !quiet ) {
		printf( "\nAccuracy check of multiplexing routines.\n" );
		printf( "Comparing a multiplex measurement with separate measurements.\n\n" );
	}

	/* Initialize PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Iterate through event list and remove those that aren't suitable */
	nevents = MAXEVENTS;
	for ( i = 0; i < nevents; i++ ) {
		if (( PAPI_get_event_info( events[i], &info ) == PAPI_OK ) &&
		    (info.count && (strcmp( info.derived, "NOT_DERIVED")==0))) {
			if (!quiet) printf( "Added %s\n", info.symbol );
		}
		else {
			for ( j = i; j < MAXEVENTS-1; j++ ) {
				events[j] = events[j + 1];
			}
			nevents--;
			i--;
		}
	}

	/* Skip test if not enough events available */
	if ( nevents < 2 ) {
		test_skip( __FILE__, __LINE__, "Not enough events to multiplex...", 0 );
	}

	if (!quiet) printf( "Using %d events\n\n", nevents );

	retval = PAPI_multiplex_init(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__,
				"PAPI multiplex init fail\n", retval );
	}

	/* Find a reasonable number of iterations (each
	 * event active 20 times) during the measurement
	 */
	/* Target: 10000 usec/multiplex, 20 repeats */
	t2 = 10000 * 20 * nevents;
	if ( t2 > 30e6 ) {
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );
	}

	/* Warmup? */
	y = do_flops3( x, iters, 1 );

	/* Measure time of one run */
	t1 = PAPI_get_real_usec(  );
	y = do_flops3( x, iters, 1 );
	t1 = PAPI_get_real_usec(  ) - t1;

	if (t1==0) {
		test_fail(__FILE__, __LINE__,
			"do_flops3 takes no time to run!\n", retval);
	}

	/* Scale up execution time to match t2 */
	if ( t2 > t1 ) {
		iters = iters * ( int ) ( t2 / t1 );
		if (!quiet) {
			printf( "Modified iteration count to %d\n\n", iters );
		}
	}

	if (!quiet) fprintf(stdout,"y=%lf\n",y);

	/* Now loop through the items one at a time */

	ref_measurements( iters, &eventset, events, nevents, refvalues );

	/* Now check multiplexed */

	if ( ( retval = PAPI_create_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );


	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( eventset, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

	if ( ( retval = PAPI_set_multiplex( eventset ) ) ) {
	   if ( retval == PAPI_ENOSUPP) {
	      test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	   }

		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
	}

	if ( ( retval = PAPI_add_events( eventset, events, nevents ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_add_events", retval );

	if (!quiet) printf( "\nPAPI multiplexed measurements:\n" );
	x = 1.0;
	t1 = PAPI_get_real_usec(  );
	if ( ( retval = PAPI_start( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	y = do_flops3( x, iters, 1 );
	if ( ( retval = PAPI_stop( eventset, values ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	t2 = PAPI_get_real_usec(  );

	for ( j = 0; j < nevents; j++ ) {
		PAPI_get_event_info( events[j], &info );
		if ( !quiet ) {
			printf( "%20s = ", info.short_descr );
			printf( LLDFMT, values[j] );
			printf( "\n" );
		}
	}

	check_values( eventset, events, nevents, values, refvalues );

	if ( ( retval = PAPI_remove_events( eventset, events, nevents ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_remove_events", retval );
	if ( ( retval = PAPI_cleanup_eventset( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );
	if ( ( retval = PAPI_destroy_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );
	eventset = PAPI_NULL;

	/* Now loop through the items one at a time */

	ref_measurements( iters, &eventset, events, nevents, refvalues );

	check_values( eventset, events, nevents, values, refvalues );

	test_pass( __FILE__ );

	return 0;
}


