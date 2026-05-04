/*
 * Test example for branch accuracy and functionality, originally
 * provided by Timothy Kaiser, SDSC. It was modified to fit the
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 * and Phil Mucci <mucci@cs.utk.edu>
 * This example verifies the accuracy of branch events
 */

/* Measures 4 events:
	PAPI_BR_NTK -- branches not taken
	PAPI_BR_PRC -- branches predicted correctly
	PAPI_BR_INS -- total branch instructions
	PAPI_BR_MSP -- branches mispredicted
  First measure all 4 at once (or as many as will fit).
  Then run them one by one.
  Compare results to see if they match.

  Note: sometimes have seen failure if system is under fuzzing load

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define MAXEVENTS 4
#define MINCOUNTS 100000
#define MPX_TOLERANCE .20

int
main( int argc, char **argv )
{
	PAPI_event_info_t info;
	int i, j, retval, errors=0;
	int iters = 10000000;
	double x = 1.1, y;
	long long t1, t2;
	long long values[MAXEVENTS], refvalues[MAXEVENTS];
	double spread[MAXEVENTS];
	int nevents = MAXEVENTS;
	int eventset = PAPI_NULL;
	int events[MAXEVENTS];
	int quiet;
	char event_names[MAXEVENTS][256] = {
		"PAPI_BR_NTK",	// not taken
		"PAPI_BR_PRC",	// predicted correctly
		"PAPI_BR_INS",	// total branches
		"PAPI_BR_MSP",	// branches mispredicted
	};

	/* Set quiet variable */
	quiet = tests_quiet( argc, argv );

	/* Parse command line args */
	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) ) {

		}
	}

	events[0] = PAPI_BR_NTK;	// not taken
	events[1] = PAPI_BR_PRC;	// predicted correctly
	events[2] = PAPI_BR_INS;	// total branches
	events[3] = PAPI_BR_MSP;	// branches mispredicted

	/* Why were these disabled?
	events[3]=PAPI_BR_CN;
	events[4]=PAPI_BR_UCN;
	events[5]=PAPI_BR_TKN; */


	/* Clear out the results to zero */
	for ( i = 0; i < MAXEVENTS; i++ ) {
		values[i] = 0;
	}

	if ( !quiet ) {
		printf( "\nAccuracy check of branch presets.\n" );
		printf( "Comparing a measurement with separate measurements.\n\n" );
	}

	/* Initialize library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create Eventset */
	retval = PAPI_create_eventset( &eventset );
	if ( retval ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

#ifdef MPX
	retval = PAPI_multiplex_init(  );
	if ( retval ) {
		test_fail( __FILE__, __LINE__, "PAPI_multiplex_init", retval );
	}

	retval = PAPI_set_multiplex( eventset );
	if ( retval ) {
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
	}
#endif

	nevents = 0;

	/* Add as many of the 4 events that exist on this machine */
	for ( i = 0; i < MAXEVENTS; i++ ) {
		if ( PAPI_query_event( events[i] ) != PAPI_OK )
			continue;
		if ( PAPI_add_event( eventset, events[i] ) == PAPI_OK ) {
			events[nevents] = events[i];
			nevents++;
		}
	}

	/* If none of the events can be added, skip this test */
	if ( nevents < 1 ) {
		test_skip( __FILE__, __LINE__, "Not enough events left...", 0 );
	}

	/* Find a reasonable number of iterations (each
	 * event active 20 times) during the measurement
	 */

	/* Target: 10000 usec/multiplex, 20 repeats */
	t2 = (long long)(10000 * 20) * nevents;

	if ( t2 > 30e6 ) {
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );
	}

	/* Measure one run */
	t1 = PAPI_get_real_usec(  );
	y = do_flops3( x, iters, 1 );
	t1 = PAPI_get_real_usec(  ) - t1;

	if ( t2 > t1 )			 /* Scale up execution time to match t2 */
		iters = iters * ( int ) ( t2 / t1 );
	else if ( t1 > 30e6 )	 /* Make sure execution time is < 30s per repeated test */
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );

	x = 1.0;

	/**********************************/
	/* First run: Grouped Measurement */
	/**********************************/
	if ( !quiet ) {
		printf( "\nFirst run: Together.\n" );
	}

	t1 = PAPI_get_real_usec(  );

	retval = PAPI_start( eventset );
	if (retval) test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	y = do_flops3( x, iters, 1 );

	retval = PAPI_stop( eventset, values );
	if (retval) test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	t2 = PAPI_get_real_usec(  );

	if ( !quiet ) {
		printf( "\tOperations= %.1f Mflop", y * 1e-6 );
		printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
		printf( "PAPI grouped measurement:\n" );

		for ( j = 0; j < nevents; j++ ) {
			PAPI_get_event_info( events[j], &info );
			printf( "%20s = ", info.short_descr );
			printf( LLDFMT, values[j] );
			printf( "\n" );
		}
		printf( "\n" );
	}

	/* Remove all the events, start again */
	retval = PAPI_remove_events( eventset, events, nevents );
	if (retval) test_fail( __FILE__, __LINE__, "PAPI_remove_events", retval );

	retval = PAPI_destroy_eventset( &eventset );
	if (retval) test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	/* Recreate eventset */
	eventset = PAPI_NULL;
	retval = PAPI_create_eventset( &eventset );
	if (retval) test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	/* Run events one by one */
	for ( i = 0; i < nevents; i++ ) {

		/* Clear out old event */
		retval = PAPI_cleanup_eventset( eventset );
		if (retval) test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );
		/* Add the event */
		retval = PAPI_add_event( eventset, events[i] );
		if (retval) test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

		x = 1.0;

		if ( !quiet ) {
			printf( "\nReference measurement %d (of %d):\n", i + 1, nevents );
		}

		t1 = PAPI_get_real_usec(  );

		retval = PAPI_start( eventset );
		if (retval) test_fail( __FILE__, __LINE__, "PAPI_start", retval );

		y = do_flops3( x, iters, 1 );

		retval = PAPI_stop( eventset, &refvalues[i] );
		if (retval) test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

		t2 = PAPI_get_real_usec(  );


		if ( !quiet ) {
			printf( "\tOperations= %.1f Mflop", y * 1e-6 );
			printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
			PAPI_get_event_info( events[i], &info );
			printf( "PAPI results:\n%20s = ", info.short_descr );
			printf( LLDFMT, refvalues[i] );
			printf( "\n" );
		}
	}

	if ( !quiet ) {
		printf( "\n" );
	}

	/* Validate the results */

	if ( !quiet ) {
		printf( "\n\nRelative accuracy:\n" );
		printf( "\tEvent\t\tGroup\t\tIndividual\tSpread\n");
	}

	for ( j = 0; j < nevents; j++ ) {
		spread[j] = abs( ( int ) ( refvalues[j] - values[j] ) );
		if ( values[j] )
			spread[j] /= ( double ) values[j];
		if ( !quiet ) {
			printf( "\t%02d: ",j);
			printf( "%s",event_names[j]);
			printf( "\t%10lld", values[j] );
			printf( "\t%10lld", refvalues[j] );
			printf("\t%10.3g\n", spread[j] );
		}

		/* Make sure that NaN get counted as errors */
		if ( spread[j] > MPX_TOLERANCE ) {

			/* Neglect inprecise results with low counts */
			if ( refvalues[j] < MINCOUNTS ) {
			}
			else {
				errors++;
				if (!quiet) {
					printf("\tError: Spread > %lf\n",MPX_TOLERANCE);
				}
			}
		}
	}
	if ( !quiet ) {
		printf( "\n\n" );
	}

	if ( errors ) {
		test_fail( __FILE__, __LINE__, "Values outside threshold", i );
	}

	test_pass( __FILE__ );

	return 0;
}


