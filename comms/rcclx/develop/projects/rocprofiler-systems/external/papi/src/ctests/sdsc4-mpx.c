/*
 * Test example for multiplex functionality, originally
 * provided by Timothy Kaiser, SDSC. It was modified to fit the
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 *
 * This example verifies the adding and removal of multiplexed
 * events in an event set.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define MAXEVENTS 9
#define REPEATS (MAXEVENTS * 4)
#define SLEEPTIME 100
#define MINCOUNTS 100000
#define MPX_TOLERANCE	0.20
#define NUM_FLOPS  20000000

int
main( int argc, char **argv )
{
	PAPI_event_info_t info;
	char name2[PAPI_MAX_STR_LEN];
	int i, j, retval, idx, repeats;
	int iters = NUM_FLOPS;
	double x = 1.1, y, dtmp;
	long long t1, t2;
	long long values[MAXEVENTS], refvals[MAXEVENTS];
	int nsamples[MAXEVENTS], truelist[MAXEVENTS], ntrue;
#ifdef STARTSTOP
	long long dummies[MAXEVENTS];
#endif
	int sleep_time = SLEEPTIME;
	double valsample[MAXEVENTS][REPEATS];
	double valsum[MAXEVENTS];
	double avg[MAXEVENTS];
	double spread[MAXEVENTS];
	int nevents = MAXEVENTS, nev1;
	int eventset = PAPI_NULL;
	int events[MAXEVENTS];
	int eventidx[MAXEVENTS];
	int eventmap[MAXEVENTS];
	int fails;
	int quiet;

	quiet =	tests_quiet( argc, argv );

	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "quiet" ) ) {
		}
		else {
			sleep_time = atoi( argv[1] );
			if ( sleep_time <= 0 )
				sleep_time = SLEEPTIME;
		}
	}

	events[0] = PAPI_FP_INS;
	events[1] = PAPI_TOT_CYC;
	events[2] = PAPI_TOT_INS;
	events[3] = PAPI_TOT_IIS;
	events[4] = PAPI_INT_INS;
	events[5] = PAPI_STL_CCY;
	events[6] = PAPI_BR_INS;
	events[7] = PAPI_SR_INS;
	events[8] = PAPI_LD_INS;

	for ( i = 0; i < MAXEVENTS; i++ ) {
		values[i] = 0;
		valsum[i] = 0;
		nsamples[i] = 0;
	}

	/* Print test summary */
	if ( !quiet ) {
		printf( "\nFunctional check of multiplexing routines.\n" );
		printf( "Adding and removing events from an event set.\n\n" );
	}

	/* Init the library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Enable multiplexing */
#ifdef MPX
	retval = PAPI_multiplex_init(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI multiplex init fail\n", retval );
	}
#endif

	/* Create an eventset */
	if ( ( retval = PAPI_create_eventset( &eventset ) ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* Enable multiplexing on the eventset */
#ifdef MPX

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
#endif

	/* See which events are available and remove the ones that aren't */
	nevents = MAXEVENTS;
	for ( i = 0; i < nevents; i++ ) {
		if ( ( retval = PAPI_add_event( eventset, events[i] ) ) ) {
			for ( j = i; j < MAXEVENTS-1; j++ )
				events[j] = events[j + 1];
			nevents--;
			i--;
		}
	}

	/* We want at least three events? */
	/* Seems arbitrary.  Might be because intel machines used to */
	/* Only have two event slots */
	if ( nevents < 3 ) {
		test_skip( __FILE__, __LINE__, "Not enough events left...", 0 );
	}

	/* Find a reasonable number of iterations (each
	 * event active 20 times) during the measurement
	 */

	/* TODO: find Linux multiplex interval */
	/*       not sure if 10ms is close or not */
	/* Target: 10000 usec/multiplex, 20 repeats */
	t2 = 10000 * 20 * nevents;
	if ( t2 > 30e6 ) {
		test_skip( __FILE__, __LINE__,
				"This test takes too much time", retval );
	}

	/* Measure one run */
	t1 = PAPI_get_real_usec(  );
	y = do_flops3( x, iters, 1 );
	t1 = PAPI_get_real_usec(  ) - t1;

	/* Scale up execution time to match t2 */
	if ( t2 > t1 ) {
		iters = iters * ( int ) ( t2 / t1 );
	}
	/* Make sure execution time is < 30s per repeated test */
	else if ( t1 > 30e6 ) {
		test_skip( __FILE__, __LINE__,
				"This test takes too much time", retval );
	}

	/* Split the events up by odd and even? */
	j = nevents;
	for ( i = 1; i < nevents; i = i + 2 )
		eventidx[--j] = i;
	for ( i = 0; i < nevents; i = i + 2 )
		eventidx[--j] = i;
	assert( j == 0 );

	/* put event mapping in eventmap? */
	for ( i = 0; i < nevents; i++ )
		eventmap[i] = i;

	x = 1.0;

	/* Make a reference run */
	if ( !quiet ) {
		printf( "\nReference run:\n" );
	}

	t1 = PAPI_get_real_usec(  );
	if ( ( retval = PAPI_start( eventset ) ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}
	y = do_flops3( x, iters, 1 );
	PAPI_read( eventset, refvals );
	t2 = PAPI_get_real_usec(  );

	/* Print results */
	ntrue = nevents;
	PAPI_list_events( eventset, truelist, &ntrue );
	if ( !quiet ) {
		printf( "\tOperations= %.1f Mflop", y * 1e-6 );
		printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
		printf( "%20s   %16s   %-15s %-15s\n", "PAPI measurement:",
				"Acquired count", "Expected event", "PAPI_list_events" );

		for ( j = 0; j < nevents; j++ ) {
			PAPI_get_event_info( events[j], &info );
			PAPI_event_code_to_name( truelist[j], name2 );
			printf( "%20s = %16lld   %-15s %-15s %s\n",
				info.short_descr, refvals[j],
				info.symbol, name2,
				strcmp( info.symbol,name2 ) ?
					"*** MISMATCH ***" : "" );
		}
		printf( "\n" );
	}

	/* Make repeated runs while removing/readding events */

	nev1 = nevents;
	repeats = nevents * 4;

	/* Repeat four times for each event? */

	for ( i = 0; i < repeats; i++ ) {

		/* What's going on here?  as example, nevents=4, repeats=16*/
		/* 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 == i*/
		/* 0 1 2 3 0 1 2 3 0 1  2  3  0  1  2  3 == i%nevents */
		/* 1 2 3 4 1 2 3 4 1 2  3  4  1  2  3  4 == (i%nevents)+1 */
		/* 0 0 0 1 0 0 0 1 0 0  0  1  0  0  0  1 */
		/* so we skip nevery NEVENTS time through the loop? */
		if ( ( i % nevents ) + 1 == nevents ) continue;

		if ( !quiet ) {
			printf( "\nTest %d (of %d):\n",
				i + 1 - (i / nevents), repeats - 4 );
		}

		/* Stop the counter, it's been left running */
		if ( ( retval = PAPI_stop( eventset, values ) ) ) {
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		}

		/* We run through a 4-way pattern */
		/* 1st quarter, remove events */
		/* 2nd quarter, add back events */
		/* 3rd quarter, remove events again */
		/* 4th wuarter, re-add events */
		j = eventidx[i % nevents];
		if ( ( i / nevents ) % 2 == 0 ) {

			/* Remove event */
			PAPI_get_event_info( events[j], &info );
			if ( !quiet ) {
				printf( "Removing event[%d]: %s\n",
							j, info.short_descr );
			}

			retval = PAPI_remove_event( eventset, events[j] );
			if (retval != PAPI_OK ) {
				test_fail( __FILE__, __LINE__,
						"PAPI_remove_event", retval );
			}

			/* Update the complex event mapping */
			nev1--;
			for ( idx = 0; eventmap[idx] != j; idx++ );
			for ( j = idx; j < nev1; j++ )
				eventmap[j] = eventmap[j + 1];
		} else {

			/* Add an event back in */
			PAPI_get_event_info( events[j], &info );
			if ( !quiet ) {
				printf( "Adding event[%d]: %s\n",
						j, info.short_descr );
			}
			retval = PAPI_add_event( eventset, events[j] );
			if (retval != PAPI_OK ) {
				test_fail( __FILE__, __LINE__,
						"PAPI_add_event", retval );
			}

			eventmap[nev1] = j;
			nev1++;
		}

		if ( ( retval = PAPI_start( eventset ) ) ) {
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
		}

		x = 1.0;

		// This startstop is leftover from sdsc2? */
#ifndef STARTSTOP
		if ( ( retval = PAPI_reset( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_reset", retval );
#else
		if ( ( retval = PAPI_stop( eventset, dummies ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		if ( ( retval = PAPI_start( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
#endif

		/* Run the actual workload */
		t1 = PAPI_get_real_usec(  );
		y = do_flops3( x, iters, 1 );
		PAPI_read( eventset, values );
		t2 = PAPI_get_real_usec(  );

		/* Print approximate flops plus header */
		if ( !quiet ) {
			printf( "\n(calculated independent of PAPI)\n" );
			printf( "\tOperations= %.1f Mflop", y * 1e-6 );
			printf( "\t(%g Mflop/s)\n\n",
					( y / ( double ) ( t2 - t1 ) ) );

			printf( "%20s   %16s   %-15s %-15s\n",
				"PAPI measurement:",
				"Acquired count",
				"Expected event",
				"PAPI_list_events" );


			ntrue = nev1;
			PAPI_list_events( eventset, truelist, &ntrue );
			for ( j = 0; j < nev1; j++ ) {
				idx = eventmap[j];
				/* printf("Mapping: Counter %d -> slot %d.\n",j,idx); */
				PAPI_get_event_info( events[idx], &info );
				PAPI_event_code_to_name( truelist[j], name2 );
				printf( "%20s = %16lld   %-15s %-15s %s\n",
					info.short_descr, values[j],
					info.symbol, name2,
					strcmp( info.symbol, name2 ) ?
						"*** MISMATCH ***" : "" );
			}
			printf( "\n" );
		}

		/* Calculate results */
		for ( j = 0; j < nev1; j++ ) {
			idx = eventmap[j];
			dtmp = ( double ) values[j];
			valsum[idx] += dtmp;
			valsample[idx][nsamples[idx]] = dtmp;
			nsamples[idx]++;
		}
	}

	/* Stop event for good */
	if ( ( retval = PAPI_stop( eventset, values ) ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if ( !quiet ) {
		printf( "\n\nEstimated variance relative "
			"to average counts:\n" );
		for ( j = 0; j < nev1; j++ ) {
			printf( "   Event %.2d", j );
		}
		printf( "\n" );
	}

	fails = nevents;

	/* Due to limited precision of floating point cannot really use
	   typical standard deviation compuation for large numbers with
	   very small variations. Instead compute the std devation
	   problems with precision.
	 */

	/* Update so that if our event count is small (<1000 or so) */
	/* Then don't fail with high variation.  Since we're multiplexing */
	/* it's hard to capture such small counts, and it makes the test */
	/* fail on machines such as Haswell and the PAPI_SR_INS event */

	for ( j = 0; j < nev1; j++ ) {

		avg[j] = valsum[j] / nsamples[j];
		spread[j] = 0;
		for ( i = 0; i < nsamples[j]; ++i ) {
			double diff = ( valsample[j][i] - avg[j] );
			spread[j] += diff * diff;
		}
		spread[j] = sqrt( spread[j] / nsamples[j] ) / avg[j];
		if ( !quiet ) {
			printf( "%9.2g  ", spread[j] );
		}
	}

	for ( j = 0; j < nev1; j++ ) {

		/* Make sure that NaN get counted as errors */
		if ( spread[j] < MPX_TOLERANCE ) {
			if (!quiet) printf("Event %d tolerance good\n",j);
			fails--;
		}
		/* Neglect inprecise results with low counts */
		else if ( avg[j] < MINCOUNTS ) {
			if (!quiet) printf("Event %d too small to fail\n",j);
			fails--;
		}
		else {
			if (!quiet) printf("Event %d failed!\n",j);
		}
	}

	if ( !quiet ) {
		printf( "\n\n" );
		for ( j = 0; j < nev1; j++ ) {
			PAPI_get_event_info( events[j], &info );
			printf( "Event %.2d: mean=%10.0f, "
				"sdev/mean=%7.2g nrpt=%2d -- %s\n",
				j, avg[j], spread[j],
				nsamples[j], info.short_descr );
		}
		printf( "\n\n" );
	}

	if ( fails ) {
		test_fail( __FILE__, __LINE__, "Values differ from reference", fails );
	}

	test_pass( __FILE__ );

	return 0;
}
