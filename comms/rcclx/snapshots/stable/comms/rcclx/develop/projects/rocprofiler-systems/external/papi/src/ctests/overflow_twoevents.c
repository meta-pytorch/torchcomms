/*
* File:    overflow_twoevents.c
* Author:  min@cs.utk.edu
*          Min Zhou
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file performs the following test: overflow dispatch on 2 counters. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define OVER_FMT	"handler(%d) Overflow at %p! vector=%#llx\n"
#define OUT_FMT		"%-12s : %18lld%18lld%18lld\n"
#define VEC_FMT		"        at vector %#llx, event %-12s : %6d\n"

typedef struct
{
	long long mask;
	int count;
} ocount_t;

/* there are two experiments: batch and interleaf; for each experiment there
   are three possible vectors, one counter overflows, the other
   counter overflows, both overflow */
static ocount_t overflow_counts[2][3] =
	{ {{0, 0}, {0, 0}, {0, 0}}, {{0, 0}, {0, 0}, {0, 0}} };
static int total_unknown = 0;

static void
handler( int mode, void *address, long long overflow_vector, void *context )
{
	( void ) context;		 /*unused */
	int i;

	if ( !TESTS_QUIET ) {
		fprintf( stderr, OVER_FMT, mode, address, overflow_vector );
	}

	/* Look for the overflow_vector entry */

	for ( i = 0; i < 3; i++ ) {
		if ( overflow_counts[mode][i].mask == overflow_vector ) {
			overflow_counts[mode][i].count++;
			return;
		}
	}

	/* Didn't find it so add it. */

	for ( i = 0; i < 3; i++ ) {
		if ( overflow_counts[mode][i].mask == ( long long ) 0 ) {
			overflow_counts[mode][i].mask = overflow_vector;
			overflow_counts[mode][i].count = 1;
			return;
		}
	}

	/* Unknown entry!?! */

	total_unknown++;
}

static void
handler_batch( int EventSet, void *address, long long overflow_vector,
			   void *context )
{
	( void ) EventSet;		 /*unused */
	handler( 0, address, overflow_vector, context );
}

static void
handler_interleaf( int EventSet, void *address, long long overflow_vector,
				   void *context )
{
	( void ) EventSet;		 /*unused */
	handler( 1, address, overflow_vector, context );
}


int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	long long ( values[3] )[2];
	int retval;
	int PAPI_event, k, idx[4];
	char event_name[3][PAPI_MAX_STR_LEN];
	int num_events1;
	int threshold = THRESHOLD;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* decide which of PAPI_FP_INS, PAPI_FP_OPS or PAPI_TOT_INS to add,
	   depending on the availability and derived status of the event on
	   this platform */
	if ( ( PAPI_event = find_nonderived_event(  ) ) == 0 ) {
		if (!quiet) printf("No events found!\n");
		test_skip( __FILE__, __LINE__, "no PAPI_event", 0 );
	}

	if ( ( retval = PAPI_add_event( EventSet, PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	if ( ( retval = PAPI_add_event( EventSet, PAPI_TOT_CYC ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_stop( EventSet, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	/* Set both overflows after adding both events (batch) */
	retval = PAPI_overflow( EventSet, PAPI_event, threshold, 0,
						  handler_batch );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	retval = PAPI_overflow( EventSet, PAPI_TOT_CYC, threshold, 0,
						  handler_batch );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	num_events1 = 1;
	retval =
		PAPI_get_overflow_event_index( EventSet, 1, &idx[0], &num_events1 );
	if ( retval != PAPI_OK ) {
		printf( "PAPI_get_overflow_event_index error: %s\n",
				PAPI_strerror( retval ) );
	}

	num_events1 = 1;
	retval =
		PAPI_get_overflow_event_index( EventSet, 2, &idx[1], &num_events1 );
	if ( retval != PAPI_OK ) {
		printf( "PAPI_get_overflow_event_index error: %s\n",
				PAPI_strerror( retval ) );
	}

	if ( ( retval = PAPI_cleanup_eventset( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	/* Add each event and set its overflow (interleaved) */
	if ( ( retval = PAPI_add_event( EventSet, PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	if ( ( retval =
		   PAPI_overflow( EventSet, PAPI_event, threshold, 0,
						  handler_interleaf ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	if ( ( retval = PAPI_add_event( EventSet, PAPI_TOT_CYC ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	if ( ( retval =
		   PAPI_overflow( EventSet, PAPI_TOT_CYC, threshold, 0,
						  handler_interleaf ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_stop( EventSet, values[2] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	num_events1 = 1;
	retval =
		PAPI_get_overflow_event_index( EventSet, 1, &idx[2], &num_events1 );
	if ( retval != PAPI_OK ) {
		printf( "PAPI_get_overflow_event_index error: %s\n",
				PAPI_strerror( retval ) );
	}

	num_events1 = 1;
	retval = PAPI_get_overflow_event_index( EventSet, 2, &idx[3],
								&num_events1 );
	if ( retval != PAPI_OK ) {
		printf( "PAPI_get_overflow_event_index error: %s\n",
				PAPI_strerror( retval ) );
	}

	if ( ( retval = PAPI_cleanup_eventset( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_event_code_to_name( PAPI_event, event_name[0] );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	retval = PAPI_event_code_to_name( PAPI_TOT_CYC, event_name[1] );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	strcpy( event_name[2], "Unknown" );

	if (!TESTS_QUIET) {
		printf( "Test case: Overflow dispatch of both events in set with 2 events.\n" );
		printf( "---------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", threshold );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-----------------------------------------------\n" );

		printf( "Test type    : %18s%18s%18s\n", "1 (no overflow)", "2 (batch)",
			"3 (interleaf)" );
		printf( OUT_FMT, event_name[0], ( values[0] )[0], ( values[1] )[0],
			( values[2] )[0] );
		printf( OUT_FMT, event_name[1], ( values[0] )[1], ( values[1] )[1],
			( values[2] )[1] );
		printf( "\n" );

		printf( "Predicted overflows at event %-12s : %6d\n", event_name[0],
			( int ) ( ( values[0] )[0] / threshold ) );
		printf( "Predicted overflows at event %-12s : %6d\n", event_name[1],
			( int ) ( ( values[0] )[1] / threshold ) );

		printf( "\nBatch overflows (add, add, over, over):\n" );
		for ( k = 0; k < 2; k++ ) {
			if ( overflow_counts[0][k].mask ) {
				printf( VEC_FMT, ( long long ) overflow_counts[0][k].mask,
					event_name[idx[k]], overflow_counts[0][k].count );
			}
		}

		printf( "\nInterleaved overflows (add, over, add, over):\n" );
		for ( k = 0; k < 2; k++ ) {
			if ( overflow_counts[1][k].mask )
				printf( VEC_FMT,
					( long long ) overflow_counts[1][k].mask,
					event_name[idx[k + 2]],
					overflow_counts[1][k].count );
		}

		printf( "\nCases 2+3 Unknown overflows: %d\n", total_unknown );
		printf( "-----------------------------------------------\n" );
	}

	if ( overflow_counts[0][0].count == 0 || overflow_counts[0][1].count == 0 )
		test_fail( __FILE__, __LINE__, "a batch counter had no overflows", 1 );

	if ( overflow_counts[1][0].count == 0 || overflow_counts[1][1].count == 0 )
		test_fail( __FILE__, __LINE__,
				   "an interleaved counter had no overflows", 1 );

	if ( total_unknown > 0 )
		test_fail( __FILE__, __LINE__, "Unknown counter had overflows", 1 );

	test_pass( __FILE__ );

	return 0;
}

