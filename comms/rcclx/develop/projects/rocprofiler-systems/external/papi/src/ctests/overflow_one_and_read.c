/*
* File:    overflow_one_and_read.c : based on overflow_twoevents.c
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*          Kevin London
*          london@cs.utk.edu
*/

/* This file performs the following test: overflow dispatch on 1 counter.
 * In the handler read events.
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define OVER_FMT	"handler(%d) Overflow at %p! vector=%#llx\n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"

typedef struct
{
	long long mask;
	int count;
} ocount_t;

/* there are three possible vectors, one counter overflows, the other
   counter overflows, both overflow */
/*not used*/ ocount_t overflow_counts[3] = { {0, 0}, {0, 0}, {0, 0} };
/*not used*/ int total_unknown = 0;

/*added*/ long long dummyvalues[2];

void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	int retval;

	( void ) context;

	if ( !TESTS_QUIET ) {
		fprintf( stderr, OVER_FMT, EventSet, address, overflow_vector );
	}

	if ( ( retval = PAPI_read( EventSet, dummyvalues ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );

	if ( !TESTS_QUIET ) {
		fprintf( stderr, TWO12, dummyvalues[0], dummyvalues[1],
				 "(Reading  counters)\n" );
	}
	if ( dummyvalues[1] == 0 )
		test_fail( __FILE__, __LINE__, "Total Cycles == 0", 1 );
}

int
main( int argc, char **argv )
{
	int EventSet;
	long long **values = NULL;
	int retval;
	int PAPI_event;
	char event_name[PAPI_MAX_STR_LEN];
	int num_events1, mask1;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depends on the availability of the event on the
	   platform */
/* NOTE: Only adding one overflow on PAPI_event -- no overflow for PAPI_TOT_CYC*/
	EventSet = add_two_nonderived_events( &num_events1,
						&PAPI_event, &mask1 );
	if (num_events1==0) {
		if (!quiet) printf("Trouble adding events\n");
		test_skip(__FILE__,__LINE__,"Adding event",1);
	}

	values = allocate_test_space( 2, num_events1 );

	if ( ( retval =
		   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_overflow( EventSet, PAPI_event, THRESHOLD, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	remove_test_events( &EventSet, mask1 );


	if ( !TESTS_QUIET ) {
		printf
			( "Test case: Overflow dispatch of 1st event in set with 2 events.\n" );
		printf
			( "---------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", THRESHOLD );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-----------------------------------------------\n" );

		printf( "Test type    : %16d%16d\n", 1, 2 );
		printf( OUT_FMT, event_name, ( values[0] )[0], ( values[1] )[0] );
		printf( OUT_FMT, "PAPI_TOT_CYC", ( values[0] )[1], ( values[1] )[1] );
	}

	test_pass( __FILE__ );

	return 0;
}
