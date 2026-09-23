/*
* File:    overflow.c
* Author:  Nils Smeds  [Based on tests/overflow.c by Philip Mucci]
*          smeds@pdc.kth.se
*/

/* This file performs the following test: overflow dispatch

     The Eventset contains:
     + PAPI_TOT_CYC (overflow monitor)
     + PAPI_FP_INS

   - Start eventset 1
   - Do flops
   - Stop and measure eventset 1
   - Set up overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define OVER_FMT    "handler(%d ) Overflow at %p! bit=%#llx \n"
#define OUT_FMT     "%-12s : %16lld%16lld\n"

int total = 0;						   /* total overflows */

void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	( void ) context;

	if ( !TESTS_QUIET ) {
		fprintf( stderr, OVER_FMT, EventSet, address, overflow_vector );
	}
	total++;
}

int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	long long ( values[2] )[2];
	long long min, max;
	int num_flops, retval;
	int PAPI_event, mythreshold;
	char event_name[PAPI_MAX_STR_LEN];
	const PAPI_hw_info_t *hw_info = NULL;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

#if defined(POWER3) || defined(__sparc__)
	PAPI_event = PAPI_TOT_INS;
#else
	/* query and set up the right instruction to monitor */
	PAPI_event = find_nonderived_event( );
#endif

	if (PAPI_event==0) {
		if (!quiet) printf("Trouble creating events\n");
		test_skip(__FILE__,__LINE__,"Creating event",1);
	}

	if (( PAPI_event == PAPI_FP_OPS ) || ( PAPI_event == PAPI_FP_INS ))
		mythreshold = THRESHOLD;
	else
#if defined(linux)
		mythreshold = ( int ) hw_info->cpu_max_mhz * 10000 * 2;
#else
		mythreshold = THRESHOLD * 2;
#endif

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	retval = PAPI_add_event( EventSet, PAPI_event );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	retval = PAPI_add_event( EventSet, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_overflow( EventSet, PAPI_event, mythreshold, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	num_flops = NUM_FLOPS;
#if defined(linux) || defined(__ia64__) || defined(_POWER4)
	num_flops *= 2;
#endif

	if ( !quiet ) {
		if ( ( retval =
			   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

		printf
			( "Test case: Overflow dispatch of 1st event in set with 2 events.\n" );
		printf
			( "---------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", mythreshold );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-----------------------------------------------\n" );

		printf( "Test type    : %16d%16d\n", 1, 2 );
		printf( OUT_FMT, event_name, ( values[0] )[0], ( values[1] )[0] );
		printf( OUT_FMT, "PAPI_TOT_CYC", ( values[0] )[1], ( values[1] )[1] );
		printf( "Overflows    : %16s%16d\n", "", total );
		printf( "-----------------------------------------------\n" );

		printf( "Verification:\n" );
/*
	if (PAPI_event == PAPI_FP_INS)
		printf("Row 1 approximately equals %d %d\n", num_flops, num_flops);
*/
		/* Note that the second run prints output on stdout. On some systems
		 * this is costly. PAPI_TOT_INS or PAPI_TOT_CYC are likely to be _very_
		 * different between the two runs.
		 * printf("Column 1 approximately equals column 2\n"); 
		 */
		printf( "Row 3 approximately equals %u +- %u %%\n",
				( unsigned ) ( ( values[0] )[0] / ( long long ) mythreshold ),
				( unsigned ) ( OVR_TOLERANCE * 100.0 ) );
	}
/*
  min = (long long)((values[0])[0]*(1.0-TOLERANCE));
  max = (long long)((values[0])[0]*(1.0+TOLERANCE));
  if ( (values[1])[0] > max || (values[1])[0] < min )
  	test_fail(__FILE__, __LINE__, event_name, 1);
*/

	min =
		( long long ) ( ( ( double ) values[0][0] * ( 1.0 - OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	max =
		( long long ) ( ( ( double ) values[0][0] * ( 1.0 + OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	if ( total > max || total < min )
		test_fail( __FILE__, __LINE__, "Overflows", 1 );

	test_pass( __FILE__ );

	return 0;
}
