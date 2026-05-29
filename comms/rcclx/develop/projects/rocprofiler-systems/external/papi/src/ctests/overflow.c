/*
* File:    overflow.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file performs the following test: overflow dispatch

     The Eventset contains:
     + PAPI_TOT_CYC
     + PAPI_FP_INS (overflow monitor)

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

#define OVER_FMT	"handler(%d ) Overflow at %p! bit=%#llx \n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"

static int total = 0;				   /* total overflows */


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
	int num_flops = NUM_FLOPS, retval;
	int PAPI_event, mythreshold = THRESHOLD;
	char event_name1[PAPI_MAX_STR_LEN];
	const PAPI_hw_info_t *hw_info = NULL;
	int num_events, mask;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Get hardware info */
	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	/* add PAPI_TOT_CYC and one of the events in     */
        /*     PAPI_FP_INS, PAPI_FP_OPS or PAPI_TOT_INS, */
	/* depending on the availability of the event on */
	/* the platform */
	EventSet = add_two_nonderived_events( &num_events, &PAPI_event, &mask );

	if (num_events==0) {
		if (!quiet) printf("Trouble adding event!\n");
		test_skip(__FILE__,__LINE__,"Event add",1);
	}

	if (!quiet) {
		printf("Using %#x for the overflow event\n",PAPI_event);
	}

	if ( PAPI_event == PAPI_FP_INS ) {
		mythreshold = THRESHOLD;
	}
	else {
#if defined(linux)
		mythreshold = ( int ) hw_info->cpu_max_mhz * 20000;
#else
		mythreshold = THRESHOLD * 2;
#endif
	}

	/* Start the run calibration run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	/* stop the calibration run */
	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );


	/* set up overflow handler */
	retval = PAPI_overflow( EventSet, PAPI_event, mythreshold, 0, handler );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	/* Start overflow run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( num_flops );

	/* stop overflow run */
	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	if ( !TESTS_QUIET ) {
		retval = PAPI_event_code_to_name( PAPI_event, event_name1 );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
		}

		printf( "Test case: Overflow dispatch of 2nd event in set with 2 events.\n" );
		printf( "---------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", mythreshold );
		printf( "Using %d iterations of c += a*b\n", num_flops );
		printf( "-----------------------------------------------\n" );

		printf( "Test type    : %16d%16d\n", 1, 2 );
		printf( OUT_FMT, event_name1, ( values[0] )[1], ( values[1] )[1] );
		printf( OUT_FMT, "PAPI_TOT_CYC", ( values[0] )[0], ( values[1] )[0] );
		printf( "Overflows    : %16s%16d\n", "", total );
		printf( "-----------------------------------------------\n" );
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if ( !TESTS_QUIET ) {
		printf( "Verification:\n" );
#if defined(linux) || defined(__ia64__) || defined(_POWER4)
		num_flops *= 2;
#endif
		if ( PAPI_event == PAPI_FP_INS || PAPI_event == PAPI_FP_OPS ) {
			printf( "Row 1 approximately equals %d %d\n", num_flops,
					num_flops );
		}
		printf( "Column 1 approximately equals column 2\n" );
		printf( "Row 3 approximately equals %u +- %u %%\n",
				( unsigned ) ( ( values[0] )[1] / ( long long ) mythreshold ),
				( unsigned ) ( OVR_TOLERANCE * 100.0 ) );
	}
/*
  min = (long long)((values[0])[1]*(1.0-TOLERANCE));
  max = (long long)((values[0])[1]*(1.0+TOLERANCE));
  if ( (values[0])[1] > max || (values[0])[1] < min )
  	test_fail(__FILE__, __LINE__, event_name, 1);
*/

	min =
		( long long ) ( ( ( double ) values[0][1] * ( 1.0 - OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	max =
		( long long ) ( ( ( double ) values[0][1] * ( 1.0 + OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	if (!quiet) {
		printf( "Overflows: total(%d) > max(%lld) || "
			"total(%d) < min(%lld) \n", total, max, total, min );
	}
	if ( total > max || total < min ) {
		test_fail( __FILE__, __LINE__, "Overflows", 1 );
	}

	test_pass( __FILE__ );
	return 0;

}
