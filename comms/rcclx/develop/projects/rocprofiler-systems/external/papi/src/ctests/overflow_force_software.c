/*
* File: overflow_force_software.c
* Author: Kevin London
* london@cs.utk.edu
* Mods: Maynard Johnson
* maynardj@us.ibm.com
* Philip Mucci
* mucci@cs.utk.edu
* Haihang You
* you@cs.utk.edu
* <your name here>
* <your email address>
*/

/* This file performs the following test: overflow dispatch of an eventset
with just a single event. Using both Hardware and software overflows

The Eventset contains:
+ PAPI_FP_INS (overflow monitor)

- Start eventset 1
- Do flops
- Stop and measure eventset 1
- Set up overflow on eventset 1
- Start eventset 1
- Do flops
- Stop eventset 1
- Set up forced software overflow on eventset 1
- Start eventset 1
- Do flops
- Stop eventset 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define OVER_FMT "handler(%d) Overflow at %p overflow_vector=%#llx!\n"
#define OUT_FMT		"%-12s : %16lld%16d%16lld\n"

#define SOFT_TOLERANCE 0.90
#define MY_NUM_TESTS 5

static int total[MY_NUM_TESTS] = { 0, };	/* total overflows */
static int use_total = 0;			   /* which total field to bump */
static long long values[MY_NUM_TESTS] = { 0, };

void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	( void ) context;

	if ( !TESTS_QUIET ) {
		fprintf( stderr, OVER_FMT, EventSet, address, overflow_vector );
	}

	total[use_total]++;
}

int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	long long hard_min, hard_max, soft_min, soft_max;
	int retval;
	int PAPI_event = 0, mythreshold;
	char event_name[PAPI_MAX_STR_LEN];
	PAPI_option_t opt;
	PAPI_event_info_t info;
	PAPI_option_t itimer;
	const PAPI_hw_info_t *hw_info = NULL;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	/* query and set up the right instruction to monitor */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
			PAPI_get_event_info( PAPI_FP_INS, &info );
			if ( info.count == 1 || 
                             !strcmp( info.derived, "DERIVED_CMPD" ) )
				PAPI_event = PAPI_FP_INS;
		}
	}
	if ( PAPI_event == 0 ) {
		if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK ) {
			PAPI_get_event_info( PAPI_FP_OPS, &info );
			if ( info.count == 1 || 
                             !strcmp( info.derived, "DERIVED_CMPD" ) )
				PAPI_event = PAPI_FP_OPS;
		}
	}
	if ( PAPI_event == 0 ) {
		if ( PAPI_query_event( PAPI_TOT_INS ) == PAPI_OK ) {
			PAPI_get_event_info( PAPI_TOT_INS, &info );
			if ( info.count == 1 || 
                             !strcmp( info.derived, "DERIVED_CMPD" ) )
				PAPI_event = PAPI_TOT_INS;
		}
	}

	if ( PAPI_event == 0 )
		test_skip( __FILE__, __LINE__, "No suitable event for this test found!",
				   0 );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	if ( PAPI_event == PAPI_FP_INS )
		mythreshold = THRESHOLD;
	else
#if defined(linux)
		mythreshold = ( int ) hw_info->cpu_max_mhz * 20000;
#else
		mythreshold = THRESHOLD * 2;
#endif

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	retval = PAPI_add_event( EventSet, PAPI_event );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	retval = PAPI_get_opt( PAPI_COMPONENTINFO, &opt );
	if ( retval != PAPI_OK )
		test_skip( __FILE__, __LINE__,
				   "Platform does not support Hardware overflow", 0 );

	do_stuff(  );

	/* Do reference count */

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values[use_total] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	use_total++;

	/* Now do hardware overflow reference count */

	retval = PAPI_overflow( EventSet, PAPI_event, mythreshold, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values[use_total] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	use_total++;

	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	/* Now do software overflow reference count, uses SIGPROF */

	retval =
		PAPI_overflow( EventSet, PAPI_event, mythreshold,
					   PAPI_OVERFLOW_FORCE_SW, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values[use_total] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	use_total++;

	retval =
		PAPI_overflow( EventSet, PAPI_event, 0, PAPI_OVERFLOW_FORCE_SW,
					   handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	/* Now do software overflow with SIGVTALRM */

	memset( &itimer, 0, sizeof ( itimer ) );
	itimer.itimer.itimer_num = ITIMER_VIRTUAL;
	itimer.itimer.itimer_sig = SIGVTALRM;

	if ( PAPI_set_opt( PAPI_DEF_ITIMER, &itimer ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

	retval =
		PAPI_overflow( EventSet, PAPI_event, mythreshold,
					   PAPI_OVERFLOW_FORCE_SW, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values[use_total] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	use_total++;

	retval =
		PAPI_overflow( EventSet, PAPI_event, 0, PAPI_OVERFLOW_FORCE_SW,
					   handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	/* Now do software overflow with SIGALRM */

	memset( &itimer, 0, sizeof ( itimer ) );
	itimer.itimer.itimer_num = ITIMER_REAL;
	itimer.itimer.itimer_sig = SIGALRM;
	if ( PAPI_set_opt( PAPI_DEF_ITIMER, &itimer ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

	retval =
		PAPI_overflow( EventSet, PAPI_event, mythreshold,
					   PAPI_OVERFLOW_FORCE_SW, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values[use_total] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	use_total++;

	retval =
		PAPI_overflow( EventSet, PAPI_event, 0, PAPI_OVERFLOW_FORCE_SW,
					   handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	if ( !TESTS_QUIET ) {
		if ( ( retval =
			   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

		printf
			( "Test case: Software overflow of various types with 1 event in set.\n" );
		printf
			( "------------------------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", mythreshold );
		printf
			( "------------------------------------------------------------------------------\n" );

		printf( "Test type   : %11s%13s%13s%13s%13s\n", "Reference", "Hardware",
				"ITIMER_PROF", "ITIMER_VIRT", "ITIMER_REAL" );
		printf( "%-12s: %11lld%13lld%13lld%13lld%13lld\n", info.symbol,
				values[0], values[1], values[2], values[3], values[4] );
		printf( "Overflows   : %11d%13d%13d%13d%13d\n", total[0], total[1],
				total[2], total[3], total[4] );
		printf
			( "------------------------------------------------------------------------------\n" );

		printf( "Verification:\n" );

		printf
			( "Overflow in Column 2 greater than or equal to overflows in Columns 3, 4, 5\n" );
		printf( "Overflow in Columns 3, 4, 5 greater than 0\n" );
	}

	hard_min =
		( long long ) ( ( ( double ) values[0] * ( 1.0 - OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	hard_max =
		( long long ) ( ( ( double ) values[0] * ( 1.0 + OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	soft_min =
		( long long ) ( ( ( double ) values[0] * ( 1.0 - SOFT_TOLERANCE ) ) /
						( double ) mythreshold );
	soft_max =
		( long long ) ( ( ( double ) values[0] * ( 1.0 + SOFT_TOLERANCE ) ) /
						( double ) mythreshold );
	
	if ( total[1] > hard_max || total[1] < hard_min )
		test_fail( __FILE__, __LINE__, "Hardware Overflows outside limits", 1 );

	if ( total[2] > soft_max || total[3] > soft_max || total[4] > soft_max )
		test_fail( __FILE__, __LINE__,
				   "Software Overflows exceed theoretical maximum", 1 );

	if ( total[2] < soft_min || total[3] < soft_min || total[4] < soft_min )
		printf( "WARNING: Software Overflow occurring but suspiciously low\n" );

	if ( ( total[2] == 0 ) || ( total[3] == 0 ) || ( total[4] == 0 ) )
		test_fail( __FILE__, __LINE__, "Software Overflows", 1 );

	test_pass( __FILE__ );

	return 0;

}
