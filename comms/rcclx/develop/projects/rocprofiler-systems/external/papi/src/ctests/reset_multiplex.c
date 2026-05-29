/* This file performs the same tests as the reset test
   but does it with the events multiplexed.

   This is mostly to test perf_event, where resetting
   multiplexed events is handled differently than grouped events.

*/

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval, num_tests = 9, num_events, tmp, i;
	long long **values;
	int EventSet = PAPI_NULL;
	int PAPI_event, mask;
	char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_2MAX_STR_LEN];
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

        retval = PAPI_multiplex_init(  );
        if ( retval == PAPI_ENOSUPP) {
           test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
        }
        else if ( retval != PAPI_OK ) {
           test_fail( __FILE__, __LINE__, "PAPI_multiplex_init", retval );
	}


	/* add PAPI_TOT_CYC and one of the events in
	   PAPI_FP_INS, PAPI_FP_OPS or PAPI_TOT_INS,
	   depending on the availability of the event
	   on the platform */
	EventSet = add_two_events( &num_events, &PAPI_event, &mask );

	/* Set multiplexing on the eventset */

        retval = PAPI_set_multiplex( EventSet );
        if ( retval != PAPI_OK) {
           test_fail(__FILE__, __LINE__, "Setting multiplex", retval);
        }

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}
	sprintf( add_event_str, "PAPI_add_event[%s]", event_name );

	values = allocate_test_space( num_tests, num_events );

	/*===== Test 1: Start/Stop =======================*/

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/*===== Test 2 Start/Stop =======================*/

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/*===== Test 3: Reset/Start/Stop =======================*/

	retval = PAPI_reset( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_reset", retval );
	}

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet, values[2] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/*===== Test 4: Start/Read =======================*/

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS / 2 );

	retval = PAPI_read( EventSet, values[3] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	}

	/*===== Test 5: Read =======================*/

	do_flops( NUM_FLOPS / 2 );

	retval = PAPI_read( EventSet, values[4] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	}

	/*===== Test 6: Read/Accum =======================*/

	do_flops( NUM_FLOPS / 2 );

	retval = PAPI_read( EventSet, values[5] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	}

	retval = PAPI_accum( EventSet, values[5] );

	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_accum", retval );
	}

	/*===== Test 7: Read =======================*/

	do_flops( NUM_FLOPS / 2 );

	retval = PAPI_read( EventSet, values[6] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	}

	/*===== Test 8 Reset/Stop =======================*/
	retval = PAPI_reset( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_reset", retval );
	}

	do_flops( NUM_FLOPS / 2 );

	retval = PAPI_stop( EventSet, values[7] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/*===== Test 9: Reset/Read =======================*/
	retval = PAPI_reset( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_reset", retval );
	}

	retval = PAPI_read( EventSet, values[8] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	}

	remove_test_events( &EventSet, mask );

	if (!quiet) {
		printf( "Test case: Start/Stop/Read/Accum/Reset.\n" );
		printf( "----------------------------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
			stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
			stringify_granularity( tmp ) );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-------------------------------------------------------------------------\n" );

		sprintf( add_event_str, "%s:", event_name );
		printf( "                           PAPI_TOT_CYC    %s\n", event_name );
		printf( "1. start,ops,stop          %10lld      %10lld\n", values[0][0],
			values[0][1] );
		printf( "2. start,ops,stop          %10lld      %10lld\n", values[1][0],
			values[1][1] );
		printf( "3. reset,start,ops,stop    %10lld      %10lld\n", values[2][0],
			values[2][1] );
		printf( "4. start,ops/2,read        %10lld      %10lld\n", values[3][0],
			values[3][1] );
		printf( "5. ops/2,read              %10lld      %10lld\n", values[4][0],
			values[4][1] );
		printf( "6. ops/2,accum             %10lld      %10lld\n", values[5][0],
			values[5][1] );
		printf( "7. ops/2,read              %10lld      %10lld\n", values[6][0],
			values[6][1] );
		printf( "8. reset,ops/2,stop        %10lld      %10lld\n", values[7][0],
			values[7][1] );
		printf( "9. reset,read              %10lld      %10lld\n", values[8][0],
			values[8][1] );
		printf( "-------------------------------------------------------------------------\n" );
		printf( "Verification:\n" );
		printf( "Row 1 approximately equals rows 2 and 3 \n" );
		printf( "Row 4 approximately equals 1/2 of row 3\n" );
		printf( "Row 5 approximately equals twice row 4\n" );
		printf( "Row 6 approximately equals 6 times row 4\n" );
		printf( "Rows 7 and 8 approximately equal row 4\n" );
		printf( "Row 9 equals 0\n" );
		printf( "%% difference between %s 1 & 2: %.2f\n", "PAPI_TOT_CYC",
			100.0 * ( float ) values[0][0] / ( float ) values[1][0] );
		printf( "%% difference between %s 1 & 2: %.2f\n", add_event_str,
			100.0 * ( float ) values[0][1] / ( float ) values[1][1] );
	}

	for ( i = 0; i <= 1; i++ ) {
		if ( !approx_equals
			 ( ( double ) values[0][i], ( double ) values[1][i] ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[1][i], ( double ) values[2][i] ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[2][i], ( double ) values[3][i] * 2.0 ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[2][i], ( double ) values[4][i] ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[5][i], ( double ) values[3][i] * 6.0 ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[6][i], ( double ) values[3][i] ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( !approx_equals
			 ( ( double ) values[7][i], ( double ) values[3][i] ) )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
		if ( values[8][i] != 0LL )
			test_fail( __FILE__, __LINE__,
					   ( ( i == 0 ) ? "PAPI_TOT_CYC" : add_event_str ), 1 );
	}

	test_pass( __FILE__ );

	return 0;
}
