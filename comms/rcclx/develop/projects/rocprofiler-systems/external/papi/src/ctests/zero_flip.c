/* This file performs the following test: start, stop and timer functionality

   - It attempts to use the following two counters.
     It may use less depending on hardware counter resource limitations.
     These are counted in the default counting domain and default granularity,
     depending on the platform.
     Usually this is the user domain (PAPI_DOM_USER) and
     thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval, eventcnt, events[2], i, tmp;
	int EventSet1 = PAPI_NULL, EventSet2 = PAPI_NULL;
	int PAPI_event;
	long long values1[2], values2[2];
	long long elapsed_us, elapsed_cyc;
	char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_2MAX_STR_LEN];
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	/* query and set up the right instruction to monitor */
	if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK )
		PAPI_event = PAPI_FP_OPS;
	else
		PAPI_event = PAPI_TOT_INS;

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	sprintf( add_event_str, "PAPI_add_event[%s]", event_name );

	retval = PAPI_create_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	/* Add the events */
	if (!quiet) printf( "Adding: %s\n", event_name );
	retval = PAPI_add_event( EventSet1, PAPI_event );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Trouble adding event\n");
		test_skip( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	retval = PAPI_add_event( EventSet1, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	/* Add them reversed to EventSet2 */

	retval = PAPI_create_eventset( &EventSet2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	eventcnt = 2;
	retval = PAPI_list_events( EventSet1, events, &eventcnt );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_list_events", retval );

	for ( i = eventcnt - 1; i >= 0; i-- ) {
		retval = PAPI_event_code_to_name( events[i], event_name );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

		retval = PAPI_add_event( EventSet2, events[i] );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet1, values1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_start( EventSet2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet2, values2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	retval = PAPI_cleanup_eventset( EventSet1 );	/* JT */
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	retval = PAPI_cleanup_eventset( EventSet2 );	/* JT */
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if ( !quiet ) {
		printf( "Test case 0: start, stop.\n" );
		printf( "-----------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf
			( "-------------------------------------------------------------------------\n" );

		printf( "Test type    : \t           1\t           2\n" );

		sprintf( add_event_str, "%-12s : \t", event_name );
		printf( TAB2, add_event_str, values1[0], values2[1] );
		printf( TAB2, "PAPI_TOT_CYC : \t", values1[1], values2[0] );
		printf( TAB1, "Real usec    : \t", elapsed_us );
		printf( TAB1, "Real cycles  : \t", elapsed_cyc );

		printf
			( "-------------------------------------------------------------------------\n" );

		printf( "Verification: none\n" );
	}

	test_pass( __FILE__ );

	return 0;

}
