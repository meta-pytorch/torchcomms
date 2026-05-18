/* This file performs the following test: start, stop with a derived event */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define EVENTSLEN 2

unsigned int PAPI_events[EVENTSLEN] = { 0, 0 };
static const int PAPI_events_len = 1;

int
main( int argc, char **argv )
{
	int retval, tmp;
	int EventSet = PAPI_NULL;
	int i;
	PAPI_event_info_t info;
	long long values;
	char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_2MAX_STR_LEN];
	int quiet=0;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf( "Test case %s: start, stop with a derived counter.\n",
				 __FILE__ );
		printf( "------------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
				 stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n\n", tmp,
				 stringify_granularity( tmp ) );
	}

	i = PAPI_PRESET_MASK;
	do {
		if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
			if ( info.count > 1 ) {
				PAPI_events[0] = ( unsigned int ) info.event_code;
				break;
			}
		}
	} while ( PAPI_enum_event( &i, 0 ) == PAPI_OK );

	if ( PAPI_events[0] == 0 ) {
		test_skip(__FILE__, __LINE__, "No events found", 0);
	}

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail(__FILE__, __LINE__,  "PAPI_create_eventset", retval );
	}

	for ( i = 0; i < PAPI_events_len; i++ ) {
		PAPI_event_code_to_name( ( int ) PAPI_events[i], event_name );
		if ( !quiet ) {
			printf( "Adding %s\n", event_name );
		}
		retval = PAPI_add_event( EventSet, ( int ) PAPI_events[i] );
		if ( retval != PAPI_OK ) {
			test_fail(__FILE__, __LINE__,  "PAPI_add_event", retval );
		}
	}

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	if (!quiet) printf( "Running do_stuff().\n" );

	do_stuff(  );

	retval = PAPI_stop( EventSet, &values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	if (!quiet) {

		sprintf( add_event_str, "%-12s : \t", event_name );
		printf( TAB1, add_event_str, values );
		printf( "------------------------------------------------\n" );
	}

	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK ) {
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );
	}

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );
	}

	if (!quiet) printf( "Verification: Does it produce a non-zero value?\n" );

	if ( values != 0 ) {
		if (!quiet) {
			printf( "Yes: " );
			printf( LLDFMT, values );
			printf( "\n" );
		}
	}
	else {
		test_fail(__FILE__,__LINE__, "Validation", 1 );
	}

	test_pass(__FILE__);

	return 0;
}
