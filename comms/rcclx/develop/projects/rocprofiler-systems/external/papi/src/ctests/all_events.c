/* This file tries to add,start,stop, and remove all pre-defined events.
 * It is meant not to test the accuracy of the mapping but to make sure
 * that all events in the component will at least start (Helps to
 * catch typos).
 *
 * Author: Kevin London
 *         london@cs.utk.edu
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval, i;
	int EventSet = PAPI_NULL, count = 0, err_count = 0;
	long long values;
	PAPI_event_info_t info;
	int quiet=0;
	char error_message[BUFSIZ];

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	if (!quiet) {
		printf("\nTrying all pre-defined events:\n");
	}

	/* Initialize PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create an EventSet */
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* Add all preset events */
	for ( i = 0; i < PAPI_MAX_PRESET_EVENTS; i++ ) {

		if ( PAPI_get_event_info( PAPI_PRESET_MASK | i, &info ) != PAPI_OK )
			continue;

		if ( !( info.count ) )
			continue;

		if (!quiet) printf( "Adding %-14s", info.symbol );

		retval = PAPI_add_event( EventSet, ( int ) info.event_code );
		if ( retval != PAPI_OK ) {
			if (!quiet) {
				printf("Error adding event %s\n",info.symbol);
				if (retval==PAPI_ECNFLCT) {
					printf("Probably NMI watchdog related\n");
				}
			}
			if (retval==PAPI_ECNFLCT) {
				sprintf(error_message,"Problem adding %s (probably NMI Watchdog related)",info.symbol);
			}
			else {
				sprintf(error_message,"Problem adding %s",info.symbol);
			}
			test_warn( __FILE__, __LINE__, error_message, retval );
			err_count++;
		} else {
			retval = PAPI_start( EventSet );
			if ( retval != PAPI_OK ) {
				PAPI_perror( "PAPI_start" );
				err_count++;
			} else {
				retval = PAPI_stop( EventSet, &values );
				if ( retval != PAPI_OK ) {
					PAPI_perror( "PAPI_stop" );
					err_count++;
				} else {
					if (!quiet) printf( "successful\n" );
					count++;
				}
			}
			retval = PAPI_remove_event( EventSet, ( int ) info.event_code );
			if ( retval != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_remove_event", retval );
		}
	}
	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if (!quiet) {
		printf( "Successfully added, started and stopped %d events.\n", count );
	}

	if ( err_count ) {
		if (!quiet) printf( "Failed to add, start or stop %d events.\n", err_count );
	}

	if (count<=0) {
		test_fail( __FILE__, __LINE__, "No events added", 1 );
	}

	test_pass( __FILE__ );

	return 0;

}
