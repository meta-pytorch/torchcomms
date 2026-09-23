/*
* File:    multiplex.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file tests the multiplex functionality, originally developed by
   John May of LLNL. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"


/* Tests that we can really multiplex a lot. */

static int
case1( void )
{
	int retval, i, EventSet = PAPI_NULL, j = 0, k = 0, allvalid = 1;
	int max_mux, nev, *events;
	long long *values;
	PAPI_event_info_t pset;
	char evname[PAPI_MAX_STR_LEN];

	/* Initialize PAPI */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_multiplex_init(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI multiplex init fail\n", retval );
	}

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif
	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );
	}
#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
        else if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
	}

	max_mux = PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL );
	if ( max_mux > 32 ) max_mux = 32;

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif

	/* Fill up the event set with as many non-derived events as we can */
	if (!TESTS_QUIET) {
		printf( "\nFilling the event set with as many non-derived events as we can...\n" );
	}

	i = PAPI_PRESET_MASK;
	do {
		if ( PAPI_get_event_info( i, &pset ) == PAPI_OK ) {
			if ( pset.count && ( strcmp( pset.derived, "NOT_DERIVED" ) == 0 ) ) {
				retval = PAPI_add_event( EventSet, ( int ) pset.event_code );
				if ( retval != PAPI_OK ) {
				   printf("Failed trying to add %s\n",pset.symbol);
				   break;
				}
				else {
					if (!TESTS_QUIET) printf( "Added %s\n", pset.symbol );
					j++;
				}
			}
		}
	} while ( ( PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK ) &&
			  ( j < max_mux ) );

	if (j==0) {
		if (!TESTS_QUIET) printf("No events found\n");
		test_skip(__FILE__,__LINE__,"No events",0);
	}

	events = ( int * ) malloc( ( size_t ) j * sizeof ( int ) );
	if ( events == NULL )
		test_fail( __FILE__, __LINE__, "malloc events", 0 );

	values = ( long long * ) malloc( ( size_t ) j * sizeof ( long long ) );
	if ( values == NULL )
		test_fail( __FILE__, __LINE__, "malloc values", 0 );

	do_stuff(  );

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	nev = j;
	retval = PAPI_list_events( EventSet, events, &nev );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_list_events", retval );

	if (!TESTS_QUIET) printf( "\nEvent Counts:\n" );
	for ( i = 0, allvalid = 0; i < j; i++ ) {
		PAPI_event_code_to_name( events[i], evname );
		if (!TESTS_QUIET) printf( TAB1, evname, values[i] );
		if ( values[i] == 0 )
			allvalid++;
	}
	if (!TESTS_QUIET) {
		printf( "\n" );
		if ( allvalid ) {
			printf( "Caution: %d counters had zero values\n", allvalid );
		}
	}

        if (allvalid==j) {
	   test_fail( __FILE__, __LINE__, "All counters returned zero", 5 );
	}

	for ( i = 0, allvalid = 0; i < j; i++ ) {
		for ( k = i + 1; k < j; k++ ) {
			if ( ( i != k ) && ( values[i] == values[k] ) ) {
				allvalid++;
				break;
			}
		}
	}

	if (!TESTS_QUIET) {
		if ( allvalid ) {
			printf( "Caution: %d counter pair(s) had identical values\n",
				allvalid );
		}
	}

	free( events );
	free( values );

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	return ( SUCCESS );
}

int
main( int argc, char **argv )
{
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if (!quiet) {
		printf( "%s: Does PAPI_multiplex_init() handle lots of events?\n",
				argv[0] );
		printf( "Using %d iterations\n", NUM_ITERS );
	}

	case1(  );
	test_pass( __FILE__ );

	return 0;
}
