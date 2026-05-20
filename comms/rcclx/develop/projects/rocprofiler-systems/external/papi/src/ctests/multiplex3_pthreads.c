/*
* File:    multiplex3_pthreads.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    John May
*          johnmay@llnl.gov
*/

/* This file tests the multiplex functionality when there are
 * threads in which the application isn't calling PAPI (and only
 * one thread that is calling PAPI.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define MAX_TO_ADD 5

/* A thread function that does nothing forever, while the other
 * tests are running.
 */
void *
thread_fn( void *dummy )
{
	( void ) dummy;
	while ( 1 ) {
		do_stuff(  );
	}
	return NULL;
}

/* Runs a bunch of multiplexed events */

static void
mainloop( int arg )
{
	int allvalid;
	long long *values;
	int EventSet = PAPI_NULL;
	int retval, i, j = 2, skipped_counters=0;
	PAPI_event_info_t pset;

	( void ) arg;

	/* Initialize the library */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_multiplex_init(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI multiplex init fail\n", retval );
	}

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );
	}

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
		test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	} else if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
	}

	retval = PAPI_thread_init( ( unsigned long ( * )( void ) ) ( pthread_self ) );
	if (retval != PAPI_OK ) {
		if ( retval == PAPI_ECMP )
			test_skip( __FILE__, __LINE__, "PAPI_thread_init", retval );
		else
			test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );
	}

	retval = PAPI_add_event( EventSet, PAPI_TOT_INS );
	if ( ( retval != PAPI_OK ) && ( retval != PAPI_ECNFLCT ) ) {
		if (!TESTS_QUIET) printf("Trouble adding PAPI_TOT_INS\n");
		test_skip( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if ( !TESTS_QUIET ) {
		printf( "Added %s\n", "PAPI_TOT_INS" );
	}

	retval = PAPI_add_event( EventSet, PAPI_TOT_CYC );
	if ( ( retval != PAPI_OK ) && ( retval != PAPI_ECNFLCT ) )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	if ( !TESTS_QUIET ) {
		printf( "Added %s\n", "PAPI_TOT_CYC" );
	}

	values = ( long long * ) malloc( MAX_TO_ADD * sizeof ( long long ) );
	if ( values == NULL )
		test_fail( __FILE__, __LINE__, "malloc", 0 );

	for ( i = 0; i < PAPI_MAX_PRESET_EVENTS; i++ ) {
		retval = PAPI_get_event_info( i | PAPI_PRESET_MASK, &pset );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_get_event_info", retval );

		if ( pset.count ) {
			if (!TESTS_QUIET) printf( "Adding %s\n", pset.symbol );

			retval = PAPI_add_event( EventSet, ( int ) pset.event_code );
			if ( ( retval != PAPI_OK ) && ( retval != PAPI_ECNFLCT ) )
				test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

			if ( retval == PAPI_OK ) {
				if (!TESTS_QUIET) printf( "Added %s\n", pset.symbol );
			} else {
				if (!TESTS_QUIET) printf( "Could not add %s\n", pset.symbol );
			}

			do_stuff(  );

			if ( retval == PAPI_OK ) {
				retval = PAPI_start( EventSet );
				if ( retval != PAPI_OK )
					test_fail( __FILE__, __LINE__, "PAPI_start", retval );

				do_stuff(  );

				retval = PAPI_stop( EventSet, values );
				if ( retval != PAPI_OK )
					test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

				if ( values[j] ) {
					if ( ++j >= MAX_TO_ADD )
						break;
				} else {
					retval =
						PAPI_remove_event( EventSet, ( int ) pset.event_code );
					if ( retval == PAPI_OK )
						if (!TESTS_QUIET) printf( "Removed %s\n", pset.symbol );
				        /* This added because the test */
				        /* can take a long time if mplexing */
				        /* is broken and all values are 0   */
				        skipped_counters++;
				        if (skipped_counters>MAX_TO_ADD) break;

				}
			}
		}
	}

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if (!TESTS_QUIET) {
		test_print_event_header( "multiplex3_pthreads:\n", EventSet );
	}
	allvalid = 0;
	for ( i = 0; i < MAX_TO_ADD; i++ ) {
		if (!TESTS_QUIET) printf( ONENUM, values[i] );
		if ( values[i] != 0 )
			allvalid++;
	}
	if (!TESTS_QUIET) printf( "\n" );
	if ( !allvalid )
		test_fail( __FILE__, __LINE__, "all counter registered no counts", 1 );

	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	free( values );
	PAPI_shutdown(  );
}

int
main( int argc, char **argv )
{
	int i, rc, retval;
	pthread_t id[NUM_THREADS];
	pthread_attr_t attr;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if (!quiet) {
		printf( "%s: Using %d threads\n\n", argv[0], NUM_THREADS );
		printf( "Does non-threaded multiplexing work "
			"with extraneous threads present?\n" );
	}

	/* Create a bunch of unused pthreads, to simulate threads created
	 * by the system that the user doesn't know about.
	 */
	pthread_attr_init( &attr );
#ifdef PTHREAD_CREATE_UNDETACHED
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_UNDETACHED );
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
	retval = pthread_attr_setscope( &attr, PTHREAD_SCOPE_SYSTEM );
	if ( retval != 0 )
		test_skip( __FILE__, __LINE__, "pthread_attr_setscope", retval );
#endif

#ifdef PPC64
	sigset_t sigprof;
	sigemptyset( &sigprof );
	sigaddset( &sigprof, SIGPROF );
	retval = sigprocmask( SIG_BLOCK, &sigprof, NULL );
	if ( retval != 0 )
		test_fail( __FILE__, __LINE__, "sigprocmask SIG_BLOCK", retval );
#endif

	for ( i = 0; i < NUM_THREADS; i++ ) {
		rc = pthread_create( &id[i], &attr, thread_fn, NULL );
		if ( rc )
			test_fail( __FILE__, __LINE__, "pthread_create", rc );
	}
	pthread_attr_destroy( &attr );

#ifdef PPC64
	retval = sigprocmask( SIG_UNBLOCK, &sigprof, NULL );
	if ( retval != 0 )
		test_fail( __FILE__, __LINE__, "sigprocmask SIG_UNBLOCK", retval );
#endif

	mainloop( NUM_ITERS );

	test_pass( __FILE__ );

	return 0;

}
