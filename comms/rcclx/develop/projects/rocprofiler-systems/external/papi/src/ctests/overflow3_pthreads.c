/*
* File:    overflow3_pthreads.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file tests the overflow functionality when there are
 * threads in which the application isn't calling PAPI (and only
 * one thread that is calling PAPI.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int total = 0;

void *
thread_fn( void *dummy )
{
	( void ) dummy;
	while ( 1 ) {
		do_stuff(  );
	}
	return ( NULL );
}

void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	( void ) overflow_vector;
	( void ) context;
	if ( !TESTS_QUIET ) {
		fprintf( stderr, "handler(%d ) Overflow at %p, thread %#lx!\n",
				 EventSet, address, PAPI_thread_id(  ) );
	}
	total++;
}

void
mainloop( int arg )
{
	int retval, num_tests = 1;
	int EventSet1 = PAPI_NULL;
	int mask1 = 0x0;
	int num_events1;
	long long **values;
	int PAPI_event;
	char event_name[PAPI_MAX_STR_LEN];

	( void ) arg;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depending on the availability of the event on the
	   platform */
	EventSet1 = add_two_nonderived_events( &num_events1,
			&PAPI_event, &mask1 );

	if (num_events1==0) {
		if (!TESTS_QUIET) printf("Trouble creating events\n");
		test_skip(__FILE__,__LINE__,"Creating events",0);
	}

	values = allocate_test_space( num_tests, num_events1 );

	if ( ( retval =
		   PAPI_overflow( EventSet1, PAPI_event, THRESHOLD, 0,
						  handler ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	do_stuff(  );

	if ( ( retval = PAPI_start( EventSet1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	if ( ( retval = PAPI_stop( EventSet1, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	/* clear the papi_overflow event */
	if ( ( retval =
		   PAPI_overflow( EventSet1, PAPI_event, 0, 0, NULL ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	if ( ( retval =
		   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

	if ( !TESTS_QUIET ) {
		printf( "Thread %#x %s : \t%lld\n", ( int ) pthread_self(  ),
				event_name, ( values[0] )[0] );
		printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", ( int ) pthread_self(  ),
				( values[0] )[1] );
	}

	retval = PAPI_cleanup_eventset( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	free_test_space( values, num_tests );
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
	quiet=tests_quiet( argc, argv );

	if (!quiet) {
		printf( "%s: Using %d threads\n\n", argv[0], NUM_THREADS );
		printf( "Does non-threaded overflow work "
			"with extraneous threads present?\n" );
	}

	pthread_attr_init( &attr );
#ifdef PTHREAD_CREATE_UNDETACHED
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_UNDETACHED );
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
	retval = pthread_attr_setscope( &attr, PTHREAD_SCOPE_SYSTEM );
	if ( retval != 0 )
		test_skip( __FILE__, __LINE__, "pthread_attr_setscope", retval );
#endif

	for ( i = 0; i < NUM_THREADS; i++ ) {
		rc = pthread_create( &id[i], &attr, thread_fn, NULL );
		if ( rc )
			test_fail( __FILE__, __LINE__, "pthread_create", rc );
	}
	pthread_attr_destroy( &attr );

	mainloop( NUM_ITERS );

	test_pass( __FILE__ );

	return 0;
}
