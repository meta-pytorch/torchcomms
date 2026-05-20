/*
* File:    multiplex1_pthreads.c
* Author:  Rick Kufrin
*          rkufrin@ncsa.uiuc.edu
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file really bangs on the multiplex pthread functionality */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

static int *events;
static int numevents = 0;
static int max_events=0;

double
loop( long n )
{
	long i;
	double a = 0.0012;

	for ( i = 0; i < n; i++ ) {
		a += 0.01;
	}
	return a;
}

void *
thread( void *arg )
{
	( void ) arg;			 /*unused */
	int eventset = PAPI_NULL;
	long long *values;

	int ret = PAPI_register_thread(  );
	if ( ret != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", ret );
	ret = PAPI_create_eventset( &eventset );
	if ( ret != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );

	values=calloc(max_events,sizeof(long long));

	if (!TESTS_QUIET) printf( "Event set %d created\n", eventset );

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	ret = PAPI_assign_eventset_component( eventset, 0 );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component", ret );
	}

	ret = PAPI_set_multiplex( eventset );
      if ( ret == PAPI_ENOSUPP) {
         free(values);       
	      test_skip( __FILE__, __LINE__, "Multiplexing not supported", 1 );
	}
	else if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", ret );
	}

	ret = PAPI_add_events( eventset, events, numevents );
	if ( ret < PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_add_events", ret );
	}

	ret = PAPI_start( eventset );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_start", ret );
	}

	do_stuff(  );

	ret = PAPI_stop( eventset, values );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
	}

	ret = PAPI_cleanup_eventset( eventset );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", ret );
	}

	ret = PAPI_destroy_eventset( &eventset );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", ret );
	}

	ret = PAPI_unregister_thread(  );
	if ( ret != PAPI_OK ) {
      free(values);
		test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", ret );
   }

   free(values);
	return ( NULL );
}

int
main( int argc, char **argv )
{
	int nthreads = 8, retval, i;
	PAPI_event_info_t info;
	pthread_t *threads;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if ( !quiet ) {
		if ( argc > 1 ) {
			int tmp = atoi( argv[1] );
			if ( tmp >= 1 )
				nthreads = tmp;
		}
	}

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_thread_init( ( unsigned long ( * )( void ) ) pthread_self );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );
	}

	retval = PAPI_multiplex_init(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_multiplex_init", retval );
	}

	if ((max_events = PAPI_get_cmp_opt(PAPI_MAX_MPX_CTRS,NULL,0)) <= 0) {
		test_fail( __FILE__, __LINE__, "PAPI_get_cmp_opt", max_events );
	}

	if ((events = calloc(max_events,sizeof(int))) == NULL) {
		test_fail( __FILE__, __LINE__, "calloc", PAPI_ESYS );
	}

	/* Fill up the event set with as many non-derived events as we can */

	i = PAPI_PRESET_MASK;
	do {
		if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
			if ( info.count == 1 ) {
				events[numevents++] = ( int ) info.event_code;
				if (!quiet) printf( "Added %s\n", info.symbol );
			} else {
				if (!quiet) printf( "Skipping derived event %s\n", info.symbol );
			}
		}
	} while ( ( PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK )
			  && ( numevents < max_events ) );

	if (!quiet) printf( "Found %d events\n", numevents );

	if (numevents==0) {
		test_skip(__FILE__,__LINE__,"No events found",0);
	}

	do_stuff(  );

	if (!quiet) printf( "Creating %d threads:\n", nthreads );

	threads =
		( pthread_t * ) malloc( ( size_t ) nthreads * sizeof ( pthread_t ) );
	if ( threads == NULL ) {
      free(events);
		test_fail( __FILE__, __LINE__, "malloc", PAPI_ENOMEM );
	}

	/* Create the threads */
	for ( i = 0; i < nthreads; i++ ) {
		retval = pthread_create( &threads[i], NULL, thread, NULL );
		if ( retval != 0 ) {
         free(events);
         free(threads); 
			test_fail( __FILE__, __LINE__, "pthread_create", PAPI_ESYS );
		}
	}

	/* Wait for thread completion */
	for ( i = 0; i < nthreads; i++ ) {
		retval = pthread_join( threads[i], NULL );
		if ( retval != 0 ) {
         free(events);
         free(threads); 
			test_fail( __FILE__, __LINE__, "pthread_join", PAPI_ESYS );
		}
	}

	if (!quiet) printf( "Done." );

   free(events);
   free(threads); 
	test_pass( __FILE__ );

	pthread_exit( NULL );

	return 0;
}
