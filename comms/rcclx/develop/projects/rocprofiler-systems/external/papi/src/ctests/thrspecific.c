/* This file performs the following test: start, stop and timer
functionality for 2 slave pthreads */

/* No it doesn't, that description is *completely* wrong */

/* I think this is trying to test the pthread thread-specific */
/* implementation but it is unclear and the git commit history */
/* does not help at all here */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

static volatile int processing = 1;

void *
Thread( void *arg )
{
	int retval;
	void *arg2;
	int i;

	retval = PAPI_register_thread(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", retval );
	}

	if (!TESTS_QUIET) {
		printf( "Thread %#x started, specific data is at %p\n",
			( int ) pthread_self(  ), arg );
	}

	retval = PAPI_set_thr_specific( PAPI_USR1_TLS, arg );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_set_thr_specific", retval );
	}

	retval = PAPI_get_thr_specific( PAPI_USR1_TLS, &arg2 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_thr_specific", retval );
	}

	if ( arg != arg2 ) {
		test_fail( __FILE__, __LINE__, "set vs get specific", 0 );
	}

	while ( processing ) {
		if ( *( ( int * ) arg ) == 500000 ) {
			sleep( 1 );

			PAPI_all_thr_spec_t data;
			data.num = 10;
			data.id = ( unsigned long * ) malloc( ( size_t ) data.num *
											sizeof ( unsigned long ) );
			data.data = ( void ** ) malloc( ( size_t ) data.num * sizeof ( void * ) );

			retval = PAPI_get_thr_specific( PAPI_USR1_TLS | PAPI_TLS_ALL_THREADS,
									   ( void ** ) &data );
			if ( retval != PAPI_OK ) {
				test_fail( __FILE__, __LINE__, "PAPI_get_thr_specific",
						   retval );
			}

			if ( data.num != 5 ) {
				test_fail( __FILE__, __LINE__, "data.num != 5", 0 );
			}

			if (!TESTS_QUIET) for ( i = 0; i < data.num; i++ ) {
				printf( "Entry %d, Thread %#lx, Data Pointer %p, Value %d\n",
						i, data.id[i], data.data[i], *( int * ) data.data[i] );
			}
			free(data.id);
			free(data.data);
			processing = 0;
		}
	}

	retval = PAPI_unregister_thread(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", retval );
	}

	return NULL;
}

int
main( int argc, char **argv )
{
	pthread_t e_th, f_th, g_th, h_th;
	int flops1, flops2, flops3, flops4, flops5;
	int retval, rc;
	pthread_attr_t attr;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if (!quiet) printf("Testing threads\n");

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_thread_init( ( unsigned long ( * )( void ) )
				( pthread_self ) );
	if ( retval != PAPI_OK ) {
		if ( retval == PAPI_ECMP ) {
			test_skip( __FILE__, __LINE__,
					"PAPI_thread_init", retval );
		}
		else {
			test_fail( __FILE__, __LINE__,
				"PAPI_thread_init", retval );
		}
	}

	pthread_attr_init( &attr );

#ifdef PTHREAD_CREATE_UNDETACHED
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_UNDETACHED );
#endif

#ifdef PTHREAD_SCOPE_SYSTEM
	retval = pthread_attr_setscope( &attr, PTHREAD_SCOPE_SYSTEM );
	if ( retval != 0 ) {
		test_skip( __FILE__, __LINE__, "pthread_attr_setscope", retval );
	}
#endif

	flops1 = 1000000;
	rc = pthread_create( &e_th, &attr, Thread, ( void * ) &flops1 );
	if ( rc ) {
		retval = PAPI_ESYS;
		test_fail( __FILE__, __LINE__, "pthread_create", retval );
	}

	flops2 = 2000000;
	rc = pthread_create( &f_th, &attr, Thread, ( void * ) &flops2 );
	if ( rc ) {
		retval = PAPI_ESYS;
		test_fail( __FILE__, __LINE__, "pthread_create", retval );
	}

	flops3 = 4000000;
	rc = pthread_create( &g_th, &attr, Thread, ( void * ) &flops3 );
	if ( rc ) {
		retval = PAPI_ESYS;
		test_fail( __FILE__, __LINE__, "pthread_create", retval );
	}

	flops4 = 8000000;
	rc = pthread_create( &h_th, &attr, Thread, ( void * ) &flops4 );
	if ( rc ) {
		retval = PAPI_ESYS;
		test_fail( __FILE__, __LINE__, "pthread_create", retval );
	}

	pthread_attr_destroy( &attr );

	flops5 = 500000;
	Thread( &flops5 );

	pthread_join( h_th, NULL );
	pthread_join( g_th, NULL );
	pthread_join( f_th, NULL );
	pthread_join( e_th, NULL );

	test_pass( __FILE__ );

	pthread_exit( NULL );

	return 1;
}
