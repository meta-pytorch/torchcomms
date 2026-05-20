/* This file performs the following test: profile for pthreads */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define THR 1000000
#define FLOPS 100000000

unsigned int length;
vptr_t my_start, my_end;

void *
Thread( void *arg )
{
	int retval, num_tests = 1, i;
	int EventSet1 = PAPI_NULL, mask1, PAPI_event;
	int num_events1;
	long long **values;
	long long elapsed_us, elapsed_cyc;
	unsigned short *profbuf;
	char event_name[PAPI_MAX_STR_LEN];

	retval = PAPI_register_thread(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", retval );
	}

	profbuf = ( unsigned short * ) malloc( length * sizeof ( unsigned short ) );
	if ( profbuf == NULL ) {
		test_fail(__FILE__, __LINE__, "Allocate memory",0);
	}

	memset( profbuf, 0x00, length * sizeof ( unsigned short ) );

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depends on the availability of the event on the
	   platform */
	EventSet1 = add_two_nonderived_events( &num_events1, &PAPI_event, &mask1 );

	values = allocate_test_space( num_tests, num_events1 );

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_profil( profbuf, length, my_start, 65536,
				EventSet1, PAPI_event, THR, PAPI_PROFIL_POSIX );
	if ( retval ) {
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}

	retval = PAPI_start( EventSet1 );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( *( int * ) arg );

	retval = PAPI_stop( EventSet1, values[0] );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	/* to remove the profile flag */
	retval = PAPI_profil( profbuf, length, my_start, 65536,
				EventSet1, PAPI_event, 0, PAPI_PROFIL_POSIX );
	if ( retval ) {
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}

	remove_test_events( &EventSet1, mask1 );

	if ( !TESTS_QUIET ) {
		if ( mask1 == 0x3 ) {
			printf( "Thread %#x PAPI_TOT_INS : \t%lld\n",
					( int ) pthread_self(  ), ( values[0] )[0] );
		} else {
			printf( "Thread %#x PAPI_FP_INS : \t%lld\n",
					( int ) pthread_self(  ), ( values[0] )[0] );
		}
		printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", ( int ) pthread_self(  ),
				( values[0] )[1] );
		printf( "Thread %#x Real usec   : \t%lld\n", ( int ) pthread_self(  ),
				elapsed_us );
		printf( "Thread %#x Real cycles : \t%lld\n", ( int ) pthread_self(  ),
				elapsed_cyc );

		printf( "Test case: PAPI_profil() for pthreads\n" );
		printf( "----Profile buffer for Thread %#x---\n",
				( int ) pthread_self(  ) );
		for ( i = 0; i < ( int ) length; i++ ) {
			if ( profbuf[i] )
				printf( "%#lx\t%d\n", ( unsigned long ) ( my_start + 2 * i ),
						profbuf[i] );
		}
	}
	for ( i = 0; i < ( int ) length; i++ )
		if ( profbuf[i] )
			break;

	if ( i >= ( int ) length ) {
		test_fail( __FILE__, __LINE__, "No information in buffers", 1 );
	}
	free_test_space( values, num_tests );

	retval = PAPI_unregister_thread(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", retval );
	}

	return NULL;
}

int
main( int argc, char **argv )
{
	pthread_t id[NUM_THREADS];
	int flops[NUM_THREADS];
	int i, rc, retval;
	pthread_attr_t attr;
	long long elapsed_us, elapsed_cyc;
	const PAPI_exe_info_t *prginfo = NULL;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	retval = PAPI_query_event(PAPI_TOT_CYC);
	if (retval != PAPI_OK) {

		if (!quiet) printf("Trouble adding event\n");
		test_skip(__FILE__,__LINE__,"No events",0);
	}

	retval = PAPI_thread_init( ( unsigned long ( * )( void ) ) ( pthread_self ));
	if (retval != PAPI_OK ) {
		if ( retval == PAPI_ECMP )
			test_skip( __FILE__, __LINE__, "PAPI_thread_init", retval );
		else
			test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );
	}

	if ( ( prginfo = PAPI_get_executable_info(  ) ) == NULL ) {
		retval = 1;
		test_fail( __FILE__, __LINE__, "PAPI_get_executable_info", retval );
	}

	my_start = prginfo->address_info.text_start;
	my_end = prginfo->address_info.text_end;
	length = ( unsigned int ) ( my_end - my_start );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

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
		flops[i] = FLOPS * ( i + 1 );
		rc = pthread_create( &id[i], &attr, Thread, ( void * ) &flops[i] );
		if ( rc )
			return ( FAILURE );
	}
	for ( i = 0; i < NUM_THREADS; i++ )
		pthread_join( id[i], NULL );

	pthread_attr_destroy( &attr );

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	if ( !quiet ) {
		printf( "Master real usec   : \t%lld\n", elapsed_us );
		printf( "Master real cycles : \t%lld\n", elapsed_cyc );
	}

	test_pass( __FILE__ );

	pthread_exit( NULL );

	return 0;

}
