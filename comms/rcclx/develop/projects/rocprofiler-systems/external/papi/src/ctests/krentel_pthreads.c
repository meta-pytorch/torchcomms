/*
 *  Test PAPI with multiple threads.
 */

#define MAX_THREADS 256

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#include "papi.h"
#include "papi_test.h"

#define EVENT  PAPI_TOT_CYC

static int program_time = 5;
static int threshold = 20000000;
static int num_threads = 3;

static long count[MAX_THREADS];
static long iter[MAX_THREADS];
static struct timeval last[MAX_THREADS];

static pthread_key_t key;

static struct timeval start;

static void
my_handler( int EventSet, void *pc, long long ovec, void *context )
{
	( void ) EventSet;
	( void ) pc;
	( void ) ovec;
	( void ) context;

	long num = ( long ) pthread_getspecific( key );

	if ( num < 0 || num > num_threads )
		test_fail( __FILE__, __LINE__, "getspecific failed", 1 );
	count[num]++;
}

static void
print_rate( long num )
{
	struct timeval now;
	long st_secs;
	double last_secs;

	gettimeofday( &now, NULL );
	st_secs = now.tv_sec - start.tv_sec;
	last_secs = ( double ) ( now.tv_sec - last[num].tv_sec )
		+ ( ( double ) ( now.tv_usec - last[num].tv_usec ) ) / 1000000.0;
	if ( last_secs <= 0.001 )
		last_secs = 0.001;

	if (!TESTS_QUIET) {
		printf( "[%ld] time = %ld, count = %ld, iter = %ld, "
			"rate = %.1f/Kiter\n",
			num, st_secs, count[num], iter[num],
			( 1000.0 * ( double ) count[num] ) / ( double ) iter[num] );
	}

	count[num] = 0;
	iter[num] = 0;
	last[num] = now;
}

static void
do_cycles( long num, int len )
{
	struct timeval start, now;
	double x, sum;

	gettimeofday( &start, NULL );

	for ( ;; ) {
		sum = 1.0;
		for ( x = 1.0; x < 250000.0; x += 1.0 )
			sum += x;
		if ( sum < 0.0 )
			printf( "==>>  SUM IS NEGATIVE !!  <<==\n" );

		iter[num]++;

		gettimeofday( &now, NULL );
		if ( now.tv_sec >= start.tv_sec + len )
			break;
	}
}

static void *
my_thread( void *v )
{
	long num = ( long ) v;
	int n;
	int EventSet = PAPI_NULL;
	long long value;

	int retval;

	retval = PAPI_register_thread(  );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", retval );
	}
	pthread_setspecific( key, v );

	count[num] = 0;
	iter[num] = 0;
	last[num] = start;

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset failed", retval );
	}

	retval = PAPI_add_event( EventSet, EVENT );
	if (retval != PAPI_OK ) {
		if (!TESTS_QUIET) printf("Trouble adding event\n");
		test_fail( __FILE__, __LINE__, "PAPI_add_event failed", retval );
	}

	if ( PAPI_overflow( EventSet, EVENT, threshold, 0, my_handler ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow failed", 1 );

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start failed", 1 );

	if (!TESTS_QUIET) printf( "launched timer in thread %ld\n", num );

	for ( n = 1; n <= program_time; n++ ) {
		do_cycles( num, 1 );
		print_rate( num );
	}

	PAPI_stop( EventSet, &value );

        retval = PAPI_overflow( EventSet, EVENT, 0, 0, my_handler);
	if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__, "PAPI_overflow failed to reset the overflow handler", retval );

	if ( PAPI_remove_event( EventSet, EVENT ) != PAPI_OK ) 
	    test_fail( __FILE__, __LINE__, "PAPI_remove_event", 1 );

	if ( PAPI_destroy_eventset( &EventSet ) != PAPI_OK ) 
	    test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", 1 );

	if ( PAPI_unregister_thread( ) != PAPI_OK ) 
            test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", 1 );

	return ( NULL );
}

int
main( int argc, char **argv )
{
	pthread_t *td = NULL;
	long n;
	int quiet,retval;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	if ( argc < 2 || sscanf( argv[1], "%d", &program_time ) < 1 )
		program_time = 6;
	if ( argc < 3 || sscanf( argv[2], "%d", &threshold ) < 1 )
		threshold = 20000000;
	if ( argc < 4 || sscanf( argv[3], "%d", &num_threads ) < 1 )
		num_threads = 3;

	td = malloc((num_threads+1) * sizeof(pthread_t));
	if (!td) {
		test_fail( __FILE__, __LINE__, "td malloc failed", 1 );
	}

	if (!quiet) {
		printf( "program_time = %d, threshold = %d, num_threads = %d\n\n",
			program_time, threshold, num_threads );
	}

	if ( PAPI_library_init( PAPI_VER_CURRENT ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init failed", 1 );

	/* Test to be sure we can add events */
	retval = PAPI_query_event( EVENT );
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Trouble finding event\n");
		test_skip(__FILE__,__LINE__,"Event not available",1);
	}

	if ( PAPI_thread_init( ( unsigned long ( * )( void ) ) ( pthread_self ) ) !=
		 PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_thread_init failed", 1 );

	if ( pthread_key_create( &key, NULL ) != 0 )
		test_fail( __FILE__, __LINE__, "pthread key create failed", 1 );

	gettimeofday( &start, NULL );

	for ( n = 1; n <= num_threads; n++ ) {
		if ( pthread_create( &(td[n]), NULL, my_thread, ( void * ) n ) != 0 )
			test_fail( __FILE__, __LINE__, "pthread create failed", 1 );
	}

	my_thread( ( void * ) 0 );

	/* wait for all the threads */
	for ( n = 1; n <= num_threads; n++ ) {
	  	if ( pthread_join( td[n], NULL))
			test_fail( __FILE__, __LINE__, "pthread join failed", 1 );
	}

	free(td);

	if (!quiet) printf( "done\n" );

	test_pass( __FILE__ );

	return 0;
}
