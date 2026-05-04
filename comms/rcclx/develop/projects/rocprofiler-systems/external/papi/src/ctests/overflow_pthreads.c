/* This file performs the following test: overflow dispatch with pthreads

   - This tests the dispatch of overflow calls from PAPI. These are counted
   in the default counting domain and default granularity, depending on
   the platform. Usually this is the user domain (PAPI_DOM_USER) and
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)
     + PAPI_TOT_CYC

   - Set up overflow
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "papi.h"
#include "do_loops.h"
#include "papi_test.h"

static const PAPI_hw_info_t *hw_info = NULL;
static int total[NUM_THREADS];
static int expected[NUM_THREADS];
static pthread_t myid[NUM_THREADS];

static void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
#if 0
	printf( "handler(%d,%#lx,%llx) Overflow %d in thread %lx\n",
			EventSet, ( unsigned long ) address, overflow_vector,
			total[EventSet], PAPI_thread_id(  ) );
	printf( "%lx vs %lx\n", myid[EventSet], PAPI_thread_id(  ) );
#else  /* eliminate unused parameter warning message */
	( void ) address;
	( void ) overflow_vector;
	( void ) context;
#endif
	total[EventSet]++;
}

static long long mythreshold=0;

static void *
Thread( void *arg )
{
	int retval, num_tests = 1;
	int EventSet1 = PAPI_NULL;
	int mask1, papi_event;
	int num_events1;
	long long **values;
	long long elapsed_us, elapsed_cyc;
	char event_name[PAPI_MAX_STR_LEN];

	retval = PAPI_register_thread(  );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", retval );

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depends on the availability of the event on the 
	   platform */
	EventSet1 =
		add_two_nonderived_events( &num_events1, &papi_event, &mask1 );

	if (EventSet1 < 0) return NULL;

	/* Wait, we're indexing a per-thread array with the EventSet number? */
	/* does that make any sense at all???? -- vmw                        */
	expected[EventSet1] = *( int * ) arg / mythreshold;
	myid[EventSet1] = PAPI_thread_id(  );

	values = allocate_test_space( num_tests, num_events1 );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	if ((retval = PAPI_overflow( EventSet1, papi_event, 
				     mythreshold, 0, handler ) ) != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	/* start_timer(1); */
	if ( ( retval = PAPI_start( EventSet1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	if ( ( retval = PAPI_stop( EventSet1, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	retval = PAPI_overflow( EventSet1, papi_event, 0, 0, NULL );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	remove_test_events( &EventSet1, mask1 );

	retval = PAPI_event_code_to_name( papi_event, event_name );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	if ( !TESTS_QUIET ) {
		printf( "Thread %#x %s : \t%lld\n", ( int ) pthread_self(  ),
				event_name, ( values[0] )[0] );
		printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", ( int ) pthread_self(  ),
				( values[0] )[1] );
		printf( "Thread %#x Real usec   : \t%lld\n", ( int ) pthread_self(  ),
				elapsed_us );
		printf( "Thread %#x Real cycles : \t%lld\n", ( int ) pthread_self(  ),
				elapsed_cyc );
	}
	free_test_space( values, num_tests );
	retval = PAPI_unregister_thread(  );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", retval );
	return ( NULL );
}

int
main( int argc, char **argv )
{
	pthread_t id[NUM_THREADS];
	int flops[NUM_THREADS];
	int i, rc, retval;
	pthread_attr_t attr;
	float ratio;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	memset( total, 0x0, NUM_THREADS * sizeof ( *total ) );
	memset( expected, 0x0, NUM_THREADS * sizeof ( *expected ) );
	memset( myid, 0x0, NUM_THREADS * sizeof ( *myid ) );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	retval = PAPI_thread_init( ( unsigned long ( * )( void ) )
					( pthread_self ) );
	if (retval != PAPI_OK ) {
		if ( retval == PAPI_ECMP )
			test_skip( __FILE__, __LINE__, "PAPI_thread_init", retval );
		else
			test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );
	}
#if defined(linux)
	mythreshold = ((long long)hw_info->cpu_max_mhz) * 10000 * 2;
#else
	mythreshold = THRESHOLD * 2;
#endif

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
		flops[i] = NUM_FLOPS * ( i + 1 );
		rc = pthread_create( &id[i], &attr, Thread, ( void * ) &flops[i] );
		if ( rc )
			test_fail( __FILE__, __LINE__, "pthread_create", PAPI_ESYS );
	}
	for ( i = 0; i < NUM_THREADS; i++ )
		pthread_join( id[i], NULL );

	pthread_attr_destroy( &attr );

	{
		long long t = 0, r = 0;
		for ( i = 0; i < NUM_THREADS; i++ ) {
			t += ( NUM_FLOPS * ( i + 1 ) ) / mythreshold;
			r += total[i];
		}
		if (!quiet) {
			printf( "Expected total overflows: %lld\n", t );
			printf( "Received total overflows: %lld\n", r );
		}
	}
// FIXME: are we actually testing this properly?

/*   ratio = (float)total[0] / (float)expected[0]; */
/*   printf("Ratio of total to expected: %f\n",ratio); */
	ratio = 1.0;
	for ( i = 0; i < NUM_THREADS; i++ ) {
		if (!quiet) printf( "Overflows thread %d: %d, expected %d\n",
				i, total[i], ( int ) ( ratio * ( float ) expected[i] ) );
	}

	for ( i = 0; i < NUM_THREADS; i++ ) {
		if ( total[i] < ( int ) ( ( ratio * ( float ) expected[i] ) / 2.0 ) )
			test_fail( __FILE__, __LINE__, "not enough overflows", PAPI_EMISC );
	}

	test_pass( __FILE__ );

	pthread_exit( NULL );

	return 0;

}
