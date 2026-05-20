/* This file performs the following test: each OMP thread measures flops
for its provided tasks, and compares this to expected flop counts, each
thread having been provided with a random amount of work, such that the
time and order that they complete their measurements varies.
Specifically tested is the case where the value returned for some threads
actually corresponds to that for another thread reading its counter values
at the same time.

   - It is based on zero_omp.c but ignored much of its functionality.
   - It attempts to use the following two counters. It may use less
depending on hardware counter resource limitations. These are counted
in the default counting domain and default granularity, depending on
the platform. Usually this is the user domain (PAPI_DOM_USER) and
thread context (PAPI_GRN_THR).

     + PAPI_FP_INS
     + PAPI_TOT_CYC

Each thread inside the Thread routine:
   - Do prework (MAX_FLOPS - flops)
   - Get cyc.
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
   - Get cyc.
   - Return flops
*/

#include "papi_test.h"

#ifdef _OPENMP
#include <omp.h>
#else
#error "This compiler does not understand OPENMP"
#endif

const int MAX_FLOPS = NUM_FLOPS;

extern int TESTS_QUIET;				   /* Declared in test_utils.c */
const PAPI_hw_info_t *hw_info = NULL;

long long
Thread( int n )
{
	int retval, num_tests = 1;
	int EventSet1 = PAPI_NULL;
	int PAPI_event, mask1;
	int num_events1;
	long long flops;
	long long **values;
	long long elapsed_us, elapsed_cyc;
	char event_name[PAPI_MAX_STR_LEN];

	/* printf("Thread(n=%d) %#x started\n", n, omp_get_thread_num()); */
	num_events1 = 2;

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depending on the availability of the event on the
	   platform */
	EventSet1 = add_two_events( &num_events1, &PAPI_event, &mask1 );

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

	values = allocate_test_space( num_tests, num_events1 );

	do_flops( MAX_FLOPS - n );	/* prework for balance */

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( n );

	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	flops = ( values[0] )[0];

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	remove_test_events( &EventSet1, mask1 );

	if ( !TESTS_QUIET ) {
		/*printf("Thread %#x %-12s : \t%lld\t%d\n", omp_get_thread_num(), event_name,
		   (values[0])[0], n); */
#if 0
		printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", omp_get_thread_num(  ),
				values[0][0] );
		printf( "Thread %#x Real usec   : \t%lld\n", omp_get_thread_num(  ),
				elapsed_us );
		printf( "Thread %#x Real cycles : \t%lld\n", omp_get_thread_num(  ),
				elapsed_cyc );
#endif
	}

	/* It is illegal for the threads to exit in OpenMP */
	/* test_pass(__FILE__,0,0); */
	free_test_space( values, num_tests );

	PAPI_unregister_thread(  );
	/* printf("Thread %#x finished\n", omp_get_thread_num()); */
	return flops;
}

int
main( int argc, char **argv )
{
	int tid, retval;
	int maxthr = omp_get_max_threads(  );
	int flopper = 0;
	long long *flops = calloc( maxthr, sizeof ( long long ) );
	long long *flopi = calloc( maxthr, sizeof ( long long ) );

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( maxthr < 2 )
		test_skip( __FILE__, __LINE__, "omp_get_num_threads < 2", PAPI_EINVAL );

	if ( ( flops == NULL ) || ( flopi == NULL ) )
		test_fail( __FILE__, __LINE__, "calloc", PAPI_ENOMEM );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	retval =
		PAPI_thread_init( ( unsigned
							long ( * )( void ) ) ( omp_get_thread_num ) );
	if ( retval != PAPI_OK )
		if ( retval == PAPI_ECMP )
			test_skip( __FILE__, __LINE__, "PAPI_thread_init", retval );
		else
			test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );

	flopper = Thread( 65536 ) / 65536;
	printf( "flopper=%d\n", flopper );

	for ( int i = 0; i < 100000; i++ )
#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num(  );
		flopi[tid] = rand(  ) * 3;
		flops[tid] = Thread( ( flopi[tid] / flopper ) % MAX_FLOPS );
#pragma omp barrier
#pragma omp master
		if ( flops[tid] < flopi[tid] ) {
			printf( "test iteration=%d\n", i );
			for ( int j = 0; j < omp_get_num_threads(  ); j++ ) {
				printf( "Thread %#x Value %6lld %c %6lld", j, flops[j],
						( flops[j] < flopi[j] ) ? '<' : '=', flopi[j] );
				for ( int k = 0; k < omp_get_num_threads(  ); k++ )
					if ( ( k != j ) && ( flops[k] == flops[j] ) )
						printf( " == Thread %#x!", k );
				printf( "\n" );
			}
			test_fail( __FILE__, __LINE__, "value returned for thread",
					   PAPI_EBUG );
		}
	}

	test_pass( __FILE__, NULL, 0 );
	exit( 0 );
}
