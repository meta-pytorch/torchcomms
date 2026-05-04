/* This code attempts to test that SHMEM works with PAPI	*/
/* SHMEM was developed by Cray and supported by various		*/
/* other vendors.						*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

void
Thread( int n )
{
	int retval, num_tests = 1;
	int EventSet1 = PAPI_NULL;
	int mask1 = 0x5;
	int num_events1;
	long long **values;
	long long elapsed_us, elapsed_cyc;

	EventSet1 = add_test_events( &num_events1, &mask1, 1 );

	/* num_events1 is greater than num_events2 so don't worry. */

	values = allocate_test_space( num_tests, num_events1 );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_start( EventSet1 );

	/* we should indicate failure somehow, not just exit */
	if ( retval != PAPI_OK )
		exit( 1 );

	do_flops( n );

	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK )
		exit( 1 );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	remove_test_events( &EventSet1, mask1 );

	printf( "Thread %#x PAPI_FP_INS : \t%lld\n", n / 1000000,
			( values[0] )[0] );
	printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", n / 1000000,
			( values[0] )[1] );
	printf( "Thread %#x Real usec   : \t%lld\n", n / 1000000,
			elapsed_us );
	printf( "Thread %#x Real cycles : \t%lld\n", n / 1000000,
			elapsed_cyc );

	free_test_space( values, num_tests );
}

int
main( int argc, char **argv )
{
	int quiet;
	long long elapsed_us, elapsed_cyc;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

#ifdef HAVE_OPENSHMEM
	/* Start 2 processing elements (SHMEM call) */
	start_pes( 2 );
	Thread( 1000000 * ( _my_pe(  ) + 1 ) );
#else
	if (!quiet) {
		printf("No OpenSHMEM support\n");
	}
	test_skip( __FILE__, __LINE__, "OpenSHMEM support not found, skipping.", 0);
#endif

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	printf( "Master real usec   : \t%lld\n", elapsed_us );
	printf( "Master real cycles : \t%lld\n", elapsed_cyc );

	return 0;
}
