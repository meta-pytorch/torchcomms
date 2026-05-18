/* This file checks to make sure the locking mechanisms work correctly	*/
/* on the platform.							*/
/* Platforms where the locking mechanisms are not implemented or are	*/
/* incorrectly implemented will fail.  -KSL				*/

#define MAX_THREADS 256
#define APPR_TOTAL_ITER 1000000

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#include "papi.h"
#include "papi_test.h"

volatile long long count = 0;
volatile long long tmpcount = 0;
volatile long long thread_iter = 0;

static int quiet=0;

void
lockloop( int iters, volatile long long *mycount )
{
	int i;
	for ( i = 0; i < iters; i++ ) {
		PAPI_lock( PAPI_USR1_LOCK );
		*mycount = *mycount + 1;
		PAPI_unlock( PAPI_USR1_LOCK );
	}
}

void *
Slave( void *arg )
{
	long long duration;

	duration = PAPI_get_real_usec(  );
	lockloop( thread_iter, &count );
	duration = PAPI_get_real_usec(  ) - duration;

	if (!quiet) {
		printf("%f lock/unlocks per us\n",
			(float)thread_iter/(float)duration);
	}
	pthread_exit( arg );
}


int
main( int argc, char **argv )
{
	pthread_t slaves[MAX_THREADS];
	int rc, i, nthr;
	int retval;
	const PAPI_hw_info_t *hwinfo = NULL;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	hwinfo = PAPI_get_hardware_info(  );
	if (hwinfo == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	retval = PAPI_thread_init((unsigned long (*)(void)) ( pthread_self ) );
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

	if ( hwinfo->ncpu > MAX_THREADS ) {
		nthr = MAX_THREADS;
	}
	else {
		nthr = hwinfo->ncpu;
	}

	/* Scale the per thread work to keep the serial runtime about the same. */
	thread_iter = APPR_TOTAL_ITER/sqrt(nthr);

	if (!quiet) {
		printf( "Creating %d threads, %lld lock/unlock\n",
			nthr , thread_iter);
	}

	for ( i = 0; i < nthr; i++ ) {
		rc = pthread_create( &slaves[i], NULL, Slave, NULL );
		if ( rc ) {
			retval = PAPI_ESYS;
			test_fail( __FILE__, __LINE__,
				"pthread_create", retval );
		}
	}

	for ( i = 0; i < nthr; i++ ) {
		pthread_join( slaves[i], NULL );
	}

	if (!quiet) {
		printf( "Expected: %lld Received: %lld\n",
			( long long ) nthr * thread_iter,
			count );
	}

	if ( nthr * thread_iter != count ) {
		test_fail( __FILE__, __LINE__, "Thread Locks", 1 );
	}

	test_pass( __FILE__ );

	return 0;

}
