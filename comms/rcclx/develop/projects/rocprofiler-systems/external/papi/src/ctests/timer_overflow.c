/*
 * File:	timer_overflow.c
 * Author:	Kevin London
 *		london@cs.utk.edu
 * Mods:	<your name here>
 * 		<your email address>
 */

/* This file looks for possible timer overflows. */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define TIMER_THRESHOLD 100

int
main( int argc, char **argv )
{
	int sleep_time = TIMER_THRESHOLD;
	int retval, i;
	long long timer;

	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) )
			tests_quiet( argc, argv );
		else {
			sleep_time = atoi( argv[1] );
			if ( sleep_time <= 0 )
				sleep_time = TIMER_THRESHOLD;
		}
	}

	if ( TESTS_QUIET ) {
		/* Skip the test in TESTS_QUIET so that the main script doesn't
		 * run this as it takes a long time to check for overflow
		 */
		printf( "%-40s SKIPPED\nLine # %d\n", __FILE__, __LINE__ );
		printf( "timer_overflow takes a long time to run, run separately.\n" );
		exit( 0 );
	}

	printf( "This test will take about: %f minutes.\n",
			( float ) ( 20 * ( sleep_time / 60.0 ) ) );
	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	timer = PAPI_get_real_usec(  );
	for ( i = 0; i <= 20; i++ ) {
		if ( timer < 0 )
			break;
		sleep( ( unsigned int ) sleep_time );
		timer = PAPI_get_real_usec(  );
	}
	if ( timer < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_get_real_usec: overflow", 1 );
	else
		test_pass( __FILE__ );

	return 0;
}
