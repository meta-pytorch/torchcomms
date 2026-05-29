/* This file performs the following test: 

   - make an event set with PAPI_TOT_INS and PAPI_TOT_CYC.
   - enable per node counting
   - enable full domain counting
   - sleeps for 5 seconds
   - print the results
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi_test.h"

int
main(  )
{
	int ncpu, nctr, i, actual_domain;
	int retval;
	int EventSet = PAPI_NULL;
	long long *values;
	long long elapsed_us, elapsed_cyc;
	PAPI_option_t options;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf( stderr, "Library mismatch: code %d, library %d\n", retval,
				 PAPI_VER_CURRENT );
		exit( 1 );
	}

	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
		exit( 1 );

	/* Set the domain as high as it will go. */

	options.domain.eventset = EventSet;
	options.domain.domain = PAPI_DOM_ALL;
	retval = PAPI_set_opt( PAPI_DOMAIN, &options );
	if ( retval != PAPI_OK )
		exit( 1 );
	actual_domain = options.domain.domain;

	/* This should only happen to an empty eventset */

	options.granularity.eventset = EventSet;
	options.granularity.granularity = PAPI_GRN_SYS_CPU;
	retval = PAPI_set_opt( PAPI_GRANUL, &options );
	if ( retval != PAPI_OK )
		exit( 1 );

	/* Malloc the output array */

	ncpu = PAPI_get_opt( PAPI_MAX_CPUS, NULL );
	nctr = PAPI_get_opt( PAPI_MAX_HWCTRS, NULL );
	values = ( long long * ) malloc( ncpu * nctr * sizeof ( long long ) );
	memset( values, 0x0, ( ncpu * nctr * sizeof ( long long ) ) );

	/* Add the counters */

	if ( PAPI_add_event( EventSet, PAPI_TOT_CYC ) != PAPI_OK )
		exit( 1 );

	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
		exit( 1 );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		exit( 1 );

	sleep( 5 );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		exit( 1 );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	printf( "Test case: per node\n" );
	printf( "-------------------\n\n" );

	printf( "This machine has %d cpus, each with %d counters.\n", ncpu, nctr );
	printf( "Test case asked for: PAPI_DOM_ALL\n" );
	printf( "Test case got: " );
	if ( actual_domain & PAPI_DOM_USER )
		printf( "PAPI_DOM_USER " );
	if ( actual_domain & PAPI_DOM_KERNEL )
		printf( "PAPI_DOM_KERNEL " );
	if ( actual_domain & PAPI_DOM_OTHER )
		printf( "PAPI_DOM_OTHER " );
	printf( "\n" );

	for ( i = 0; i < ncpu; i++ ) {
		printf( "CPU %d\n", i );
		printf( "PAPI_TOT_CYC: \t%lld\n", values[0 + i * nctr] );
		printf( "PAPI_TOT_INS: \t%lld\n", values[1 + i * nctr] );
	}

	printf
		( "\n-------------------------------------------------------------------------\n" );

	printf( "Real usec   : \t%lld\n", elapsed_us );
	printf( "Real cycles : \t%lld\n", elapsed_cyc );

	printf
		( "-------------------------------------------------------------------------\n" );

	free( values );

	PAPI_shutdown(  );

	exit( 0 );
}
