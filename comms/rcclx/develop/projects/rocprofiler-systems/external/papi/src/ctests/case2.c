/* From Dave McNamara at PSRV. Thanks! */

/* If an event is countable but you've exhausted the counter resources
and you try to add an event, it seems subsequent PAPI_start and/or
PAPI_stop will causes a Seg. Violation.

   I got around this by calling PAPI to get the # of countable events,
then making sure that I didn't try to add more than these number of
events. I still have a problem if someone adds Level 2 cache misses
and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
2 counters. */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	double c, a = 0.999, b = 1.001;
	int n = 1000;
	int EventSet = PAPI_NULL;
	int retval;
	int j = 0, i;
	long long g1[3];

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( PAPI_query_event( PAPI_BR_CN ) == PAPI_OK )
		j++;

	if ( j == 1 &&
		 ( retval = PAPI_add_event( EventSet, PAPI_BR_CN ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	i = j;
	if ( PAPI_query_event( PAPI_TOT_CYC ) == PAPI_OK )
		j++;

	if ( j == ( i + 1 ) &&
		 ( retval = PAPI_add_event( EventSet, PAPI_TOT_CYC ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	i = j;
	if ( PAPI_query_event( PAPI_TOT_INS ) == PAPI_OK )
		j++;

	if ( j == ( i + 1 ) &&
		 ( retval = PAPI_add_event( EventSet, PAPI_TOT_INS ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if ( j ) {
		if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );

		for ( i = 0; i < n; i++ ) {
			c = a * b;
		}
		if (!TESTS_QUIET) fprintf(stdout,"c=%lf\n",c);
		if ( ( retval = PAPI_stop( EventSet, g1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	test_pass( __FILE__ );

	return 0;
}
