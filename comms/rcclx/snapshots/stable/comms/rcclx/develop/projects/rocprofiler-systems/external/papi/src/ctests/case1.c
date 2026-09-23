/* From Dave McNamara at PSRV. Thanks! */

/* If you try to add an event that doesn't exist, you get the correct error
message, yet you get subsequent Seg. Faults when you try to do PAPI_start and
PAPI_stop. I would expect some bizarre behavior if I had no events added to the
event set and then tried to PAPI_start but if I had successfully added one
event, then the 2nd one get an error when I tried to add it, is it possible for
PAPI_start to work but just count the first event?
*/

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
	int i, j = 0;
	long long g1[2];

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );


	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( PAPI_query_event( PAPI_L2_TCM ) == PAPI_OK )
		j++;

	if ( j == 1 &&
		 ( retval = PAPI_add_event( EventSet, PAPI_L2_TCM ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
		j--;				 /* The event was not added */
	}

	i = j;
	if ( PAPI_query_event( PAPI_L2_DCM ) == PAPI_OK )
		j++;

	if ( j == ( i + 1 ) &&
		 ( retval = PAPI_add_event( EventSet, PAPI_L2_DCM ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
		j--;				 /* The event was not added */
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
