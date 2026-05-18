#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval;
	int preset;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	retval = PAPI_event_name_to_code( "PAPI_FP_INS", &preset );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
	if ( preset != PAPI_FP_INS )
		test_fail( __FILE__, __LINE__, "Wrong preset returned", retval );

	retval = PAPI_event_name_to_code( "PAPI_TOT_CYC", &preset );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
	if ( preset != PAPI_TOT_CYC )
		test_fail( __FILE__, __LINE__,
				   "*preset returned did not equal PAPI_TOT_CYC", retval );

	test_pass( __FILE__ );

	return 0;

}
