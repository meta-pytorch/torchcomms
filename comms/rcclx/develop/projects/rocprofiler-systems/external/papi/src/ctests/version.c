/* This file performs the following test: */
/* compare and report versions from papi.h and the papi library */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int main( int argc, char **argv ) {

	int init_version, lib_version;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	init_version = PAPI_library_init( PAPI_VER_CURRENT );

	lib_version = PAPI_get_opt( PAPI_LIB_VERSION, NULL );
	if (lib_version == PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_opt", PAPI_EINVAL );
	}

	if ( !quiet) {
		printf( "Version.c: Compare and report versions from papi.h and the papi library.\n" );
		printf( "-------------------------------------------------------------------------\n" );
		printf( "                    MAJOR  MINOR  REVISION  INCREMENT\n" );
		printf( "-------------------------------------------------------------------------\n" );

		printf( "PAPI_VER_CURRENT : %4d %6d %7d %10d\n",
				PAPI_VERSION_MAJOR( PAPI_VER_CURRENT ),
				PAPI_VERSION_MINOR( PAPI_VER_CURRENT ),
				PAPI_VERSION_REVISION( PAPI_VER_CURRENT ),
				PAPI_VERSION_INCREMENT( PAPI_VER_CURRENT ) );
		printf( "PAPI_library_init: %4d %6d %7d %10d\n",
				PAPI_VERSION_MAJOR( init_version ),
				PAPI_VERSION_MINOR( init_version ),
				PAPI_VERSION_REVISION( init_version ),
				PAPI_VERSION_INCREMENT( init_version ) );
		printf( "PAPI_VERSION     : %4d %6d %7d %10d\n",
				PAPI_VERSION_MAJOR( PAPI_VERSION ),
				PAPI_VERSION_MINOR( PAPI_VERSION ),
				PAPI_VERSION_REVISION( PAPI_VERSION ),
				PAPI_VERSION_INCREMENT (PAPI_VERSION) );
		printf( "PAPI_get_opt     : %4d %6d %7d %10d\n",
				PAPI_VERSION_MAJOR( lib_version ),
				PAPI_VERSION_MINOR( lib_version ),
				PAPI_VERSION_REVISION( lib_version ),
				PAPI_VERSION_INCREMENT( lib_version) );

		printf( "-------------------------------------------------------------------------\n" );
	}

	if ( lib_version != PAPI_VERSION ) {
		test_fail( __FILE__, __LINE__, "Version Mismatch", PAPI_EINVAL );
	}

	test_pass( __FILE__ );

	return 0;
}
