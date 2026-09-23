/*
* File:    fork2.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test:

   PAPI_library_init()
         fork();
         /    \
     parent   child
     wait()   PAPI_shutdown()
              PAPI_library_init()

 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#include "papi.h"
#include "papi_test.h"


int
main( int argc, char **argv )
{
	int retval;
	int status;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "main PAPI_library_init", retval );

	if ( fork(  ) == 0 ) {
		PAPI_shutdown();

		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT )
			test_fail( __FILE__, __LINE__, "forked PAPI_library_init", retval );
		exit( 0 );
	} else {
		wait( &status );
		if ( WEXITSTATUS( status ) != 0 )
			test_fail( __FILE__, __LINE__, "fork", WEXITSTATUS( status ) );
	}

	test_pass( __FILE__ );

	return 0;

}
