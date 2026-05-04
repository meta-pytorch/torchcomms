/*
* File:    forkexec4.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file performs the following test:

                  PAPI_library_init()
       ** unlike forkexec2/forkexec3, no shutdown here **
                       fork()
                      /      \
                  parent    child
                  wait()   PAPI_library_init()
			   execlp()
			   PAPI_library_init()

 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

#include "papi.h"
#include "papi_test.h"


int
main( int argc, char **argv )
{
	int retval;
	int status;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if ( ( argc > 1 ) && ( strcmp( argv[1], "xxx" ) == 0 ) ) {
		/* In child */
		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT ) {
			test_fail( __FILE__, __LINE__, "execed PAPI_library_init", retval );
		}
		return 0;
	} else {
		if (!quiet) printf("Testing Init/fork/exec/Init\n");

		/* Init PAPI */
		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT ) {
			test_fail( __FILE__, __LINE__, "main PAPI_library_init", retval );
		}

		if ( fork(  ) == 0 ) {
			/* In Child */
			retval = PAPI_library_init( PAPI_VER_CURRENT );
			if ( retval != PAPI_VER_CURRENT ) {
				test_fail( __FILE__, __LINE__, "forked PAPI_library_init",
						   retval );
			}

			if ( execlp( argv[0], argv[0], "xxx", NULL ) == -1 ) {
				test_fail( __FILE__, __LINE__, "execlp", PAPI_ESYS );
			}
		} else {
			/* Waiting in parent */
			wait( &status );
			if ( WEXITSTATUS( status ) != 0 ) {
				test_fail( __FILE__, __LINE__, "fork", WEXITSTATUS( status ) );
		}	}
	}

	test_pass( __FILE__ );

	return 0;
}
