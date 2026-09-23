/*
 * This utility loops through all the PAPI error codes and displays them in
 *	table format
*/

/** file error_codes.c
  * @brief papi_error_codes utility.
  *	@page papi_error_codes
  *	@section  NAME
  *		papi_error_codes - lists all currently defined PAPI error codes. 
  *
  *	@section Synopsis
  *		papi_error_codes
  *
  *	@section Description
  *		papi_error_codes is a PAPI utility program that displays all defined
  *		error codes from papi.h and their error strings from papi_data.h.
  *		If an error string is not defined, a warning is generated. This can
  *		help trap newly defined error codes for which error strings are not
  *		yet defined.
  *
  *	@section Options
  *		This utility has no command line options.
  *
  *	@section Bugs
  *		There are no known bugs in this utility.
  *		If you find a bug, it should be reported to the
  *		PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>

#include "papi.h"

int
main( int argc, char **argv )
{
	int i=0;
	int retval;

	(void)argc;
	(void)argv;

 	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error with PAPI_library_init!\n");
		return retval;
	}

	printf( "\n----------------------------------\n" );
	printf( "For PAPI Version: %d.%d.%d.%d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ),
			PAPI_VERSION_INCREMENT( PAPI_VERSION ) );
	printf( "----------------------------------\n" );
	while ( 1 ) {
		char *errstr;
		errstr = PAPI_strerror( -i );

		if ( NULL == errstr ) {
		    break;
		}

		printf( "Error code %4d: %s\n", -i, errstr );
		i++;
	}
	printf( "There are %d error codes defined\n", i );
	printf( "----------------------------------\n\n" );

	return 0;
}
