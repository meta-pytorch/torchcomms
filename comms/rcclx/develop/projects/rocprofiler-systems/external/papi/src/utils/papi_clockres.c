/** file clockres.c
  *
  * @page papi_clockres
  * @brief The papi_clockres utility.
  *	@section Name
  * papi_clockres - measures and reports clock latency and resolution for PAPI timers. 
  *
  * @section Synopsis
  *	@section Description
  *	papi_clockres is a PAPI utility program that measures and reports the
  *	latency and resolution of the four PAPI timer functions:
  *	PAPI_get_real_cyc(), PAPI_get_virt_cyc(), PAPI_get_real_usec() and PAPI_get_virt_usec().
  *
  *	@section Options
  *		This utility has no command line options.
  *
  *	@section Bugs
  *	There are no known bugs in this utility.
  *	If you find a bug, it should be reported to the PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
  *
  */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"

#include "../testlib/clockcore.h"

int
main( int argc, char **argv )
{
	(void) argc;
	(void) argv;

	int retval;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error with PAPI init!\n");
		return 1;
	}

	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"Error with PAPI_set_debug!\n");
		return 1;
	}

	printf( "Printing Clock latency and resolution.\n" );
	printf( "-----------------------------------------------\n" );

	retval=clockcore( 0 );
	if (retval<0) {
		fprintf(stderr,"Error reading clock!\n");
		return retval;
	}

	return 0;
}
