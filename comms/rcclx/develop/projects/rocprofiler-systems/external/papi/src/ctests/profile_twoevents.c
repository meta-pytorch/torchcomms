/*
* File:    profile_twoevents.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file performs the following test: profiling two events */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"
#include "prof_utils.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int i, num_tests = 6;
	unsigned long length, blength;
	int num_buckets, mask;
	char title[PAPI_2MAX_STR_LEN];
	int retval;
	const PAPI_exe_info_t *prginfo;
	vptr_t start, end;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

        if ( ( prginfo = PAPI_get_executable_info(  ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_executable_info", 1 );
	}

	mask = prof_events( num_tests );
	start = prginfo->address_info.text_start;
	end = prginfo->address_info.text_end;

	/* Must have at least FP instr or Tot ins */

	if ( ( ( mask & MASK_FP_INS ) == 0 ) && ( ( mask & MASK_TOT_INS ) == 0 ) ) {
		if (!quiet) printf("No events could be added\n");
		test_skip( __FILE__, __LINE__, "No FP or Total Ins. event", 1 );
	}

	if ( start > end )
		test_fail( __FILE__, __LINE__, "Profile length < 0!", 0 );
	length = ( unsigned long ) ( end - start );

	if (!quiet) {
		prof_print_address( "Test case profile: POSIX compatible profiling with two events.\n",
					prginfo );
		prof_print_prof_info( start, end, THRESHOLD, event_name );
	}

	prof_alloc( 2, length );

	blength =
		prof_size( length, FULL_SCALE, PAPI_PROFIL_BUCKET_16, &num_buckets );
	do_no_profile( quiet );

	if ( !quiet ) {
		printf( "Test type   : \tPAPI_PROFIL_POSIX\n" );
	}
	if ( ( retval =
		   PAPI_profil( profbuf[0], ( unsigned int ) blength, start, FULL_SCALE,
						EventSet, PAPI_event, THRESHOLD,
						PAPI_PROFIL_POSIX ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}
	if ( ( retval =
		   PAPI_profil( profbuf[1], ( unsigned int ) blength, start, FULL_SCALE,
						EventSet, PAPI_TOT_CYC, THRESHOLD,
						PAPI_PROFIL_POSIX ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );

	do_stuff(  );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	if ( ( retval = PAPI_stop( EventSet, values[1] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( !quiet ) {
		printf( TAB1, event_name, ( values[1] )[0] );
		printf( TAB1, "PAPI_TOT_CYC:", ( values[1] )[1] );
	}
	if ( ( retval =
		   PAPI_profil( profbuf[0], ( unsigned int ) blength, start, FULL_SCALE,
						EventSet, PAPI_event, 0,
						PAPI_PROFIL_POSIX ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );

	if ( ( retval =
		   PAPI_profil( profbuf[1], ( unsigned int ) blength, start, FULL_SCALE,
						EventSet, PAPI_TOT_CYC, 0,
						PAPI_PROFIL_POSIX ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_profil", retval );

	sprintf( title, "   \t\t    %s\tPAPI_TOT_CYC\naddress\t\t\tcounts\tcounts\n",
			 event_name );

	if (!quiet) {
		prof_head( blength, PAPI_PROFIL_BUCKET_16, num_buckets, title );
		prof_out( start, 2, PAPI_PROFIL_BUCKET_16, num_buckets, FULL_SCALE );
	}

	remove_test_events( &EventSet, mask );

	retval = prof_check( 2, PAPI_PROFIL_BUCKET_16, num_buckets );

	for ( i = 0; i < 2; i++ ) {
		free( profbuf[i] );
	}

	if ( retval == 0 ) {
		test_fail( __FILE__, __LINE__, "No information in buffers", 1 );
	}

	test_pass( __FILE__ );

	return 0;
}
