/*
* File:    profile.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@cs.utk.edu
*/

/* This file performs the following test: profiling and program info option call

   - This tests the SVR4 profiling interface of PAPI. These are counted
   in the default counting domain and default granularity, depending on
   the platform. Usually this is the user domain (PAPI_DOM_USER) and
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (to profile)
     + PAPI_TOT_CYC

   - Set up profile
   - Start eventset 1
   - Do both (flops and reads)
   - Stop eventset 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"
#include "prof_utils.h"

#include "do_loops.h"

#undef THRESHOLD
#define THRESHOLD 1000

static void
ear_no_profile( void )
{
	int retval;

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_l1misses( 10000 );

	if ( ( retval = PAPI_stop( EventSet, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	printf( "Test type   : \tNo profiling\n" );
	printf( TAB1, event_name, ( values[0] )[0] );
	printf( TAB1, "PAPI_TOT_CYC:", ( values[0] )[1] );
}

static int
do_profile( vptr_t start, unsigned long plength, unsigned scale, int thresh,
			int bucket )
{
	int i, retval;
	unsigned long blength;
	int num_buckets;
	const char *profstr[2] = { "PAPI_PROFIL_POSIX", "PAPI_PROFIL_INST_EAR" };
	int profflags[2] =
		{ PAPI_PROFIL_POSIX, PAPI_PROFIL_POSIX | PAPI_PROFIL_INST_EAR };
	int num_profs;

	do_stuff(  );

	num_profs = sizeof ( profflags ) / sizeof ( int );
	ear_no_profile(  );
	blength = prof_size( plength, scale, bucket, &num_buckets );
	prof_alloc( num_profs, blength );

	for ( i = 0; i < num_profs; i++ ) {
		if ( !TESTS_QUIET )
			printf( "Test type   : \t%s\n", profstr[i] );

		if ( ( retval = PAPI_profil( profbuf[i], blength, start, scale,
									 EventSet, PAPI_event, thresh,
									 profflags[i] | bucket ) ) != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		}
		if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );

		do_stuff(  );

		if ( ( retval = PAPI_stop( EventSet, values[1] ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

		if ( !TESTS_QUIET ) {
			printf( TAB1, event_name, ( values[1] )[0] );
			printf( TAB1, "PAPI_TOT_CYC:", ( values[1] )[1] );
		}
		if ( ( retval = PAPI_profil( profbuf[i], blength, start, scale,
									 EventSet, PAPI_event, 0,
									 profflags[i] ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}

	prof_head( blength, bucket, num_buckets,
			   "address\t\t\tPOSIX\tINST_DEAR\n" );
	prof_out( start, num_profs, bucket, num_buckets, scale );

	retval = prof_check( num_profs, bucket, num_buckets );

	for ( i = 0; i < num_profs; i++ ) {
		free( profbuf[i] );
	}

	return retval;
}


int
main( int argc, char **argv )
{
	int num_events, num_tests = 6;
	long length;
	int retval, retval2;
	const PAPI_hw_info_t *hw_info;
	const PAPI_exe_info_t *prginfo;
	vptr_t start, end;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if ( ( prginfo = PAPI_get_executable_info(  ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_executable_info", 1 );
	}

	if ( ( hw_info = PAPI_get_hardware_info(  ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 0 );
	}

	if ( ( strncasecmp( hw_info->model_string, "Itanium",
				strlen( "Itanium" ) ) != 0 ) &&
			( strncasecmp( hw_info->model_string, "32",
				strlen( "32" ) ) != 0 ) ) {
		if (!quiet) printf("Itanium only for now.\n");
		test_skip( __FILE__, __LINE__, "Test unsupported", PAPI_ENOIMPL );
	}

//	if ( quiet ) {
//	   test_skip( __FILE__, __LINE__,
//		     "Test deprecated in quiet mode for PAPI 3.6", 0 );
//	}

	sprintf( event_name, "DATA_EAR_CACHE_LAT4" );
	if ( ( retval =
		   PAPI_event_name_to_code( event_name, &PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( ( retval = PAPI_add_event( EventSet, PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	if ( ( retval = PAPI_add_event( EventSet, PAPI_TOT_CYC ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	num_events = 2;
	values = allocate_test_space( num_tests, num_events );

	/* use these lines to profile entire code address space */
	start = prginfo->address_info.text_start;
	end = prginfo->address_info.text_end;
	length = end - start;
	if ( length < 0 )
		test_fail( __FILE__, __LINE__, "Profile length < 0!", length );

	prof_print_address
		( "Test earprofile: POSIX compatible event address register profiling.\n",
		  prginfo );
	prof_print_prof_info( start, end, THRESHOLD, event_name );
	retval =
		do_profile( start, length, FULL_SCALE, THRESHOLD,
					PAPI_PROFIL_BUCKET_16 );

	retval2 = PAPI_remove_event( EventSet, PAPI_event );
	if ( retval2 == PAPI_OK )
		retval2 = PAPI_remove_event( EventSet, PAPI_TOT_CYC );
	if ( retval2 != PAPI_OK )
		test_fail( __FILE__, __LINE__, "Can't remove events", retval2 );

	if ( retval )
		test_pass( __FILE__ );
	else
		test_fail( __FILE__, __LINE__, "No information in buffers", 1 );

	return 1;

}
