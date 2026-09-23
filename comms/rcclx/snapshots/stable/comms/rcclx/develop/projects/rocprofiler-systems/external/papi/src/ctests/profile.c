/*
* File:    profile.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
*/

/* This file performs the following test:
	profiling and program info option call

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

#include "papi.h"
#include "papi_test.h"
#include "prof_utils.h"

#include "do_loops.h"

#define PROFILE_ALL

static int
do_profile( vptr_t start, unsigned long plength, unsigned scale, int thresh,
			int bucket )
{
	int i, retval;
	unsigned long blength;
	int num_buckets;

	const char *profstr[5] = { "PAPI_PROFIL_POSIX",
		"PAPI_PROFIL_RANDOM",
		"PAPI_PROFIL_WEIGHTED",
		"PAPI_PROFIL_COMPRESS",
		"PAPI_PROFIL_<all>"
	};

	int profflags[5] = { PAPI_PROFIL_POSIX,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED |
			PAPI_PROFIL_RANDOM | PAPI_PROFIL_COMPRESS
	};

	do_no_profile( TESTS_QUIET );
	blength = prof_size( plength, scale, bucket, &num_buckets );
	prof_alloc( 5, blength );

	for ( i = 0; i < 5; i++ ) {
		if ( !TESTS_QUIET ) {
			printf( "Test type   : \t%s\n", profstr[i] );
		}

#ifndef SWPROFILE
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, thresh,
							profflags[i] | bucket ) ) != PAPI_OK ) {
		   if (retval==PAPI_ENOSUPP) {
		      char warning[BUFSIZ];

		      sprintf(warning,"PAPI_profil %s not supported",
			      profstr[i]);
		      test_warn( __FILE__, __LINE__, warning, 1 );
		   }
		   else {
		      test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		   }
		}
#else
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, thresh,
							profflags[i] | bucket | PAPI_PROFIL_FORCE_SW ) ) !=
			 PAPI_OK ) {
		   test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		}
#endif

		if ( retval != PAPI_OK )
			break;

		if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );

		do_flops( getenv( "NUM_FLOPS" ) ? atoi( getenv( "NUM_FLOPS" ) ) :
				  NUM_FLOPS );

		if ( ( retval = PAPI_stop( EventSet, values[1] ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

		if ( !TESTS_QUIET ) {
			printf( TAB1, event_name, ( values[1] )[0] );
			printf( TAB1, "PAPI_TOT_CYC", ( values[1] )[1] );
		}
		retval = PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, 0,
							profflags[i] );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		}
	}

	if ( retval == PAPI_OK ) {
		if (!TESTS_QUIET) prof_head( blength, bucket, num_buckets,
				   "address\t\t\tflat\trandom\tweight\tcomprs\tall\n" );
		if (!TESTS_QUIET) prof_out( start, 5, bucket, num_buckets, scale );
		retval = prof_check( 5, bucket, num_buckets );
	}

	for ( i = 0; i < 5; i++ ) {
		free( profbuf[i] );
	}

	return retval;
}


int
main( int argc, char **argv )
{
	int num_tests = 6;
	long length;
	int mask;
	int retval;
	int mythreshold = THRESHOLD;
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

	retval = PAPI_query_event(PAPI_TOT_CYC);
	if (retval!=PAPI_OK) {
		if (!quiet) printf("No events found\n");
		test_skip(__FILE__, __LINE__,"No events found",1);
	}

	mask = prof_events( num_tests );

#ifdef PROFILE_ALL
/* use these lines to profile entire code address space */
	start = prginfo->address_info.text_start;
	end = prginfo->address_info.text_end;
#else
/* use these lines to profile only do_flops address space */
	start = ( vptr_t ) do_flops;
	end = ( vptr_t ) fdo_flops;
/* Itanium and ppc64 processors return function descriptors instead of function addresses.
   You must dereference the descriptor to get the address.
*/
#if defined(ITANIUM1) || defined(ITANIUM2) || defined(__powerpc64__)
	start = ( vptr_t ) ( ( ( struct fdesc * ) start )->ip );
	end = ( vptr_t ) ( ( ( struct fdesc * ) end )->ip );
#endif
#endif

#if defined(linux)
	{
		char *tmp = getenv( "THRESHOLD" );
		if ( tmp )
			mythreshold = atoi( tmp );
	}
#endif

	length = end - start;
	if ( length < 0 ) {
		test_fail( __FILE__, __LINE__, "Profile length < 0!", ( int ) length );
	}

	if (!quiet) {
		prof_print_address( "Test case profile: "
			"POSIX compatible profiling with hardware counters.\n",
			prginfo );
		prof_print_prof_info( start, end, mythreshold, event_name );
	}

	retval = do_profile( start, ( unsigned long ) length, FULL_SCALE,
				mythreshold, PAPI_PROFIL_BUCKET_16 );
	if ( retval == PAPI_OK ) {
		retval = do_profile( start, ( unsigned long ) length,
						FULL_SCALE, mythreshold,
						PAPI_PROFIL_BUCKET_32 );
	}
	if ( retval == PAPI_OK ) {
		retval = do_profile( start, ( unsigned long ) length,
						FULL_SCALE, mythreshold,
						PAPI_PROFIL_BUCKET_64 );
	}

	remove_test_events( &EventSet, mask );

	test_pass( __FILE__ );

	return 0;
}

