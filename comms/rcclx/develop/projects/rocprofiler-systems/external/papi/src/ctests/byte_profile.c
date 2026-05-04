/*
* File:    byte_profile.c
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
*/

/* This file profiles multiple events with byte level address resolution.
   It's patterned after code suggested by John Mellor-Crummey, Rob Fowler,
   and Nathan Tallent.
   It is intended to illustrate the use of Multiprofiling on a very tight
   block of code at byte level resolution of the instruction addresses.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"
#include "prof_utils.h"

#include "do_loops.h"

#define PROFILE_ALL

static const PAPI_hw_info_t *hw_info;

static int num_events = 0;

#define N (1 << 23)
#define T (10)

double aa[N], bb[N];
double s = 0, s2 = 0;

static void
cleara( double a[N] )
{
	int i;

	for ( i = 0; i < N; i++ ) {
		a[i] = 0;
	}
}

static int
my_dummy( int i )
{
	return ( i + 1 );
}

static void
my_main( void )
{
	int i, j;

	for ( j = 0; j < T; j++ ) {
		for ( i = 0; i < N; i++ ) {
			bb[i] = 0;
		}
		cleara( aa );
		memset( aa, 0, sizeof ( aa ) );
		for ( i = 0; i < N; i++ ) {
			s += aa[i] * bb[i];
			s2 += aa[i] * aa[i] + bb[i] * bb[i];
		}
	}
}

static int
do_profile( vptr_t start, unsigned long plength, unsigned scale, int thresh,
	    int bucket, unsigned int mask ) {

	int i, retval;
	unsigned long blength;
	int num_buckets,j=0;

	int num_bufs = num_events;
	int event = num_events;

	int events[MAX_TEST_EVENTS];
	char header[BUFSIZ];

	strncpy(header,"address\t\t",BUFSIZ);

	//= "address\t\t\tcyc\tins\tfp_ins\n";

	for(i=0;i<MAX_TEST_EVENTS;i++) {
	  if (mask & test_events[i].mask) {
	    events[j]=test_events[i].event;

	    if (events[j]==PAPI_TOT_CYC) {
	       strncat(header,"\tcyc",BUFSIZ-1);
	    }
	    if (events[j]==PAPI_TOT_INS) {
	       strncat(header,"\tins",BUFSIZ-1);
	    }
	    if (events[j]==PAPI_FP_INS) {
	       strncat(header,"\tfp_ins",BUFSIZ-1);
	    }
	    if (events[j]==PAPI_FP_OPS) {
	       strncat(header,"\tfp_ops",BUFSIZ-1);
	    }
	    if (events[j]==PAPI_L2_TCM) {
	       strncat(header,"\tl2_tcm",BUFSIZ-1);
	    }

	    j++;

	  }
	}

	strncat(header,"\n",BUFSIZ-1);



	blength = prof_size( plength, scale, bucket, &num_buckets );
	prof_alloc( num_bufs, blength );

	if ( !TESTS_QUIET ) printf( "Overall event counts:\n" );

	for ( i = 0; i < num_events; i++ ) {
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, events[i], thresh,
							PAPI_PROFIL_POSIX | bucket ) ) != PAPI_OK ) {
	           if (retval == PAPI_EINVAL) {
		      test_warn( __FILE__, __LINE__, "Trying to profile with derived event", 1);
		      num_events=i;
		      break;
		   }
                   else {
		        printf("Failed with event %d %#x\n",i,events[i]);
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		   }
		}
	}

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	my_main(  );

	if ( ( retval = PAPI_stop( EventSet, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		printf( TAB1, "PAPI_TOT_CYC:", ( values[0] )[--event] );
		if ( strcmp( hw_info->model_string, "POWER6" ) != 0 ) {
			printf( TAB1, "PAPI_TOT_INS:", ( values[0] )[--event] );
		}
#if defined(__powerpc__)
		printf( TAB1, "PAPI_FP_INS", ( values[0] )[--event] );
#else
		if ( strcmp( hw_info->model_string, "Intel Pentium III" ) != 0 ) {
			printf( TAB1, "PAPI_FP_OPS:", ( values[0] )[--event] );
			printf( TAB1, "PAPI_L2_TCM:", ( values[0] )[--event] );
		}
#endif
	}

	for ( i = 0; i < num_events; i++ ) {
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, events[i], 0,
							PAPI_PROFIL_POSIX ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}

	if (!TESTS_QUIET) {
		prof_head( blength, bucket, num_buckets, header );
		prof_out( start, num_events, bucket, num_buckets, scale );
	}
	retval = prof_check( num_bufs, bucket, num_buckets );
	for ( i = 0; i < num_bufs; i++ ) {
		free( profbuf[i] );
	}
	return retval;
}



int
main( int argc, char **argv )
{
	long length;
	int mask;
	int retval;
	const PAPI_exe_info_t *prginfo;
	vptr_t start, end;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT) {
                test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if ( ( prginfo = PAPI_get_executable_info(  ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_executable_info", 1 );
	}

	hw_info = PAPI_get_hardware_info(  );
        if ( hw_info == NULL ) {
	  test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

       	mask = MASK_TOT_CYC | MASK_TOT_INS | MASK_FP_OPS | MASK_L2_TCM;

#if defined(__powerpc__)
	if ( strcmp( hw_info->model_string, "POWER6" ) == 0 )
		mask = MASK_TOT_CYC | MASK_FP_INS;
	else
		mask = MASK_TOT_CYC | MASK_TOT_INS | MASK_FP_INS;
#endif

#if defined(ITANIUM2)
	mask = MASK_TOT_CYC | MASK_FP_OPS | MASK_L2_TCM | MASK_L1_DCM;
#endif
	EventSet = add_test_events( &num_events, &mask, 0 );
	if (num_events==0) {
		if (!quiet) printf("Trouble adding events\n");
		test_skip(__FILE__,__LINE__,"add_test_events",2);
	}
	values = allocate_test_space( 1, num_events );

/* profile the cleara and my_main address space */
	start = ( vptr_t ) cleara;
	end = ( vptr_t ) my_dummy;

/* Itanium and PowerPC64 processors return function descriptors instead
 * of function addresses. You must dereference the descriptor to get the address.
*/
#if defined(ITANIUM1) || defined(ITANIUM2) \
    || (defined(__powerpc64__) && (_CALL_ELF != 2))
	start = ( vptr_t ) ( ( ( struct fdesc * ) start )->ip );
	end = ( vptr_t ) ( ( ( struct fdesc * ) end )->ip );
        /* PPC64 Big Endian is ELF version 1 which uses function descriptors.
         *  PPC64 Little Endian is ELF version 2 which does not use
         * function descriptors
         */
#endif

	/* call dummy so it doesn't get optimized away */
	retval = my_dummy( 1 );

	length = end - start;
	if ( length < 0 )
		test_fail( __FILE__, __LINE__, "Profile length < 0!", ( int ) length );

	if (!quiet) {
		prof_print_address( "Test case byte_profile: "
				"Multi-event profiling at byte resolution.\n",
				prginfo );
		prof_print_prof_info( start, end, THRESHOLD, event_name );
	}

	retval = do_profile( start, ( unsigned ) length,
			    FULL_SCALE * 2, THRESHOLD,
			    PAPI_PROFIL_BUCKET_32, mask );

	remove_test_events( &EventSet, mask );

	if (retval == 0) {
		test_fail( __FILE__, __LINE__, "No information in buffers", 1 );
	}

	test_pass( __FILE__ );

	return 0;
}




