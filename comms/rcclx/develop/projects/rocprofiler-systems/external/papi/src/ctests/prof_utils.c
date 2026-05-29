/*
* File:    prof_utils.c
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
*/

/* This file contains utility functions useful for all profiling tests
   It can be used by:
   - profile.c,
   - sprofile.c,
   - profile_pthreads.c,
   - profile_twoevents.c,
   - earprofile.c,
   - future profiling tests.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#include "prof_utils.h"

/* variables global to profiling tests */
long long **values;
char event_name[PAPI_MAX_STR_LEN];
int PAPI_event;
int EventSet = PAPI_NULL;
void *profbuf[5];


/* Many profiling tests count one of {FP_INS, FP_OPS, TOT_INS} and TOT_CYC.
   This function creates an event set containing the appropriate pair of events.
   It also initializes the global event_name string to the event selected.
   Assumed globals: EventSet, PAPI_event, event_name.
*/
int
prof_events( int num_tests)
{
	int retval;
	int num_events, mask;

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depends on the availability of the event on the
	   platform */
	EventSet = add_two_nonderived_events( &num_events, &PAPI_event, &mask );

	if (num_events==0) {
		return 0;
	}

	values = allocate_test_space( num_tests, num_events );

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	return mask;
}

/* This function displays info from the prginfo structure in a standardized format.
*/
void
prof_print_address( const char *title, const PAPI_exe_info_t * prginfo )
{
	printf( "%s\n", title );
	printf
		( "----------------------------------------------------------------\n" );
	printf( "Text start: %p, Text end: %p, Text length: %#x\n",
			prginfo->address_info.text_start, prginfo->address_info.text_end,
			( unsigned int ) ( prginfo->address_info.text_end -
							   prginfo->address_info.text_start ) );
	printf( "Data start: %p, Data end: %p\n", prginfo->address_info.data_start,
			prginfo->address_info.data_end );
	printf( "BSS start : %p, BSS end : %p\n", prginfo->address_info.bss_start,
			prginfo->address_info.bss_end );

	printf
		( "----------------------------------------------------------------\n" );
}

/* This function displays profining information useful for several profile tests.
   It (probably inappropriately) assumes use of a common THRESHOLD. This should
   probably be a passed parameter.
   Assumed globals: event_name, start, stop.
*/
void
prof_print_prof_info( vptr_t start, vptr_t end, int threshold,
					  char *event_name )
{
	printf( "Profiling event  : %s\n", event_name );
	printf( "Profile Threshold: %d\n", threshold );
	printf( "Profile Iters    : %d\n",
			( getenv( "NUM_ITERS" ) ? atoi( getenv( "NUM_ITERS" ) ) :
			  NUM_ITERS ) );
	printf( "Profile Range    : %p to %p\n", start, end );
	printf
		( "----------------------------------------------------------------\n" );
	printf( "\n" );
}

/* Most profile tests begin by counting the eventset with no profiling enabled.
   This function does that work. It assumes that the 'work' routine is do_both().
   A better implementation would pass a pointer to the work function.
   Assumed globals: EventSet, values, event_name.
*/
void
do_no_profile( int quiet )
{
	int retval;

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( getenv( "NUM_ITERS" ) ? atoi( getenv( "NUM_ITERS" ) ) :
			  NUM_ITERS );

	if ( ( retval = PAPI_stop( EventSet, values[0] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if (!quiet) {
		printf( "Test type   : \t%s\n", "No profiling" );
		printf( TAB1, event_name, ( values[0] )[0] );
		printf( TAB1, "PAPI_TOT_CYC", ( values[0] )[1] );
	}
}

/* This routine allocates and initializes up to 5 equal sized profiling buffers.
   They need to be freed when profiling is completed.
   The number and size are passed parameters.
   The profbuf[] array of void * pointers is an assumed global.
   It should be cast to the required type by the parent routine.
*/
void
prof_alloc( int num, unsigned long blength )
{
	int i;

	for ( i = 0; i < num; i++ ) {
		profbuf[i] = malloc( blength );
		if ( profbuf[i] == NULL ) {
			test_fail( __FILE__, __LINE__, "malloc", PAPI_ESYS );
		}
		memset( profbuf[i], 0x00, blength );
	}
}

/* Given the profiling type (16, 32, or 64) this function returns the 
   bucket size in bytes. NOTE: the bucket size does not ALWAYS correspond
   to the expected value, esp on architectures like Cray with weird data types.
   This is necessary because the posix_profile routine in extras.c relies on
   the data types and sizes produced by the compiler.
*/
int
prof_buckets( int bucket )
{
	int bucket_size;
	switch ( bucket ) {
	case PAPI_PROFIL_BUCKET_16:
		bucket_size = sizeof ( short );
		break;
	case PAPI_PROFIL_BUCKET_32:
		bucket_size = sizeof ( int );
		break;
	case PAPI_PROFIL_BUCKET_64:
		bucket_size = sizeof ( unsigned long long );
		break;
	default:
		bucket_size = 0;
		break;
	}
	return ( bucket_size );
}

/* A standardized header printing routine. No assumed globals.
*/
void
prof_head( unsigned long blength, int bucket, int num_buckets, const char *header )
{
	int bucket_size = prof_buckets( bucket );
	printf
		( "\n------------------------------------------------------------\n" );
	printf( "PAPI_profil() hash table, Bucket size: %d bits.\n",
			bucket_size * 8 );
	printf( "Number of buckets: %d.\nLength of buffer: %ld bytes.\n",
			num_buckets, blength );
	printf( "------------------------------------------------------------\n" );
	printf( "%s\n", header );
}

/* This function prints a standardized profile output based on the bucket size.
   A row consisting of an address and 'n' data elements is displayed for each
   address with at least one non-zero bucket.
   Assumes global profbuf[] array pointers.
*/
void
prof_out( vptr_t start, int n, int bucket, int num_buckets,
		  unsigned int scale )
{
	int i, j;
	unsigned short buf_16;
	unsigned int buf_32;
	unsigned long long buf_64;
	unsigned short **buf16 = ( unsigned short ** ) profbuf;
	unsigned int **buf32 = ( unsigned int ** ) profbuf;
	unsigned long long **buf64 = ( unsigned long long ** ) profbuf;

	if ( !TESTS_QUIET ) {
		/* printf("%#lx\n",(unsigned long) start + (unsigned long) (2 * i)); */
		/* printf("start: %p; i: %#x; scale: %#x; i*scale: %#x; i*scale >>15: %#x\n", start, i, scale, i*scale, (i*scale)>>15); */
		switch ( bucket ) {
		case PAPI_PROFIL_BUCKET_16:
			for ( i = 0; i < num_buckets; i++ ) {
				for ( j = 0, buf_16 = 0; j < n; j++ )
					buf_16 |= ( buf16[j] )[i];
				if ( buf_16 ) {
/* On 32bit builds with gcc 4.3 gcc complained about casting vptr_t => long long
 * Thus the unsigned long to long long cast */
					printf( "%#-16llx",
						(long long) (unsigned long)start +
						( ( ( long long ) i * scale ) >> 15 ) );
					for ( j = 0, buf_16 = 0; j < n; j++ )
						printf( "\t%d", ( buf16[j] )[i] );
					printf( "\n" );
				}
			}
			break;
		case PAPI_PROFIL_BUCKET_32:
			for ( i = 0; i < num_buckets; i++ ) {
				for ( j = 0, buf_32 = 0; j < n; j++ )
					buf_32 |= ( buf32[j] )[i];
				if ( buf_32 ) {
					printf( "%#-16llx",
						(long long) (unsigned long)start +
						( ( ( long long ) i * scale ) >> 15 ) );
					for ( j = 0, buf_32 = 0; j < n; j++ )
						printf( "\t%d", ( buf32[j] )[i] );
					printf( "\n" );
				}
			}
			break;
		case PAPI_PROFIL_BUCKET_64:
			for ( i = 0; i < num_buckets; i++ ) {
				for ( j = 0, buf_64 = 0; j < n; j++ )
					buf_64 |= ( buf64[j] )[i];
				if ( buf_64 ) {
					printf( "%#-16llx",
						(long long) (unsigned long)start +
					        ( ( ( long long ) i * scale ) >> 15 ) );
					for ( j = 0, buf_64 = 0; j < n; j++ )
						printf( "\t%lld", ( buf64[j] )[i] );
					printf( "\n" );
				}
			}
			break;
		}
		printf
			( "------------------------------------------------------------\n\n" );
	}
}

/* This function checks to make sure that some buffer value somewhere is nonzero.
   If all buffers are empty, zero is returned. This usually indicates a profiling
   failure. Assumes global profbuf[].
*/
int
prof_check( int n, int bucket, int num_buckets )
{
	int i, j;
	int retval = 0;
	unsigned short **buf16 = ( unsigned short ** ) profbuf;
	unsigned int **buf32 = ( unsigned int ** ) profbuf;
	unsigned long long **buf64 = ( unsigned long long ** ) profbuf;

	switch ( bucket ) {
	case PAPI_PROFIL_BUCKET_16:
		for ( i = 0; i < num_buckets; i++ )
			for ( j = 0; j < n; j++ )
				retval = retval || buf16[j][i];
		break;
	case PAPI_PROFIL_BUCKET_32:
		for ( i = 0; i < num_buckets; i++ )
			for ( j = 0; j < n; j++ )
				retval = retval || buf32[j][i];
		break;
	case PAPI_PROFIL_BUCKET_64:
		for ( i = 0; i < num_buckets; i++ )
			for ( j = 0; j < n; j++ )
				retval = retval || buf64[j][i];
		break;
	}
	return ( retval );
}

/* Computes the length (in bytes) of the buffer required for profiling.
   'plength' is the profile length, or address range to be profiled.
   By convention, it is assumed that there are half as many buckets as addresses.
   The scale factor is a fixed point fraction in which 0xffff = ~1
                                                         0x8000 = 1/2
                                                         0x4000 = 1/4, etc.
   Thus, the number of profile buckets is (plength/2) * (scale/65536),
   and the length (in bytes) of the profile buffer is buckets * bucket size.
   */
unsigned long
prof_size( unsigned long plength, unsigned scale, int bucket, int *num_buckets )
{
	unsigned long blength;
	long long llength = ( ( long long ) plength * scale );
	int bucket_size = prof_buckets( bucket );
	*num_buckets = ( int ) ( llength / 65536 / 2 );
	blength = ( unsigned long ) ( *num_buckets * bucket_size );
	return ( blength );
}
