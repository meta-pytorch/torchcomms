#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int FPEventSet = PAPI_NULL;
	long long values;
	int PAPI_event, retval;
	char event_name[PAPI_MAX_STR_LEN];
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

        /* init PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Use PAPI_FP_INS if available, otherwise use PAPI_TOT_INS */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK )
		PAPI_event = PAPI_FP_INS;
	else
		PAPI_event = PAPI_TOT_INS;

	retval = PAPI_query_event( PAPI_event );
	if (retval != PAPI_OK ) {
		if (!quiet) printf("Trouble querying event\n");
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );
	}

        /* Create the eventset */
	if ( ( retval = PAPI_create_eventset( &FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

        /* Add event to the eventset */
	if ( ( retval = PAPI_add_event( FPEventSet, PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	/* Start counting */
	if ( ( retval = PAPI_start( FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	/* Try to cleanup while running */
        /* Fail test if this isn't refused */
	if ( ( retval = PAPI_cleanup_eventset( FPEventSet ) ) != PAPI_EISRUN )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	/* Try to destroy eventset while running */
	/* Fail test if this isn't refused */
	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_EISRUN )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	/* do some work */
	do_flops( 1000000 );

	/* stop counting */
	if ( ( retval = PAPI_stop( FPEventSet, &values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	/* Try to destroy eventset without cleaning first */
	/* Fail test if this isn't refused */
	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_EINVAL )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	/* Try to cleanup eventset.  */
	/* This should pass.         */
	if ( ( retval = PAPI_cleanup_eventset( FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	/* Try to destroy eventset.  */
	/* This should pass.         */
	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	/* Make sure eventset was set to PAPI_NULL */
	if ( FPEventSet != PAPI_NULL )
		test_fail( __FILE__, __LINE__, "FPEventSet != PAPI_NULL", retval );

	if ( !quiet ) {
		if ( ( retval =
			   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

		printf( "Test case John May 2: cleanup / destroy eventset.\n" );
		printf( "-------------------------------------------------\n" );
		printf( "Test run    : \t1\n" );
		printf( "%s : \t", event_name );
		printf( LLDFMT, values );
		printf( "\n" );
		printf( "-------------------------------------------------\n" );
		printf( "The following messages will appear if PAPI is compiled with debug enabled:\n" );
		printf
			( "\tPAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n" );
		printf
			( "\tPAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n" );
		printf( "\tPAPI Error Code -1: PAPI_EINVAL: Invalid argument\n" );
	}
	test_pass( __FILE__ );

	return 0;
}
