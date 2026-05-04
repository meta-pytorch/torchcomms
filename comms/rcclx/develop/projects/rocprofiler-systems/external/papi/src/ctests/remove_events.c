/* This test checks if removing events works properly at the low level

   by Vince Weaver (vweaver1@eecs.utk.edu)

  */

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval;
	int EventSet = PAPI_NULL;
	long long values1[2],values2[2];
	const char *event_names[] = {"PAPI_TOT_CYC","PAPI_TOT_INS"};
	char add_event_str[PAPI_MAX_STR_LEN];
	double instructions_error;
	long long old_instructions;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create an empty event set */
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* add the events named above */
	retval = PAPI_add_named_event( EventSet, event_names[0] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[0] );
		if (!quiet) printf("Trouble %s\n",add_event_str);
		test_skip( __FILE__, __LINE__, add_event_str, retval );
	}

	retval = PAPI_add_named_event( EventSet, event_names[1] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[1] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}

	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our test code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	retval = PAPI_stop( EventSet, values1 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}


	old_instructions=values1[1];

	if ( !quiet ) {

	   printf( "========================\n" );

	   /* cycles is first, other event second */
	   sprintf( add_event_str, "%-12s : \t", event_names[0] );
	   printf( TAB1, add_event_str, values1[0] );
	   sprintf( add_event_str, "%-12s : \t", event_names[1] );
           printf( TAB1, add_event_str, values1[1] );
	}


	/* remove PAPI_TOT_CYC */
	retval = PAPI_remove_named_event( EventSet, event_names[0] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[0] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}


	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our test code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	retval = PAPI_stop( EventSet, values2 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}


	/* test if after removing the event, the second event */
	/* still points to the proper native event            */

	/* this only works if IPC != 1 */

	if ( !quiet ) {

	   printf( "==========================\n" );
	   printf( "After removing PAP_TOT_CYC\n");
	   sprintf( add_event_str, "%-12s : \t", event_names[1] );
           printf( TAB1, add_event_str, values2[0] );

	   instructions_error=((double)old_instructions - (double)values2[0])/
	     (double)old_instructions;
	   if (instructions_error>10.0) {
	     printf("Error of %.2f%%\n",instructions_error);
	     test_fail( __FILE__, __LINE__, "validation", 0 );
	   }

	}
	test_pass( __FILE__ );

	return 0;
}
