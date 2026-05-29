/* This test exercises the PAPI_{query, add, remove}_event APIs
   for PRESET events.
   It more or less duplicates the functionality of the classic "zero" test.
*/

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval, num_tests = 1, tmp;
	int EventSet = PAPI_NULL;
	int num_events = 2;
	long long **values;
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	const char *event_names[] = {"PAPI_TOT_CYC","PAPI_TOT_INS"};
	char add_event_str[PAPI_MAX_STR_LEN];
	double cycles_error;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Verify that the named events exist */
	retval = PAPI_query_named_event(event_names[0]);
	if ( retval == PAPI_OK) {
		retval = PAPI_query_named_event(event_names[1]);
	}
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Trouble querying events\n");
		test_skip( __FILE__, __LINE__, "PAPI_query_named_event", retval );
	}

	/* Create an empty event set */
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
	   test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	/* add the events named above */
	retval = PAPI_add_named_event( EventSet, event_names[0] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[0] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}

	retval = PAPI_add_named_event( EventSet, event_names[1] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[1] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}

	values = allocate_test_space( num_tests, num_events );

	/* Gather before stats */
	elapsed_us = PAPI_get_real_usec(  );
	elapsed_cyc = PAPI_get_real_cyc(  );
	elapsed_virt_us = PAPI_get_virt_usec(  );
	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	/* Start PAPI */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our test code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/* Calculate total values */
	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;
	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;
	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	/* remove PAPI_TOT_CYC and PAPI_TOT_INS */
	retval = PAPI_remove_named_event( EventSet, event_names[0] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[0] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}

	retval = PAPI_remove_named_event( EventSet, event_names[1] );
	if ( retval != PAPI_OK ) {
		sprintf( add_event_str, "PAPI_add_named_event[%s]", event_names[1] );
		test_fail( __FILE__, __LINE__, add_event_str, retval );
	}

	if ( !quiet ) {
	   printf( "PAPI_{query, add, remove}_named_event API test.\n" );
	   printf( "-----------------------------------------------\n" );
	   tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
	   printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
	   tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
	   printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
	   printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
	   printf( "-------------------------------------------------------------------------\n" );

	   printf( "Test type    : \t            1\n" );

	   /* cycles is first, other event second */
	   sprintf( add_event_str, "%-12s : \t", event_names[0] );
	   printf( TAB1, add_event_str, values[0][0] );
	   sprintf( add_event_str, "%-12s : \t", event_names[1] );
       printf( TAB1, add_event_str, values[0][1] );

	   printf( TAB1, "Real usec    : \t", elapsed_us );
	   printf( TAB1, "Real cycles  : \t", elapsed_cyc );
	   printf( TAB1, "Virt usec    : \t", elapsed_virt_us );
	   printf( TAB1, "Virt cycles  : \t", elapsed_virt_cyc );

	   printf( "-------------------------------------------------------------------------\n" );

	   printf( "Verification: PAPI_TOT_CYC should be roughly real_cycles\n" );
	   cycles_error=100.0*((double)values[0][0] - (double)elapsed_cyc)/
	     (double)values[0][0];
	   if (cycles_error>10.0) {
	     printf("Error of %.2f%%\n",cycles_error);
	     test_fail( __FILE__, __LINE__, "validation", 0 );
	   }

	}
	test_pass( __FILE__ );

	return 0;
}
