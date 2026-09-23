/* This file performs the following test: start, stop and timer functionality for derived events

   NOTE: This test becomes useless when rate events like PAPI_FLOPS are removed.

   - It tests the derived metric FLOPS using the following two counters.
     They are counted in the default counting domain and default
     granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include "papi_test.h"

extern int TESTS_QUIET;				   /* Declared in test_utils.c */

int
main( int argc, char **argv )
{
	int retval, num_tests = 2, tmp;
	int EventSet1 = PAPI_NULL;
	int EventSet2 = PAPI_NULL;
	int mask1 = 0x80001;			   /* FP_OPS and TOT_CYC */
	int mask2 = 0x8;				   /* FLOPS */
	int num_events1;
	int num_events2;
	long long **values;
	int clockrate;
	double test_flops;


	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	/* gotta count flops to run this test */
	if ( ( retval = PAPI_query_event( PAPI_FP_OPS ) ) != PAPI_OK )
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );

	EventSet1 = add_test_events( &num_events1, &mask1 );
/*   EventSet2 = add_test_events(&num_events2, &mask2); */

	if ( num_events1 == 0 || num_events2 == 0 )
		test_skip( __FILE__, __LINE__, "add_test_events", PAPI_ENOEVNT );

	/* num_events1 is greater than num_events2 so don't worry. */

	values = allocate_test_space( num_tests, num_events1 );

	clockrate = PAPI_get_opt( PAPI_CLOCKRATE, NULL );
	if ( clockrate < 1 )
		test_fail( __FILE__, __LINE__, "PAPI_get_opt", retval );

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
/*
   retval = PAPI_start(EventSet2);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet2, values[1]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
*/
	remove_test_events( &EventSet1, mask1 );
/*   remove_test_events(&EventSet2, mask2); */

	test_flops =
		( double ) ( values[0] )[0] *
		( double ) clockrate *( double ) 1000000.0;
	test_flops = test_flops / ( double ) ( values[0] )[1];

	if ( !TESTS_QUIET ) {
		printf( "Test case 9: start, stop for derived event PAPI_FLOPS.\n" );
		printf( "------------------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf
			( "-------------------------------------------------------------------------\n" );

		printf( "Test type   : %12s%12s\n", "1", "2" );
		printf( TAB2, "PAPI_FP_OPS : ", ( values[0] )[0], ( long long ) 0 );
		printf( TAB2, "PAPI_TOT_CYC: ", ( values[0] )[1], ( long long ) 0 );
		printf( TAB2, "PAPI_FLOPS  : ", ( long long ) 0, ( values[1] )[0] );
		printf
			( "-------------------------------------------------------------------------\n" );

		printf( "Verification:\n" );
		printf( "Last number in row 3 approximately equals %f\n", test_flops );
		printf( "This test is no longer valid: PAPI_FLOPS is deprecated.\n" );
	}
/*   {
      double min, max;
      min = values[1][0] * .9;
      max = values[1][0] * 1.1;
      if (test_flops > max || test_flops < min)
         test_fail(__FILE__, __LINE__, "PAPI_FLOPS", 1);
   }
*/
	test_pass( __FILE__, values, num_tests );
	exit( 1 );
}
