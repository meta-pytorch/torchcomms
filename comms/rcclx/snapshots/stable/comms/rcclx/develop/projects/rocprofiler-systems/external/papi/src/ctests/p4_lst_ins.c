/* This code demonstrates the behavior of PAPI_LD_INS, PAPI_SR_INS and PAPI_LST_INS
	on a Pentium 4 processor. Because of the way these events are implemented in
	hardware, LD and SR cannot be counted in the presence of either of the other 
	two events.
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval, num_tests = 6, tmp;
	long long **values;
	int EventSet = PAPI_NULL;
	const PAPI_hw_info_t *hw_info;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	if ( hw_info->vendor == PAPI_VENDOR_INTEL ) {
	    /* Check for Pentium4 */
	    if ( hw_info->cpuid_family != 15 ) {
		test_skip( __FILE__, __LINE__,
			"This test is intended only for Pentium 4.", 1 );
	    }
	} else {
	    test_skip( __FILE__, __LINE__, 
		    "This test is intended only for Pentium 4.", 1 );
	}

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	values = allocate_test_space( num_tests, 2 );

/* First test: just PAPI_LD_INS */
	retval = PAPI_add_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_LD_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_remove_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_LD_INS",
				   retval );

/* Second test: just PAPI_SR_INS */
	retval = PAPI_add_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_SR_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[1] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_remove_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_SR_INS",
				   retval );

/* Third test: just PAPI_LST_INS */
	retval = PAPI_add_event( EventSet, PAPI_LST_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_LST_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[2] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

/* Fourth test: PAPI_LST_INS and PAPI_LD_INS */
	retval = PAPI_add_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_LD_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[3] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_remove_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_LD_INS",
				   retval );

/* Fifth test: PAPI_LST_INS and PAPI_SR_INS */
	retval = PAPI_add_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_SR_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[4] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_remove_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_SR_INS",
				   retval );

	retval = PAPI_remove_event( EventSet, PAPI_LST_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_LST_INS",
				   retval );

/* Sixth test: PAPI_LD_INS and PAPI_SR_INS */
	retval = PAPI_add_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_LD_INS", retval );

	retval = PAPI_add_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event: PAPI_SR_INS", retval );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS / 10 );

	retval = PAPI_stop( EventSet, values[5] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_remove_event( EventSet, PAPI_LD_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_LD_INS",
				   retval );

	retval = PAPI_remove_event( EventSet, PAPI_SR_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event: PAPI_SR_INS",
				   retval );



	if ( !TESTS_QUIET ) {
		printf( "Pentium 4 Load / Store tests.\n" );
		printf
			( "These PAPI events are counted by setting a tag at the front of the pipeline,\n" );
		printf
			( "and counting tags at the back of the pipeline. All the tags are the same 'color'\n" );
		printf
			( "and can't be distinguished from each other. Therefore, PAPI_LD_INS and PAPI_SR_INS\n" );
		printf
			( "cannot be counted with the other two events, or the answer will always == PAPI_LST_INS.\n" );
		printf
			( "-------------------------------------------------------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS / 10 );
		printf
			( "-------------------------------------------------------------------------------------------\n" );

		printf
			( "Test:                1            2            3            4            5            6\n" );
		printf( "%s %12lld %12s %12s %12lld %12s %12lld\n", "PAPI_LD_INS: ",
				( values[0] )[0], "------", "------", ( values[3] )[1],
				"------", ( values[5] )[0] );
		printf( "%s %12s %12lld %12s %12s %12lld %12lld\n", "PAPI_SR_INS: ",
				"------", ( values[1] )[0], "------", "------",
				( values[4] )[1], ( values[5] )[1] );
		printf( "%s %12s %12s %12lld %12lld %12lld %12s\n", "PAPI_LST_INS:",
				"------", "------", ( values[2] )[0], ( values[3] )[0],
				( values[4] )[0], "------" );
		printf
			( "-------------------------------------------------------------------------------------------\n" );

		printf( "Test 1: PAPI_LD_INS only.\n" );
		printf( "Test 2: PAPI_SR_INS only.\n" );
		printf( "Test 3: PAPI_LST_INS only.\n" );
		printf( "Test 4: PAPI_LD_INS and PAPI_LST_INS.\n" );
		printf( "Test 5: PAPI_SR_INS and PAPI_LST_INS.\n" );
		printf( "Test 6: PAPI_LD_INS and PAPI_SR_INS.\n" );
		printf
			( "Verification: Values within each column should be the same.\n" );
		printf( "              R3C3 ~= (R1C1 + R2C2) ~= all other entries.\n" );
	}

	test_pass( __FILE__ );

	return 0;

}
