/*  This examples show the essentials in using the PAPI low-level
    interface. The program consists of 3 examples where the work
    done over some work-loops. The example tries to illustrate
    some simple mistakes that are easily made and how a correct
    code would accomplish the same thing.

    Example 1: The total count over two work loops (Loops 1 and 2)
    are supposed to be measured. Due to a mis-understanding of the
    semantics of the API the total count gets wrong.
    The example also illustrates that it is legal to read both
    running and stopped counters.

    Example 2: The total count over two work loops (Loops 1 and 3)
    is supposed to be measured while discarding the counts made in
    loop 2. Instead the counts in loop1 are counted twice and the
    counts in loop2 are added to the total number of counts.

    Example 3: One correct way of accomplishing the result aimed for
    in example 2.
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define NUM_EVENTS 2

int
main( int argc, char **argv )
{
	int retval;
	long long values[NUM_EVENTS], dummyvalues[NUM_EVENTS];
	int Events[NUM_EVENTS];
	int EventSet = PAPI_NULL;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* query and set up the right events to monitor */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		Events[0] = PAPI_FP_INS;
		Events[1] = PAPI_TOT_CYC;
	} else {
		Events[0] = PAPI_TOT_INS;
		Events[1] = PAPI_TOT_CYC;
	}

	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval = PAPI_add_events( EventSet, ( int * ) Events, NUM_EVENTS );
	if (retval < PAPI_OK ) {
		if (!quiet) printf("Trouble adding events\n");
		test_skip( __FILE__, __LINE__, "PAPI_add_events", retval );
	}

	if ( !quiet ) {
		printf( "\n   Incorrect usage of read and accum.\n" );
		printf( "   Some cycles are counted twice\n" );
	}
	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	/* Loop 1 */
	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_read( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters continuing...)\n" );

	/* Loop 2 */
	do_flops( NUM_FLOPS );

	/* Using PAPI_accum here is incorrect. The result is that Loop 1 *
	 * is being counted twice                                        */
	if ( ( retval = PAPI_accum( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_accum", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters being accumulated)\n" );

	/* Loop 3 */
	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_stop( EventSet, dummyvalues ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( ( retval = PAPI_read( EventSet, dummyvalues ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );

	if ( !quiet ) {
		printf( TWO12, dummyvalues[0], dummyvalues[1],
				"(Reading stopped counters)\n" );

		printf( TWO12, values[0], values[1], "" );

		printf( "\n   Incorrect usage of read and accum.\n" );
		printf( "   Another incorrect use\n" );
	}
	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	/* Loop 1 */
	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_read( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters continuing...)\n" );

	/* Loop 2 */
	/* Code that should not be counted */
	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_read( EventSet, dummyvalues ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );

	if ( !quiet )
		printf( TWO12, dummyvalues[0], dummyvalues[1],
				"(Intermediate counts...)\n" );

	/* Loop 3 */
	do_flops( NUM_FLOPS );

	/* Since PAPI_read does not reset the counters it's use above after    *
	 * loop 2 is incorrect. Instead Loop1 will in effect be counted twice. *
	 * and the counts in loop 2 are included in the total counts           */
	if ( ( retval = PAPI_accum( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_accum", retval );
	if ( !quiet )
		printf( TWO12, values[0], values[1], "" );

	if ( ( retval = PAPI_stop( EventSet, dummyvalues ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( !quiet ) {
		printf( "\n   Correct usage of read and accum.\n" );
		printf( "   PAPI_reset and PAPI_accum used to skip counting\n" );
		printf( "   a section of the code.\n" );
	}
	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_read( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read", retval );
	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters continuing)\n" );

	/* Code that should not be counted */
	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_reset( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_reset", retval );

	if ( !quiet )
		printf( "%12s %12s  (Counters reset)\n", "", "" );

	do_flops( NUM_FLOPS );

	if ( ( retval = PAPI_accum( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_accum", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "" );

	if ( !quiet ) {
		printf( "----------------------------------\n" );
		printf( "Verification: The last line in each experiment should be\n" );
		printf( "approximately twice the value of the first line.\n" );
		printf
			( "The third case illustrates one possible way to accomplish this.\n" );
	}
	test_pass( __FILE__ );

	return 0;
}
