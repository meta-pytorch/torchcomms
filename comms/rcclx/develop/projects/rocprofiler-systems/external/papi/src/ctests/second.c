/* This file performs the following test: counter domain testing

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations.
     + PAPI_TOT_INS
     + PAPI_TOT_CYC
   - Start system domain counters
   - Do flops
   - Stop and read system domain counters
   - Start kernel domain counters
   - Do flops
   - Stop and read kernel domain counters
   - Start user domain counters
   - Do flops
   - Stop and read user domain counters
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define TAB_DOM	"%s%12lld%15lld%17lld\n"

#define CASE2 0
#define CREATE 1
#define ADD 2
#define MIDDLE 3
#define CHANGE 4
#define SUPERVISOR 5

void
dump_and_verify( int test_case, long long **values )
{
	long long min, max, min2, max2;

	if (!TESTS_QUIET) {
		printf( "-----------------------------------------------------------------\n" );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-------------------------------------------------------------\n" );
	}
	if ( test_case == CASE2 ) {
		if (!TESTS_QUIET) {
			printf( "Test type   :   Before Create   Before Add       Between Adds\n" );
			printf( TAB_DOM, "PAPI_TOT_INS: ", ( values[0] )[0], ( values[1] )[0],
				( values[2] )[0] );
			printf( TAB_DOM, "PAPI_TOT_CYC: ", ( values[0] )[1], ( values[1] )[1],
				( values[2] )[1] );
			printf( "-------------------------------------------------------------\n" );
			printf( "Verification:\n" );
			printf( "Both rows equal 'n  N  N' where n << N\n" );
			return;
		}
	} else if ( test_case == CHANGE ) {
		min = ( long long ) ( ( double ) values[0][0] * ( 1 - TOLERANCE ) );
		max = ( long long ) ( ( double ) values[0][0] * ( 1 + TOLERANCE ) );
		if ( values[1][0] > max || values[1][0] < min )
		test_fail( __FILE__, __LINE__, "PAPI_TOT_INS", 1 );

		min = ( long long ) ( ( double ) values[1][1] * ( 1 - TOLERANCE ) );
		max = ( long long ) ( ( double ) values[1][1] * ( 1 + TOLERANCE ) );
		if ( ( values[2][1] + values[0][1] ) > max ||
			 ( values[2][1] + values[0][1] ) < min )
			test_fail( __FILE__, __LINE__, "PAPI_TOT_CYC", 1 );

		if (!TESTS_QUIET) {
			printf( "Test type   :   PAPI_DOM_ALL  PAPI_DOM_KERNEL  PAPI_DOM_USER\n" );
			printf( TAB_DOM, "PAPI_TOT_INS: ", ( values[1] )[0], ( values[2] )[0],
				( values[0] )[0] );
			printf( TAB_DOM, "PAPI_TOT_CYC: ", ( values[1] )[1], ( values[2] )[1],
				( values[0] )[1] );
			printf( "-------------------------------------------------------------\n" );

			printf( "Verification:\n" );
			printf( "Both rows approximately equal '(N+n)  n  N', where n << N\n" );
			printf( "Column 1 approximately equals column 2 plus column 3\n" );
		}
	} else if ( test_case == SUPERVISOR ) {
		if (!TESTS_QUIET) {
			printf( "Test type   :   PAPI_DOM_ALL  All-minus-supervisor  Supervisor-only\n" );
			printf( TAB_DOM, "PAPI_TOT_INS: ", ( values[0] )[0], ( values[1] )[0],
				( values[2] )[0] );
			printf( TAB_DOM, "PAPI_TOT_CYC: ", ( values[0] )[1], ( values[1] )[1],
				( values[2] )[1] );
			printf( "-------------------------------------------------------------\n" );
			printf( "Verification:\n" );
			printf( "Both rows approximately equal '(N+n)  n  N', where n << N\n" );
			printf( "Column 1 approximately equals column 2 plus column 3\n" );
		}
	} else {
		min = ( long long ) ( ( double ) values[2][0] * ( 1 - TOLERANCE ) );
		max = ( long long ) ( ( double ) values[2][0] * ( 1 + TOLERANCE ) );

		min2 = ( long long ) ( ( double ) values[0][1] * ( 1 - TOLERANCE ) );
		max2 =
			( long long ) ( ( double ) ( double ) values[0][1] *
							( 1 + TOLERANCE ) );

		if (!TESTS_QUIET) {
			printf( "Test type   :   PAPI_DOM_ALL  PAPI_DOM_KERNEL  PAPI_DOM_USER\n" );
			printf( TAB_DOM, "PAPI_TOT_INS: ", ( values[0] )[0], ( values[1] )[0],
				( values[2] )[0] );
			printf( TAB_DOM, "PAPI_TOT_CYC: ", ( values[0] )[1], ( values[1] )[1],
				( values[2] )[1] );
			printf( "-------------------------------------------------------------\n" );
			printf( "Verification:\n" );
			printf( "Both rows approximately equal '(N+n)  n  N', where n << N\n" );
			printf( "Column 1 approximately equals column 2 plus column 3\n" );
		}
		if ( values[0][0] > max || values[0][0] < min )
			test_fail( __FILE__, __LINE__, "PAPI_TOT_INS", 1 );

		if ( ( values[1][1] + values[2][1] ) > max2 ||
			 ( values[1][1] + values[2][1] ) < min2 )
			test_fail( __FILE__, __LINE__, "PAPI_TOT_CYC", 1 );
	}

	if ( values[0][0] == 0 || values[0][1] == 0 ||
		 values[1][0] == 0 || values[1][1] == 0 )
		test_fail( __FILE__, __LINE__,
				   "Verify non-zero count for all domain types", 1 );

	if ( values[2][0] == 0 || values[2][1] == 0 ) {
		if ( test_case == SUPERVISOR ) {
			if (!TESTS_QUIET) printf( "WARNING: No events counted in supervisor context.  This is expected in a non-virtualized environment.\n" );
		} else {
			test_fail( __FILE__, __LINE__,
					   "Verify non-zero count for all domain types", 1 );
		}
	}
}

/* Do the set_domain on the eventset before adding events */

void
case1( int num )
{
	int retval, num_tests = 3;
	long long **values;
	int EventSet1 = PAPI_NULL, EventSet2 = PAPI_NULL, EventSet3 = PAPI_NULL;
	PAPI_option_t options;
	const PAPI_component_info_t *cmpinfo;

	memset( &options, 0x0, sizeof ( options ) );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	/* get info from cpu component */
	cmpinfo = PAPI_get_component_info( 0 );	
	if ( cmpinfo == NULL ) {
	   test_fail( __FILE__, __LINE__,"PAPI_get_component_info", PAPI_ECMP);
	}

	if ( ( retval = PAPI_query_event( PAPI_TOT_INS ) ) != PAPI_OK )
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );

	if ( ( retval = PAPI_query_event( PAPI_TOT_CYC ) ) != PAPI_OK )
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );

	retval = PAPI_create_eventset( &EventSet1 );
	if ( retval == PAPI_OK )
		retval = PAPI_create_eventset( &EventSet2 );
	if ( retval == PAPI_OK )
		retval = PAPI_create_eventset( &EventSet3 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals. 0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet1, 0 );
	if ( retval == PAPI_OK )
		retval = PAPI_assign_eventset_component( EventSet2, 0 );
	if ( retval == PAPI_OK )
		retval = PAPI_assign_eventset_component( EventSet3, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

	if ( num == CREATE ) {
		if (!TESTS_QUIET) printf( "\nTest case CREATE: Call PAPI_set_opt(PAPI_DOMAIN) on EventSet before add\n" );
		options.domain.eventset = EventSet1;
		options.domain.domain = PAPI_DOM_ALL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

		options.domain.eventset = EventSet2;
		options.domain.domain = PAPI_DOM_KERNEL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

		options.domain.eventset = EventSet3;
		options.domain.domain = PAPI_DOM_USER;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
	}

	retval = PAPI_add_event( EventSet1, PAPI_TOT_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_INS)", retval );

	retval = PAPI_add_event( EventSet1, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_CYC)", retval );

	retval = PAPI_add_event( EventSet2, PAPI_TOT_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_INS)", retval );

	retval = PAPI_add_event( EventSet2, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_CYC)", retval );

	retval = PAPI_add_event( EventSet3, PAPI_TOT_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_INS)", retval );

	if ( num == MIDDLE ) {
		if (!TESTS_QUIET) printf( "\nTest case MIDDLE: Call PAPI_set_opt(PAPI_DOMAIN) on EventSet between adds\n" );
		options.domain.eventset = EventSet1;
		options.domain.domain = PAPI_DOM_ALL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK && retval != PAPI_ECMP ) {
		   test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
		}

		options.domain.eventset = EventSet2;
		options.domain.domain = PAPI_DOM_KERNEL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

		options.domain.eventset = EventSet3;
		options.domain.domain = PAPI_DOM_USER;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
	}

	retval = PAPI_add_event( EventSet3, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_CYC)", retval );

	if ( num == ADD ) {
		if (!TESTS_QUIET) printf( "\nTest case ADD: Call PAPI_set_opt(PAPI_DOMAIN) on EventSet after add\n" );
		options.domain.eventset = EventSet1;
		options.domain.domain = PAPI_DOM_ALL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK && retval != PAPI_ECMP ) {
		   test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
		}
		options.domain.eventset = EventSet2;
		options.domain.domain = PAPI_DOM_KERNEL;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );

		options.domain.eventset = EventSet3;
		options.domain.domain = PAPI_DOM_USER;

		retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
	}

	/* 2 events */

	values = allocate_test_space( num_tests, 2 );

	if ( num == CHANGE ) {
		/* This testcase is dependent on the CREATE testcase running immediately before it, using
		 * domain settings of "All", "Kernel" and "User", on event sets 1, 2, and 3, respectively.
		 */
		PAPI_option_t option;
		if (!TESTS_QUIET) printf( "\nTest case CHANGE 1: Change domain on EventSet between runs, using generic domain options:\n" );
		PAPI_start( EventSet1 );
		PAPI_stop( EventSet1, values[0] );

		// change EventSet1 domain from All  to User
		option.domain.domain = PAPI_DOM_USER;
		option.domain.eventset = EventSet1;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );

		PAPI_start( EventSet2 );
		PAPI_stop( EventSet2, values[1] );

		// change EventSet2 domain from Kernel to All
		option.domain.domain = PAPI_DOM_ALL;
		option.domain.eventset = EventSet2;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );

		PAPI_start( EventSet3 );
		PAPI_stop( EventSet3, values[2] );

		// change EventSet3 domain from User  to Kernel
		option.domain.domain = PAPI_DOM_KERNEL;
		option.domain.eventset = EventSet3;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );

		free_test_space( values, num_tests );
		values = allocate_test_space( num_tests, 2 );

	}

	if ( num == SUPERVISOR &&
		 ( cmpinfo->available_domains & PAPI_DOM_SUPERVISOR ) ) {
		PAPI_option_t option;

		if (!TESTS_QUIET) printf( "\nTest case CHANGE 2: Change domain on EventSets to include/exclude supervisor events:\n" );

		option.domain.domain = PAPI_DOM_ALL;
		option.domain.eventset = EventSet1;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain ALL ", retval );

		option.domain.domain = PAPI_DOM_ALL ^ PAPI_DOM_SUPERVISOR;
		option.domain.eventset = EventSet2;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK ) {

                   /* DOM_ALL is special-cased as domains_available   */
                   /* in papi.c .  Some machines don't like DOM_OTHER */
                   /* so try that if the above case fails.            */
		   option.domain.domain ^= PAPI_DOM_OTHER;
		   option.domain.eventset = EventSet2;
		   retval = PAPI_set_opt( PAPI_DOMAIN, &option );

                   if (retval != PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_set_domain ALL^SUPERVISOR ", retval );
                   }
                }

		option.domain.domain = PAPI_DOM_SUPERVISOR;
		option.domain.eventset = EventSet3;
		retval = PAPI_set_opt( PAPI_DOMAIN, &option );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain SUPERVISOR ", retval );

		free_test_space( values, num_tests );
		values = allocate_test_space( num_tests, 2 );
	}
	/* Warm it up dude */

	PAPI_start( EventSet1 );
	do_flops( NUM_FLOPS );
	PAPI_stop( EventSet1, NULL );

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_start( EventSet2 );

	do_flops( NUM_FLOPS );

	if ( retval == PAPI_OK ) {
		retval = PAPI_stop( EventSet2, values[1] );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	} else {
		values[1][0] = retval;
		values[1][1] = retval;
	}

	retval = PAPI_start( EventSet3 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet3, values[2] );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_cleanup_eventset( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup", retval );

	retval = PAPI_destroy_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy", retval );

	retval = PAPI_cleanup_eventset( EventSet2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup", retval );

	retval = PAPI_destroy_eventset( &EventSet2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy", retval );

	retval = PAPI_cleanup_eventset( EventSet3 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup", retval );

	retval = PAPI_destroy_eventset( &EventSet3 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy", retval );

	dump_and_verify( num, values );

	free(values);

	PAPI_shutdown(  );
}

void
case2( int num, int domain, long long *values )
{
	int retval;
	int EventSet1 = PAPI_NULL;
	PAPI_option_t options;

	memset( &options, 0x0, sizeof ( options ) );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_query_event( PAPI_TOT_INS ) ) != PAPI_OK )
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );

	if ( ( retval = PAPI_query_event( PAPI_TOT_CYC ) ) != PAPI_OK )
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );

	if ( num == CREATE ) {
		if (!TESTS_QUIET) {
			printf( "\nTest case 2, CREATE: Call PAPI_set_domain(%s) before create\n",
				stringify_domain( domain ) );
			printf( "This should override the domain setting for this EventSet.\n" );
		}
		retval = PAPI_set_domain( domain );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
	}

	retval = PAPI_create_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( num == ADD ) {
		if (!TESTS_QUIET) {
			printf( "\nTest case 2, ADD: Call PAPI_set_domain(%s) before add\n",
					stringify_domain( domain ) );
			printf( "This should have no effect on the domain setting for this EventSet.\n" );
		}

		retval = PAPI_set_domain( domain );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
	}

	retval = PAPI_add_event( EventSet1, PAPI_TOT_INS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_INS)", retval );

	if ( num == MIDDLE ) {
		if (!TESTS_QUIET) {
			printf( "\nTest case 2, MIDDLE: Call PAPI_set_domain(%s) between adds\n",
				stringify_domain( domain ) );
			printf( "This should have no effect on the domain setting for this EventSet.\n" );
		}

		retval = PAPI_set_domain( domain );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
	}

	retval = PAPI_add_event( EventSet1, PAPI_TOT_CYC );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_CYC)", retval );


	/* Warm it up dude */

	PAPI_start( EventSet1 );
	do_flops( NUM_FLOPS );
	PAPI_stop( EventSet1, NULL );

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	retval = PAPI_stop( EventSet1, values );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_cleanup_eventset( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup", retval );

	retval = PAPI_destroy_eventset( &EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy", retval );

	PAPI_shutdown(  );
}

void
case2_driver( void )
{
	long long **values;

	/* 3 tests, 2 events */

	values = allocate_test_space( 3, 2 );

	case2( CREATE, PAPI_DOM_KERNEL, values[0] );
	case2( ADD, PAPI_DOM_KERNEL, values[1] );
	case2( MIDDLE, PAPI_DOM_KERNEL, values[2] );

	dump_and_verify( CASE2, values );

	free(values);
}

void
case1_driver( void )
{
	case1( ADD );
	case1( MIDDLE );
	case1( CREATE );
	case1( CHANGE );
	case1( SUPERVISOR );
}

int
main( int argc, char **argv )
{
	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

#if defined(sgi) && defined(host_mips)
	uid_t id;
	id = getuid(  );
	if ( id != 0 ) {
		printf( "IRIX requires root for PAPI_DOM_KERNEL and PAPI_DOM_ALL.\n" );
		test_skip( __FILE__, __LINE__, "", 1 );
	}
#endif

	if (!TESTS_QUIET) {
		printf( "Test second.c: set domain of eventset via PAPI_set_domain and PAPI_set_opt.\n\n" );
		printf( "* PAPI_set_domain(DOMAIN) sets the default domain \napplied to subsequently created EventSets.\n" );
		printf( "It should have no effect on existing EventSets.\n\n" );
		printf( "* PAPI_set_opt(DOMAIN,xxx) sets the domain for a specific EventSet.\n" );
		printf( "It should always override the default setting for that EventSet.\n" );
	}

	case2_driver(  );
	case1_driver(  );

	test_pass( __FILE__ );

	return 0;
}
