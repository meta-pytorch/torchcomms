#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#if defined(_AIX) || defined (__FreeBSD__) || defined (__APPLE__)
#include <sys/wait.h>		 /* ARGH! */
#else
#include <wait.h>
#endif

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{
	int retval, pid, status, EventSet = PAPI_NULL;
	long long int values[] = {0,0};
	PAPI_option_t opt;
	char event_name[BUFSIZ];
	int quiet;

	quiet=tests_quiet( argc, argv );

	if ( ( retval = PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( ( retval = PAPI_assign_eventset_component( EventSet, 0 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component", retval );

	memset( &opt, 0x0, sizeof ( PAPI_option_t ) );
	opt.inherit.inherit = PAPI_INHERIT_ALL;
	opt.inherit.eventset = EventSet;
	if ( ( retval = PAPI_set_opt( PAPI_INHERIT, &opt ) ) != PAPI_OK ) {
		if ( retval == PAPI_ECMP) {
			test_skip( __FILE__, __LINE__, "Inherit not supported by current component.\n", retval );
		} else if (retval == PAPI_EPERM) {
			test_skip( __FILE__, __LINE__, "Inherit not supported by current component.\n", retval );
		} else {
			test_fail( __FILE__, __LINE__, "PAPI_set_opt", retval );
		}
	}

	if ( ( retval = PAPI_query_event( PAPI_TOT_CYC ) ) != PAPI_OK ) {
		if (!quiet) printf("Trouble finding PAPI_TOT_CYC\n");
		test_skip( __FILE__, __LINE__, "PAPI_query_event", retval );
	}

	if ( ( retval = PAPI_add_event( EventSet, PAPI_TOT_CYC ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	strcpy(event_name,"PAPI_FP_INS");
	retval = PAPI_add_named_event( EventSet, event_name );
	if (retval == PAPI_ENOEVNT) {
		strcpy(event_name,"PAPI_TOT_INS");
		retval = PAPI_add_named_event( EventSet, event_name );
	}

	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	pid = fork(  );
	if ( pid == 0 ) {
		do_flops( NUM_FLOPS );
		exit( 0 );
	}
	if ( waitpid( pid, &status, 0 ) == -1 ) {
	  perror( "waitpid()" );
	  exit( 1 );
	}

	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if (!quiet) {
	   printf( "Test case inherit: parent starts, child works, parent stops.\n" );
	   printf( "------------------------------------------------------------\n" );

	   printf( "Test run    : \t1\n" );
	   printf( "%s : \t%lld\n", event_name, values[1] );
	   printf( "PAPI_TOT_CYC: \t%lld\n", values[0] );
	   printf( "------------------------------------------------------------\n" );

	   printf( "Verification:\n" );
	   printf( "Row 1 at least %d\n", NUM_FLOPS );
	   printf( "Row 2 greater than row 1\n");
	}

	if ( values[1] < 100 ) {
		test_fail( __FILE__, __LINE__, event_name, 1 );
	}

	if ( (!strcmp(event_name,"PAPI_FP_INS")) && (values[1] < NUM_FLOPS)) {
		test_fail( __FILE__, __LINE__, "PAPI_FP_INS", 1 );
	}

	test_pass( __FILE__ );

	return 0;

}
