/* From Paul Drongowski at HP. Thanks. */

/*  I have not been able to call PAPI_describe_event without
    incurring a segv, including the sample code on the man page.
    I noticed that PAPI_describe_event is not exercised by the
    PAPI test programs, so I haven't been able to check the
    function call using known good code. (Or steal your code
    for that matter. :-)
*/

/*  PAPI_describe_event has been deprecated in PAPI 3, since
    its functionality exists in other API calls. Below shows
    several ways that this call was used, with replacement
    code compatible with PAPI 3.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	int retval;
	long long g1[2];
	int eventcode = PAPI_TOT_INS;
	PAPI_event_info_t info, info1, info2;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );


	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	if ( ( retval = PAPI_query_event( eventcode ) ) != PAPI_OK ) {
		if (!quiet) printf("Trouble checking event\n");
		test_skip( __FILE__, __LINE__, "PAPI_query_event(PAPI_TOT_INS)",
				   retval );
	}

	if ( ( retval = PAPI_add_event( EventSet, eventcode ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event(PAPI_TOT_INS)", retval );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	if ( ( retval = PAPI_stop( EventSet, g1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	/* Case 0, no info, should fail */
	eventcode = 0;
/*
   if ( ( retval = PAPI_describe_event(eventname,(int *)&eventcode,eventdesc) ) == PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);
*/
	if (!quiet) {
	    printf("This test expects a 'PAPI Error' to be returned from this PAPI call.\n");
	}
	if ( ( retval = PAPI_get_event_info( eventcode, &info ) ) == PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_get_event_info", retval );

	/* Case 1, fill in name field. */
	eventcode = PAPI_TOT_INS;
/*
   if ( ( retval = PAPI_describe_event(eventname,(int *)&eventcode,eventdesc) ) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);
*/
	if ( ( retval = PAPI_get_event_info( eventcode, &info1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_get_event_info", retval );

	if ( strcmp( info1.symbol, "PAPI_TOT_INS" ) != 0 )
		test_fail( __FILE__, __LINE__,
				   "PAPI_get_event_info symbol value is bogus", retval );
	if ( strlen( info1.long_descr ) == 0 )
		test_fail( __FILE__, __LINE__,
				   "PAPI_get_event_info long_descr value is bogus", retval );

	eventcode = 0;

	/* Case 2, fill in code field. */
/*
   if ( ( retval = PAPI_describe_event(eventname,(int *)&eventcode,eventdesc) ) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);
*/
	if ( ( retval = PAPI_event_name_to_code( info1.symbol, ( int * ) &eventcode ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
	}

	if ( eventcode != PAPI_TOT_INS )
		test_fail( __FILE__, __LINE__,
				   "PAPI_event_name_to_code code value is bogus", retval );

	if ( ( retval = PAPI_get_event_info( eventcode, &info2 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_get_event_info", retval );

	if ( strcmp( info2.symbol, "PAPI_TOT_INS" ) != 0 )
		test_fail( __FILE__, __LINE__,
				   "PAPI_get_event_info symbol value is bogus", retval );
	if ( strlen( info2.long_descr ) == 0 )
		test_fail( __FILE__, __LINE__,
				   "PAPI_get_event_info long_descr value is bogus", retval );

	test_pass( __FILE__ );

	return 0;
}
