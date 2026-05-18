/* This file performs the following test: event_code_to_name */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

static void
test_continue( const char *call, int retval )
{
	if (!TESTS_QUIET) {
		printf( "Expected error in %s: %s\n", call,
			PAPI_strerror(retval) );
	}
}

int
main( int argc, char **argv )
{
	int retval;
	int code = PAPI_TOT_CYC, last;
	char event_name[PAPI_MAX_STR_LEN];
	const PAPI_component_info_t *cmp_info;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );


	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if (!quiet) {
		printf( "Test case code2name.c: "
			"Check limits and indexing of event tables.\n");
		printf( "Looking for PAPI_TOT_CYC...\n" );
	}

	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}

	if (!quiet) printf( "Found |%s|\n", event_name );

	code = PAPI_FP_OPS;

	if (!quiet) {
		printf( "Looking for highest defined preset event "
			"(PAPI_FP_OPS): %#x...\n",code );
	}

	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

	if (!quiet) printf( "Found |%s|\n", event_name );

	code = PAPI_PRESET_MASK | ( PAPI_MAX_PRESET_EVENTS - 1 );

	if (!quiet) {
		printf( "Looking for highest allocated preset event:"
			" %#x...\n", code );
	}

	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
		test_continue( "PAPI_event_code_to_name", retval );
	}
	else {
		if (!quiet) printf( "Found |%s|\n", event_name );
	}

	code = PAPI_PRESET_MASK | ( unsigned int ) PAPI_NATIVE_AND_MASK;

	if (!quiet) {
		printf( "Looking for highest possible preset event:"
			" %#x...\n", code );
	}

	retval = PAPI_event_code_to_name( code, event_name );

	if ( retval != PAPI_OK ) {
		test_continue( "PAPI_event_code_to_name", retval );
	}
	else {
		if (!quiet) printf( "Found |%s|\n", event_name );
	}

	/* Find the first defined native event in component 0 */
	/* For platform independence, always ASK FOR the first event */
	/* Don't just assume it'll be the first numeric value */
	code = PAPI_NATIVE_MASK;
	PAPI_enum_event( &code, PAPI_ENUM_FIRST );

	if (!quiet) {
		printf( "Looking for first native event: %#x...\n", code );
	}

	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Could not find first native event\n");
		test_skip( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}
	else {
	  if (!quiet) printf( "Found |%s|\n", event_name );
	}

	/* Find the last defined native event */

	/* FIXME: hardcoded cmp 0 */
	cmp_info = PAPI_get_component_info( 0 );
	if ( cmp_info == NULL ) {
	   test_fail( __FILE__, __LINE__,
                      "PAPI_get_component_info", PAPI_ECMP );
	}

	code = PAPI_NATIVE_MASK;
	last = code;
	PAPI_enum_event( &code, PAPI_ENUM_FIRST );

	while ( PAPI_enum_event( &code, PAPI_ENUM_EVENTS ) == PAPI_OK ) {
	  last=code;
	}

	code = last;
	if (!quiet) printf( "Looking for last native event: %#x...\n", code );
	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}
	else {
	   if (!quiet) printf( "Found |%s|\n", event_name );
	}

	/* Highly doubtful we have this many natives */
	/* Turn on all bits *except* PRESET bit and COMPONENT bits */
	code = PAPI_PRESET_AND_MASK;
	if (!quiet) printf( "Looking for highest definable native event: %#x...\n", code );
	retval = PAPI_event_code_to_name( code, event_name );
	if ( retval != PAPI_OK ) {
		test_continue( "PAPI_event_code_to_name", retval );
	}
	else {
		if (!quiet) printf( "Found |%s|\n", event_name );
	}

	if ( ( retval == PAPI_ENOCMP) || ( retval == PAPI_ENOEVNT ) || ( retval == PAPI_OK ) ) {
		test_pass( __FILE__ );
	}

	test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", PAPI_EBUG );

	return 1;
}
