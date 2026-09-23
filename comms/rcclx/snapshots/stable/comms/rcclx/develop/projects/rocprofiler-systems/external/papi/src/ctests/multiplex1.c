/*
* File:    multiplex.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file tests the multiplex functionality, originally developed by
   John May of LLNL. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

/* Event to use in all cases; initialized in init_papi() */

#define TOTAL_EVENTS 6

int solaris_preset_PAPI_events[TOTAL_EVENTS] = {
  PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L2_TCM, PAPI_L1_ICM, 0
};
int power6_preset_PAPI_events[TOTAL_EVENTS] = {
	PAPI_TOT_CYC, PAPI_FP_INS, PAPI_L1_DCM, PAPI_L1_ICM, 0
};
int preset_PAPI_events[TOTAL_EVENTS] = {
  PAPI_TOT_CYC, PAPI_FP_INS, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L1_ICM, 0
};

static int PAPI_events[TOTAL_EVENTS] = { 0, };
static int PAPI_events_len = 0;

static void
init_papi( int *out_events, int *len )
{
	int retval;
	int i, real_len = 0;
	int *in_events = preset_PAPI_events;
	const PAPI_hw_info_t *hw_info;

	/* Initialize the library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail(__FILE__,__LINE__, "PAPI_library_init", retval );
	}

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	if ( strstr( hw_info->model_string, "UltraSPARC" ) ) {
		in_events = solaris_preset_PAPI_events;
	}

	if ( strcmp( hw_info->model_string, "POWER6" ) == 0 ) {
		in_events = power6_preset_PAPI_events;
		retval = PAPI_set_domain( PAPI_DOM_ALL );
		if ( retval != PAPI_OK )
			test_fail(__FILE__,__LINE__, "PAPI_set_domain", retval );
	}

	retval = PAPI_multiplex_init(  );
	if ( retval == PAPI_ENOSUPP) {
		test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
	else if ( retval != PAPI_OK ) {
		test_fail(__FILE__,__LINE__, "PAPI_multiplex_init", retval );
	}

	for ( i = 0; in_events[i] != 0; i++ ) {
		char out[PAPI_MAX_STR_LEN];
		/* query and set up the right instruction to monitor */
		retval = PAPI_query_event( in_events[i] );
		if ( retval == PAPI_OK ) {
			out_events[real_len++] = in_events[i];
			PAPI_event_code_to_name( in_events[i], out );
			if ( real_len == *len )
				break;
		} else {
			PAPI_event_code_to_name( in_events[i], out );
			if ( !TESTS_QUIET )
				printf( "%s does not exist\n", out );
		}
	}

	if ( real_len < 1 ) {
		if (!TESTS_QUIET) printf("Trouble adding events\n");
		test_skip(__FILE__,__LINE__, "No counters available", 0 );
	}
	*len = real_len;
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

int
case1( void )
{
	int retval, i, EventSet = PAPI_NULL;
	long long values[2];

	PAPI_events_len = 2;
	init_papi( PAPI_events, &PAPI_events_len );

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_create_eventset", retval );

	for ( i = 0; i < PAPI_events_len; i++ ) {
		char out[PAPI_MAX_STR_LEN];

		retval = PAPI_add_event( EventSet, PAPI_events[i] );
		if ( retval != PAPI_OK )
			test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
		PAPI_event_code_to_name( PAPI_events[i], out );
		if ( !TESTS_QUIET )
			printf( "Added %s\n", out );
	}

	do_stuff(  );

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		test_print_event_header( "case1:", EventSet );
		printf( TAB2, "case1:", values[0], values[1] );
	}
	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );

	PAPI_shutdown(  );
	return ( SUCCESS );
}

/* Tests that PAPI_set_multiplex() works before adding events */

int
case2( void )
{
	int retval, i, EventSet = PAPI_NULL;
	long long values[2];

	PAPI_events_len = 2;
	init_papi( PAPI_events, &PAPI_events_len );

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_create_eventset", retval );

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_assign_eventset_component", retval );

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
	else if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_set_multiplex", retval );

	for ( i = 0; i < PAPI_events_len; i++ ) {
		char out[PAPI_MAX_STR_LEN];

		retval = PAPI_add_event( EventSet, PAPI_events[i] );
		if ( retval != PAPI_OK )
			test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
		PAPI_event_code_to_name( PAPI_events[i], out );
		if ( !TESTS_QUIET )
			printf( "Added %s\n", out );
	}

	do_stuff(  );

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		test_print_event_header( "case2:", EventSet );
		printf( TAB2, "case2:", values[0], values[1] );
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );

	PAPI_shutdown(  );
	return ( SUCCESS );
}

/* Tests that PAPI_set_multiplex() works after adding events */

int
case3( void  )
{
	int retval, i, EventSet = PAPI_NULL;
	long long values[2];

	PAPI_events_len = 2;
	init_papi( PAPI_events, &PAPI_events_len );

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_create_eventset", retval );

	for ( i = 0; i < PAPI_events_len; i++ ) {
		char out[PAPI_MAX_STR_LEN];

		retval = PAPI_add_event( EventSet, PAPI_events[i] );
		if ( retval != PAPI_OK )
			test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
		PAPI_event_code_to_name( PAPI_events[i], out );
		if ( !TESTS_QUIET )
			printf( "Added %s\n", out );
	}

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	} else if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_set_multiplex", retval );

	do_stuff(  );

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		test_print_event_header( "case3:", EventSet );
		printf( TAB2, "case3:", values[0], values[1] );
	}

	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );

	PAPI_shutdown(  );
	return ( SUCCESS );
}

/* Tests that PAPI_set_multiplex() works before adding events */

/* Tests that PAPI_add_event() works after
   PAPI_add_event()/PAPI_set_multiplex() */

int
case4( void )
{
	int retval, i, EventSet = PAPI_NULL;
	long long values[4];
	char out[PAPI_MAX_STR_LEN];

	PAPI_events_len = 2;
	init_papi( PAPI_events, &PAPI_events_len );

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_create_eventset", retval );

	i = 0;
	retval = PAPI_add_event( EventSet, PAPI_events[i] );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
	PAPI_event_code_to_name( PAPI_events[i], out );
	if (!TESTS_QUIET) printf( "Added %s\n", out );

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
	else if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_set_multiplex", retval );

	i = 1;
	retval = PAPI_add_event( EventSet, PAPI_events[i] );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
	PAPI_event_code_to_name( PAPI_events[i], out );
	if (!TESTS_QUIET) printf( "Added %s\n", out );

	do_stuff(  );

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		test_print_event_header( "case4:", EventSet );
		printf( TAB2, "case4:", values[0], values[1] );
	}

	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );

	PAPI_shutdown(  );
	return ( SUCCESS );
}

/* Tests that PAPI_read() works immediately after
   PAPI_start() */

int
case5( void )
{
  int retval, i, j, EventSet = PAPI_NULL;
	long long start_values[4] = { 0,0,0,0 }, values[4] = {0,0,0,0};
	char out[PAPI_MAX_STR_LEN];

	PAPI_events_len = 2;
	init_papi( PAPI_events, &PAPI_events_len );

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_create_eventset", retval );

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */

	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_assign_eventset_component", retval );

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
	else if ( retval != PAPI_OK ) {
		test_fail(__FILE__,__LINE__, "PAPI_set_multiplex", retval );
	}

	/* Add 2 events... */

	i = 0;
	retval = PAPI_add_event( EventSet, PAPI_events[i] );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
	PAPI_event_code_to_name( PAPI_events[i], out );
	if (!TESTS_QUIET) printf( "Added %s\n", out );
	i++;
	retval = PAPI_add_event( EventSet, PAPI_events[i] );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_add_event", retval );
	PAPI_event_code_to_name( PAPI_events[i], out );
	if (!TESTS_QUIET) printf( "Added %s\n", out );
	i++;

	do_stuff(  );

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_start", retval );

	retval = PAPI_read( EventSet, start_values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_read", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_stop", retval );

	for (j=0;j<i;j++) {
		if (!TESTS_QUIET) {
			printf("read @start counter[%d]: %lld\n",
				j, start_values[j]);
			printf("read @stop  counter[%d]: %lld\n",
				j, values[j]);
			printf("difference  counter[%d]: %lld\n ",
				j, values[j]-start_values[j]);
		}
		if (values[j]-start_values[j] < 0LL) {
			test_fail(__FILE__,__LINE__, "Difference in start and stop resulted in negative value!", 0 );
		}
	  }

	retval = PAPI_cleanup_eventset( EventSet );	/* JT */
	if ( retval != PAPI_OK )
		test_fail(__FILE__,__LINE__, "PAPI_cleanup_eventset", retval );

	PAPI_shutdown(  );
	return ( SUCCESS );
}

int
main( int argc, char **argv )
{

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if (!TESTS_QUIET) {
		printf( "case1: Does PAPI_multiplex_init() not break regular operation?\n" );
	}
	case1(  );

	if (!TESTS_QUIET) {
		printf( "\ncase2: Does setmpx/add work?\n" );
	}
	case2(  );

	if (!TESTS_QUIET) {
		printf( "\ncase3: Does add/setmpx work?\n" );
	}
	case3(  );

	if (!TESTS_QUIET) {
		printf( "\ncase4: Does add/setmpx/add work?\n" );
	}
	case4(  );

	if (!TESTS_QUIET) {
		printf( "\ncase5: Does setmpx/add/add/start/read work?\n" );
	}
	case5(  );

	test_pass( __FILE__ );

	return 0;
}
