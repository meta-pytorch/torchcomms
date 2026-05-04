/*
 *  Test PAPI with fork() and exec().
 */

#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"
#include "testcode.h"

#define MAX_EVENTS  3

static int Event[MAX_EVENTS] = {
	PAPI_TOT_CYC,
	PAPI_FP_INS,
	PAPI_FAD_INS,
};

static int Threshold[MAX_EVENTS] = {
	8000000,
	4000000,
	4000000,
};

static struct timeval start, last;
static long count, total;

static void
my_handler( int EventSet, void *pc, long long ovec, void *context )
{
	( void ) EventSet;
	( void ) pc;
	( void ) ovec;
	( void ) context;

	count++;
	total++;
}





static void
print_rate( const char *str )
{
	static int last_count = -1;
	struct timeval now;
	double st_secs, last_secs;

	gettimeofday( &now, NULL );
	st_secs = ( double ) ( now.tv_sec - start.tv_sec )
		+ ( ( double ) ( now.tv_usec - start.tv_usec ) ) / 1000000.0;
	last_secs = ( double ) ( now.tv_sec - last.tv_sec )
		+ ( ( double ) ( now.tv_usec - last.tv_usec ) ) / 1000000.0;
	if ( last_secs <= 0.001 )
		last_secs = 0.001;

	if (!TESTS_QUIET) {
		printf( "[%d] %s, time = %.3f, total = %ld, last = %ld, rate = %.1f/sec\n",
			getpid(  ), str, st_secs, total, count,
			( ( double ) count ) / last_secs );
	}

	if ( last_count != -1 ) {
		if ( count < .1 * last_count ) {
			test_fail( __FILE__, __LINE__, "Interrupt rate changed!", 1 );
			exit( 1 );
		}
	}
	last_count = ( int ) count;
	count = 0;
	last = now;
}

static void
run( const char *str, int len )
{
	int n;

	for ( n = 1; n <= len; n++ ) {
		do_cycles( 1 );
		print_rate( str );
	}
}

int
main( int argc, char **argv )
{
	int num_events = 1;
	const char *name = "unknown";
	int ev,EventSet = PAPI_NULL;
	int quiet,retval;

	/* Used to be able to set this via command line */
	num_events=1;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	do_cycles( 1 );

	/* Zero out the Counters */
	gettimeofday( &start, NULL );
	last = start;
	count = 0;
	total = 0;

	/* Initialize PAPI */
	retval=PAPI_library_init( PAPI_VER_CURRENT );
	if (retval!=PAPI_VER_CURRENT) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init failed", 1 );
	}

	name = argv[0];
	if (!quiet) {
		printf( "[%d] %s, num_events = %d\n", getpid(),
			name, num_events );
	}

	/* Create eventset */
	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset failed", 1 );
	}

	/* Add events */
	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_add_event( EventSet, Event[ev] ) != PAPI_OK ) {
			if (!quiet) printf("Trouble adding event\n");
			test_skip( __FILE__, __LINE__, "PAPI_add_event failed", 1 );
		}
	}

	/* Set overflow */
	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_overflow( EventSet, Event[ev],
					Threshold[ev], 0, my_handler )
			 != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_overflow failed", 1 );
		}
	}

	/* Start measuring */
	if ( PAPI_start( EventSet ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start failed", 1 );
	}



	/* Tun a bit */
	run( name, 3 );

	/* Stop measuring */
	if (!quiet) {
		printf("[%d] %s, %s\n", getpid(), name, "stop");
	}

	if ( PAPI_stop( EventSet, NULL ) != PAPI_OK ) {
		test_fail( name, __LINE__, "PAPI_stop failed", 1 );
	}

	if (!quiet) {
		printf("[%d] %s, %s\n", getpid(),
			name, "exec(./child_overflow)");
	}

	/* exec the child_overflow helper program */
	/* we should never return from this */
	if ( access( "./child_overflow", X_OK ) == 0 )
		execl( "./child_overflow", "./child_overflow",
			   ( quiet ? "TESTS_QUIET" : NULL ), NULL );
	else if ( access( "./ctests/child_overflow", X_OK ) == 0 )
		execl( "./ctests/child_overflow", "./ctests/child_overflow",
			   ( quiet ? "TESTS_QUIET" : NULL ), NULL );

	test_fail( name, __LINE__, "exec failed", 1 );

	return 0;
}
