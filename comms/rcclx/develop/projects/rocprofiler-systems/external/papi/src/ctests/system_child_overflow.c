/*
 *  Use "system() to run child_overflow
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

static int num_events = 1;
static int EventSet = PAPI_NULL;
static const char *name = "unknown";
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
zero_count( void )
{
	gettimeofday( &start, NULL );
	last = start;
	count = 0;
	total = 0;
}

static void
print_here( const char *str) {

	if (!TESTS_QUIET) printf("[%d] %s, %s\n", getpid(), name, str);
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
			test_fail( name, __LINE__, "Interrupt rate changed!", 1 );
			exit( 1 );
		}
	}
	last_count = ( int ) count;
	count = 0;
	last = now;
}

static void
do_cycles( int program_time )
{
	struct timeval start, now;
	double x, sum;

	gettimeofday( &start, NULL );

	for ( ;; ) {
		sum = 1.0;
		for ( x = 1.0; x < 250000.0; x += 1.0 )
			sum += x;
		if ( sum < 0.0 )
			printf( "==>>  SUM IS NEGATIVE !!  <<==\n" );

		gettimeofday( &now, NULL );
		if ( now.tv_sec >= start.tv_sec + program_time )
			break;
	}
}

static void
my_papi_start( void )
{
	int ev;

	EventSet = PAPI_NULL;

	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
		test_fail( name, __LINE__, "PAPI_create_eventset failed", 1 );

	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_add_event( EventSet, Event[ev] ) != PAPI_OK ) {
			if (!TESTS_QUIET) printf("Trouble adding event\n");
			test_skip( name, __LINE__, "PAPI_add_event failed", 1 );
		}
	}

	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_overflow( EventSet, Event[ev], Threshold[ev], 0, my_handler )
			 != PAPI_OK ) {
			test_fail( name, __LINE__, "PAPI_overflow failed", 1 );
		}
	}

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail( name, __LINE__, "PAPI_start failed", 1 );
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
	char buf[100];

	int quiet,retval,result=0;

	/* Used to be able to set this via command line */
	num_events=1;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	do_cycles( 1 );

	zero_count(  );

	/* Init library */
	retval=PAPI_library_init( PAPI_VER_CURRENT );
	if (retval!=PAPI_VER_CURRENT) {
		test_fail( name, __LINE__, "PAPI_library_init failed", 1 );
	}

	name = argv[0];
	if (!quiet) printf( "[%d] %s, num_events = %d\n", getpid(  ), name, num_events );
	sprintf( buf, "%d", num_events );
	my_papi_start(  );
	run( name, 3 );

	print_here( "system(./child_overflow)" );

	if ( access( "./child_overflow", X_OK ) == 0 ) {
		if ( quiet) result=system( "./child_overflow TESTS_QUIET" );
		else result=system( "./child_overflow" );
	}
	else if ( access( "./ctests/child_overflow", X_OK ) == 0 ) {
		if ( quiet) result=system( "./ctests/child_overflow TESTS_QUIET" );
		else result=system( "./ctests/child_overflow" );
	}

	if (result<0) {
		test_fail(__FILE__,__LINE__,"system failed\n",1);
	}

	if (!quiet) printf("Successfully returned from system\n");

	// Rely on test_pass from child_overflow
	// otherwise the run_tests.sh output is ugly
	//test_pass(__FILE__);

	return 0;
}
