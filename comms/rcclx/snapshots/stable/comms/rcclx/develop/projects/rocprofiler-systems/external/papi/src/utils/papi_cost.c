/** file papi_cost.c
  * @brief papi_cost utility.
  *	@page papi_cost
  * @section  NAME
  *		papi_cost - computes execution time costs for basic PAPI operations.
  *
  *	@section Synopsis
  *		papi_cost [-dhps] [-b bins] [-t threshold]
  *
  *	@section Description
  *		papi_cost is a PAPI utility program that computes the min / max / mean / std. deviation
  *		of execution times for PAPI start/stop pairs and for PAPI reads.
  *		This information provides the basic operating cost to a user's program
  *		for collecting hardware counter data.
  *		Command line options control display capabilities.
  *
  *	@section Options
  *	<ul>
  *		<li>-b < bins > Define the number of bins into which the results are
  *			partitioned for display. The default is 100.
  *		<li>-d	Display a graphical distribution of costs in a vertical histogram.
  *		<li>-h	Display help information about this utility.
  *             <li>-p  Display 25/50/75 perecentile results for making boxplots.
  *		<li>-s	Show the number of iterations in each of the first 10
  *			standard deviations above the mean.
  *		<li>-t < threshold > 	Set the threshold for the number of iterations to
  *			measure costs. The default is 1,000,000.
  *	</ul>
  *
  *	@section Bugs
  *		There are no known bugs in this utility. If you find a bug,
  *		it should be reported to the PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "cost_utils.h"

/* Search for a derived event of type "type" */
static int
find_derived( int i , char *type)
{
	PAPI_event_info_t info;

	PAPI_enum_event( &i, PAPI_ENUM_FIRST );

	do {
		if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {

			if ( strcmp( info.derived, type) == 0 ) {
				return i;
			}
		}
	} while ( PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK );

	return PAPI_NULL;
}

/* Slight misnomer, find derived event != DERIVED_POSTFIX */
/* Will look for DERIVED_ADD, if not available will also accept DERIVED_SUB */
static int
find_derived_add( int i )
{
	int ret;

	ret = find_derived( i, "DERIVED_ADD");
	if (ret != PAPI_NULL) {
		return ret;
	}

	return find_derived( i, "DERIVED_SUB");
}

static int
find_derived_postfix( int i )
{
  return ( find_derived ( i, "DERIVED_POSTFIX" ) );
}

static void
print_help( void )
{
	printf( "This is the PAPI cost program.\n" );
	printf( "It computes min / max / mean / std. deviation for PAPI start/stop pairs; for PAPI reads, and for PAPI_accums.  Usage:\n\n" );
	printf( "    cost [options] [parameters]\n" );
	printf( "    cost TESTS_QUIET\n\n" );
	printf( "Options:\n\n" );
	printf( "  -b BINS       set the number of bins for the graphical distribution of costs. Default: 100\n" );
	printf( "  -d            show a graphical distribution of costs\n" );
	printf( "  -h            print this help message\n" );
	printf( "  -p            print 25/50/75th percentile results for making boxplots\n");
	printf( "  -s            show number of iterations above the first 10 std deviations\n" );
	printf( "  -t THRESHOLD  set the threshold for the number of iterations. Default: 1,000,000\n" );
	printf( "\n" );
}


static void
print_stats( int i, long long min, long long max, double average, double std )
{
	char *test[] = {
		"loop latency",
		"PAPI_start/stop (2 counters)",
		"PAPI_read (2 counters)",
		"PAPI_read_ts (2 counters)",
		"PAPI_accum (2 counters)",
		"PAPI_reset (2 counters)",
		"PAPI_read (1 derived_postfix counter)",
		"PAPI_read (1 derived_[add|sub] counter)"
	};

	printf( "\nTotal cost for %s over %d iterations\n",
		test[i], num_iters );
	printf( "min cycles   : %lld\nmax cycles   : %lld\n"
		"mean cycles  : %lf\nstd deviation: %lf\n",
		min, max, average, std );
}

static void
print_std_dev( int *s )
{
	int i;

	printf( "\n" );
	printf( "              --------# Standard Deviations Above the Mean--------\n" );
	printf( "0-------1-------2-------3-------4-------5-------6-------7-------8-------9-----10\n" );
	for ( i = 0; i < 10; i++ )
		printf( "  %d\t", s[i] );
	printf( "\n\n" );
}

static void
print_dist( long long min, long long max, int bins, int *d )
{
	int i, j;
	int step = ( int ) ( max - min ) / bins;

	printf( "\nCost distribution profile\n\n" );
	for ( i = 0; i < bins; i++ ) {
		printf( "%8d:", ( int ) min + ( step * i ) );
		if ( d[i] > 100 ) {
			printf( "**************************** %d counts ****************************",
				  d[i] );
		} else {
			for ( j = 0; j < d[i]; j++ )
				printf( "*" );
		}
		printf( "\n" );
	}
}

static void
print_percentile(long long percent25, long long percent50,
		long long percent75,long long percent99)
{
	printf("25%% cycles   : %lld\n50%% cycles   : %lld\n"
		"75%% cycles   : %lld\n99%% cycles   : %lld\n",
		percent25,percent50,percent75,percent99);
}

static void
do_output( int test_type, long long *array, int bins, int show_std_dev,
		   int show_dist, int show_percentile )
{
	int s[10];
	long long min, max;
	double average, std;
	long long percent25,percent50,percent75,percent99;

	std = do_stats( array, &min, &max, &average );
	print_stats( test_type, min, max, average, std );

	if (show_percentile) {
		do_percentile(array,&percent25,&percent50,&percent75,&percent99);
		print_percentile(percent25,percent50,percent75,percent99);
	}

	if ( show_std_dev ) {
		do_std_dev( array, s, std, average );
		print_std_dev( s );
	}

	if ( show_dist ) {
		int *d;
		d = calloc( bins , sizeof ( int ) );
		do_dist( array, min, max, bins, d );
		print_dist( min, max, bins, d );
		free( d );
	}
}


int
main( int argc, char **argv )
{
	int i, retval, EventSet = PAPI_NULL;
	int retval_start,retval_stop;
	int bins = 100;
	int show_dist = 0, show_std_dev = 0, show_percent = 0;
	long long totcyc, values[2];
	long long *array;
	int event;
	PAPI_event_info_t info;
	int c;

	/* Check command-line arguments */

	while ( (c=getopt(argc, argv, "hb:dpst:") ) != -1) {

		switch(c) {

			case 'h':
				print_help();
				exit(1);
			case 'b':
				bins=atoi(optarg);
				break;
			case 'd':
				show_dist=1;
				break;
			case 'p':
				show_percent=1;
				break;
			case 's':
				show_std_dev=1;
				break;
			case 't':
				num_iters=atoi(optarg);
				break;
			default:
				print_help();
				exit(1);
				break;
		}

	}

	printf( "Cost of execution for PAPI start/stop, read and accum.\n" );
	printf( "This test takes a while. Please be patient...\n" );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if (retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"PAPI_library_init\n");
		exit(retval);
	}
	retval = PAPI_set_debug( PAPI_VERB_ECONT );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"PAPI_set_debug\n");
		exit(retval);
	}
	retval = PAPI_query_event( PAPI_TOT_CYC );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"PAPI_query_event\n");
		exit(retval);
	}
	retval = PAPI_query_event( PAPI_TOT_INS );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"PAPI_query_event\n");
		exit(retval);
	}
	retval = PAPI_create_eventset( &EventSet );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"PAPI_create_eventset\n");
		exit(retval);
	}
	retval = PAPI_add_event( EventSet, PAPI_TOT_CYC );
	if (retval != PAPI_OK ) {
		fprintf(stderr,"PAPI_add_event\n");
		exit(retval);
	}
	retval = PAPI_add_event( EventSet, PAPI_TOT_INS );
	if (retval != PAPI_OK ) {
		retval = PAPI_add_event( EventSet, PAPI_TOT_IIS );
		if (retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_add_event\n");
			exit(retval);
		}
	}

	/* Make sure no errors and warm up */

	totcyc = PAPI_get_real_cyc(  );
	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_start");
		exit(retval);
	}
	if ( ( retval = PAPI_stop( EventSet, NULL ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_stop");
		exit(retval);
	}

	array = ( long long * ) malloc( ( size_t ) num_iters * sizeof ( long long ) );
	if ( array == NULL ) {
		fprintf(stderr,"Error allocating memory for results\n");
		exit(retval);
	}

	/* Determine clock latency */

	printf( "\nPerforming loop latency test...\n" );

	for ( i = 0; i < num_iters; i++ ) {
		totcyc = PAPI_get_real_cyc(  );
		totcyc = PAPI_get_real_cyc(  ) - totcyc;
		array[i] = totcyc;
	}

	do_output( 0, array, bins, show_std_dev, show_dist, show_percent );

	/* Start the start/stop eval */

	printf( "\nPerforming start/stop test...\n" );

	for ( i = 0; i < num_iters; i++ ) {
		totcyc = PAPI_get_real_cyc(  );
		retval_start=PAPI_start( EventSet );
		retval_stop=PAPI_stop( EventSet, values );
		totcyc = PAPI_get_real_cyc(  ) - totcyc;
		array[i] = totcyc;
		if (retval_start || retval_stop) {
		   fprintf(stderr,"PAPI start/stop\n");
			exit(retval_start );
		}
	}

	do_output( 1, array, bins, show_std_dev, show_dist, show_percent );

	/* Start the read eval */
	printf( "\nPerforming read test...\n" );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_start");
		exit(retval);
	}
	PAPI_read( EventSet, values );

	for ( i = 0; i < num_iters; i++ ) {
		totcyc = PAPI_get_real_cyc(  );
		PAPI_read( EventSet, values );
		totcyc = PAPI_get_real_cyc(  ) - totcyc;
		array[i] = totcyc;
	}
	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_stop");
		exit(retval);
	}

	do_output( 2, array, bins, show_std_dev, show_dist, show_percent );

	/* Start the read with timestamp eval */
	printf( "\nPerforming read with timestamp test...\n" );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_start");
		exit(retval);
	}
	PAPI_read_ts( EventSet, values, &totcyc );

	for ( i = 0; i < num_iters; i++ ) {
		PAPI_read_ts( EventSet, values, &array[i] );
	}
	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_stop");
		exit(retval);
	}

	/* post-process the timing array */
	for ( i = num_iters - 1; i > 0; i-- ) {
		array[i] -= array[i - 1];
	}
	array[0] -= totcyc;

	do_output( 3, array, bins, show_std_dev, show_dist, show_percent );

	/* Start the accum eval */
	printf( "\nPerforming accum test...\n" );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_start");
		exit(retval);
	}
	PAPI_accum( EventSet, values );

	for ( i = 0; i < num_iters; i++ ) {
		totcyc = PAPI_get_real_cyc(  );
		PAPI_accum( EventSet, values );
		totcyc = PAPI_get_real_cyc(  ) - totcyc;
		array[i] = totcyc;
	}
	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_stop");
		exit(retval);
	}

	do_output( 4, array, bins, show_std_dev, show_dist, show_percent );

	/* Start the reset eval */
	printf( "\nPerforming reset test...\n" );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_start");
		exit(retval);
	}

	for ( i = 0; i < num_iters; i++ ) {
		totcyc = PAPI_get_real_cyc(  );
		PAPI_reset( EventSet );
		totcyc = PAPI_get_real_cyc(  ) - totcyc;
		array[i] = totcyc;
	}
	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK ) {
		fprintf(stderr,"PAPI_stop");
		exit(retval);
	}

	do_output( 5, array, bins, show_std_dev, show_dist, show_percent );

	/* Derived POSTFIX event test */
	PAPI_cleanup_eventset( EventSet );

	event = 0 | PAPI_PRESET_MASK;

	event = find_derived_postfix( event );
	if ( event != PAPI_NULL ) {

		PAPI_get_event_info(event, &info);
		printf( "\nPerforming DERIVED_POSTFIX "
			"PAPI_read(%d counters)  test (%s)...",
			info.count, info.symbol );

		retval = PAPI_add_event( EventSet, event);
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_add_event");
			exit(retval);
		}

		retval = PAPI_start( EventSet );
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_start");
			exit(retval);
		}

		PAPI_read( EventSet, values );

		for ( i = 0; i < num_iters; i++ ) {
			totcyc = PAPI_get_real_cyc(  );
			PAPI_read( EventSet, values );
			totcyc = PAPI_get_real_cyc(  ) - totcyc;
			array[i] = totcyc;
		}

		retval = PAPI_stop( EventSet, values );
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_stop");
			exit(retval);
		}

		do_output( 6, array, bins, show_std_dev, show_dist, show_percent );

	} else {
		printf("\tI was unable to find a DERIVED_POSTFIX preset event "
			"to test on this architecture, skipping.\n");
	}

	/* Find a derived ADD event */
	PAPI_cleanup_eventset( EventSet );

	event = 0 | PAPI_PRESET_MASK;

	event = find_derived_add( event );
	if ( event != PAPI_NULL ) {

		PAPI_get_event_info(event, &info);
		printf( "\nPerforming DERIVED_[ADD|SUB] "
			"PAPI_read(%d counters)  test (%s)...",
			info.count, info.symbol );

		retval = PAPI_add_event( EventSet, event);
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_add_event\n");
			exit(retval);
		}

		retval = PAPI_start( EventSet );
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_start");
			exit(retval);
		}

		PAPI_read( EventSet, values );

		for ( i = 0; i < num_iters; i++ ) {
			totcyc = PAPI_get_real_cyc(  );
			PAPI_read( EventSet, values );
			totcyc = PAPI_get_real_cyc(  ) - totcyc;
			array[i] = totcyc;
	  	}

		retval = PAPI_stop( EventSet, values );
		if ( retval != PAPI_OK ) {
			fprintf(stderr,"PAPI_stop");
			exit(retval);
		}

		do_output( 7, array, bins, show_std_dev, show_dist, show_percent );
	} else {
		printf("\tI was unable to find a suitable DERIVED_[ADD|SUB] "
			"event to test, skipping.\n");
	}

	free( array );

	return 0;
}
