/* This file attempts to test the PAPI_TOT_CYC	*/
/* performance counter.				*/

/* by Vince Weaver, <vincent.weaver@maine.edu>	*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include <time.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"

#include "matrix_multiply.h"

#define SLEEP_RUNS 3


static long long convert_to_ns(struct timespec *before,
					struct timespec *after) {

	long long seconds;
	long long ns;

	seconds=after->tv_sec - before->tv_sec;
	ns = after->tv_nsec - before->tv_nsec;

	ns = (seconds*1000000000ULL)+ns;

	return ns;
}


int main(int argc, char **argv) {

	int quiet;
	double mmm_ghz;

	double error;

	int i;
	long long count,high=0,low=0,total=0,average=0;
	long long nsecs;
	long long mmm_count;
	long long expected;
	int retval;
	int eventset=PAPI_NULL;

	struct timespec before,after;

	quiet=tests_quiet(argc,argv);

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("\nTesting PAPI_TOT_CYC\n\n");
	}

	if (!quiet) {
		printf("Testing a sleep of 1 second (%d times):\n",SLEEP_RUNS);
	}

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset,"PAPI_TOT_CYC");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_TOT_CYC\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_TOT_CYC", retval );
	}


	for(i=0;i<SLEEP_RUNS;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		sleep(1);

		retval=PAPI_stop(eventset,&count);
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/SLEEP_RUNS;

	if (!quiet) {
		printf("\tAverage should be low, as no user cycles when sleeping\n");
		printf("\tMeasured average: %lld\n",average);
	}

	if (average>100000) {
		if (!quiet) printf("Average cycle count too high!\n");
		test_fail( __FILE__, __LINE__, "idle average", retval );
	}

	/*****************************/
	/* testing Matrix Matrix GHz */
	/*****************************/

	if (!quiet) {
		printf("\nEstimating GHz with matrix matrix multiply\n");
	}

   // We have a problem with subsequent runs requiring fewer cycles than
   // the first run. This may be system dependent; so we do a throw-away
   // run here, so we end up comparing the SECOND run to the 3rd, 4th, etc.

   naive_matrix_multiply(quiet);                // A first run, to init system.

	clock_gettime(CLOCK_REALTIME,&before);

	PAPI_reset(eventset);
	PAPI_start(eventset);

	naive_matrix_multiply(quiet);

	retval=PAPI_stop(eventset,&count);

	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "Problem stopping!", retval );
	}

	clock_gettime(CLOCK_REALTIME,&after);

	nsecs=convert_to_ns(&before,&after);

	mmm_ghz=(double)count/(double)nsecs;

	if (!quiet) {
		printf("\tActual measured cycles = %lld\n",count);
		printf("\tEstimated actual GHz = %.2lfGHz\n",mmm_ghz);
	}

	mmm_count=count;

	/************************************/
	/* Check for Linear Speedup         */
	/************************************/

	if (!quiet) printf("\nTesting for a linear cycle increase\n");

#define REPITITIONS 2

	clock_gettime(CLOCK_REALTIME,&before);

	PAPI_reset(eventset);
	PAPI_start(eventset);

	for(i=0;i<REPITITIONS;i++) {
		naive_matrix_multiply(quiet);
	}

	retval=PAPI_stop(eventset,&count);

	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "Problem stopping!", retval );
	}

	clock_gettime(CLOCK_REALTIME,&after);

	nsecs=convert_to_ns(&before,&after);

	expected=mmm_count*REPITITIONS;

	error=  100.0 * (double)(count-expected) / (double)expected;

	if (!quiet) {
		printf("\tExpected %lld, got %lld\n",expected,count);
		printf("\tError=%.2f%%\n",error);
	}

	if ((error>10.0) || (error<-10.0)) {

		if (!quiet) printf("Error too high!\n");
		test_fail( __FILE__, __LINE__, "Error too high", retval );
	}

	if (!quiet) printf("\n");

	test_pass( __FILE__ );

	return 0;
}
