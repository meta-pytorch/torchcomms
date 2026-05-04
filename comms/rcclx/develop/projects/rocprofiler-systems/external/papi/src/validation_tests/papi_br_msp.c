/* This file attempts to test the mispredicted branches		*/
/* performance event as counted by PAPI_BR_MSP			*/

/* by Vince Weaver, <vincent.weaver@maine.edu>			*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"
#include "testcode.h"



int main(int argc, char **argv) {

	int num_runs=100,i;
	int num_random_branches=500000;
	long long high=0,low=0,average=0,expected=1500000;

	long long count,total=0;
	int quiet=0,retval,ins_result;
	int total_eventset=PAPI_NULL,miss_eventset=PAPI_NULL;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("\nTesting the PAPI_BR_MSP event.\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create total eventset */
	retval=PAPI_create_eventset(&total_eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(total_eventset,"PAPI_BR_INS");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_BR_INS\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_BR_INS", retval );
	}

	/* Create miss eventset */
	retval=PAPI_create_eventset(&miss_eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(miss_eventset,"PAPI_BR_MSP");
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "adding PAPI_BR_MSP", retval );
	}

	if (!quiet) {
		printf("\nPart 1: Testing that easy to predict loop has few misses\n");
		printf("Testing a loop with %lld branches (%d times):\n",
			expected,num_runs);
		printf("\tOn a simple loop like this, "
			"miss rate should be very small.\n");
	}

	for(i=0;i<num_runs;i++) {

		PAPI_reset(miss_eventset);
                PAPI_start(miss_eventset);

		ins_result=branches_testcode();

		retval=PAPI_stop(miss_eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_skip( __FILE__, __LINE__, "unimplemented", 0);
		}

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_TOT_INS", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	if (!quiet) printf("\tAverage number of branch misses: %lld\n",average);

	if (average>1000) {
		if (!quiet) printf("Branch miss rate too high\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	/*******************/

	if (!quiet) {
		printf("\nPart 2\n");
	}

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {
		PAPI_reset(total_eventset);
                PAPI_start(total_eventset);

		ins_result=random_branches_testcode(num_random_branches,1);

		retval=PAPI_stop(total_eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_skip( __FILE__, __LINE__, "unimplemented", 0);
		}

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_TOT_INS", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/num_runs;

	expected=average;
	if (!quiet) {
		printf("\nTesting a function that branches "
			"based on a random number\n");
		printf("   The loop has %lld branches\n",expected);
		printf("   %d are random branches.\n",num_random_branches);
	}

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {
		PAPI_reset(miss_eventset);
                PAPI_start(miss_eventset);

		ins_result=random_branches_testcode(num_random_branches,1);

		retval=PAPI_stop(miss_eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_skip( __FILE__, __LINE__, "unimplemented", 0);
		}

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading eventset", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/num_runs;

	if (!quiet) {
                double rate = 100.0*(double)average/(double)num_random_branches;
		printf("\nOut of %d random branches %lld were mispredicted,\n",num_random_branches,average);
		printf("resulting in a misprediction rate = %.1lf%%.\n",rate);
		printf("Assuming a good random number generator and no freaky luck\n");
		printf("the misprediction rate should be around 50%%, and the\n");
                printf("mispredicts should at least be between %d and %d.\n",
			num_random_branches/4,(num_random_branches/4)*3);
	}

	if ( average < (num_random_branches/4)) {
		if (!quiet) printf("Mispredicts too low\n");
		test_fail( __FILE__, __LINE__, "Error too low", 1 );
	}

	if (average > (num_random_branches/4)*3) {
		if (!quiet) printf("Mispredicts too high\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) printf("\n");

	test_pass( __FILE__ );

	PAPI_shutdown();

	return 0;

}
