/* This file attempts to test the retired branches taken	*/
/* performance counter PAPI_BR_TKN				*/

/* This measures taken *conditional* branches			*/
/* Though this may fall back to total if not available.		*/

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
	long long high=0,low=0,average=0;
	long long expected_cond=500000,expected_total=1000000;
	double error;

	long long count,total=0;
	int quiet=0,retval,ins_result;
	int eventset_total=PAPI_NULL;
	int eventset_conditional=PAPI_NULL;
	int eventset_taken=PAPI_NULL;
	int eventset_nottaken=PAPI_NULL;
	long long count_total,count_conditional,count_taken,count_nottaken;
	int cond_avail=1,nottaken_avail=1;
	int not_expected=0;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("\nTesting the PAPI_BR_TKN event.\n");
		printf("\tIt measures total number of conditional branches not taken\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create Total Eventset */
	retval=PAPI_create_eventset(&eventset_total);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset_total,"PAPI_BR_INS");
	if (retval!=PAPI_OK) {
		test_skip( __FILE__, __LINE__, "adding PAPI_BR_INS", retval );
	}


	/* Create Total Eventset */
	retval=PAPI_create_eventset(&eventset_conditional);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset_conditional,"PAPI_BR_CN");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_BR_CN\n");
		cond_avail=0;
		//test_skip( __FILE__, __LINE__, "adding PAPI_BR_CN", retval );
	}

	/* Create Taken Eventset */
	retval=PAPI_create_eventset(&eventset_taken);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset_taken,"PAPI_BR_TKN");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_BR_TKN\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_BR_TKN", retval );
	}

	/* Create Not-Taken Eventset */
	retval=PAPI_create_eventset(&eventset_nottaken);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset_nottaken,"PAPI_BR_NTK");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_BR_NTK\n");
		nottaken_avail=0;
		//test_skip( __FILE__, __LINE__, "adding PAPI_BR_NTK", retval );
	}

	/* Get total count */
	PAPI_reset(eventset_total);
	PAPI_start(eventset_total);
	ins_result=branches_testcode();
	retval=PAPI_stop(eventset_total,&count_total);

	/* Get conditional count */
	if (cond_avail) {
		PAPI_reset(eventset_conditional);
		PAPI_start(eventset_conditional);
		ins_result=branches_testcode();
		retval=PAPI_stop(eventset_conditional,&count_conditional);
	}

	/* Get taken count */
	PAPI_reset(eventset_taken);
	PAPI_start(eventset_taken);
	ins_result=branches_testcode();
	retval=PAPI_stop(eventset_taken,&count_taken);


	/* Get not-taken count */
	if (nottaken_avail) {
		PAPI_reset(eventset_nottaken);
		PAPI_start(eventset_nottaken);
		ins_result=branches_testcode();
		retval=PAPI_stop(eventset_nottaken,&count_nottaken);
	}

	if (!quiet) {
		printf("The test code has:\n");
		printf("\t%lld total branches\n",count_total);
		if (cond_avail) {
			printf("\t%lld conditional branches\n",count_conditional);
		}
		printf("\t%lld taken branches\n",count_taken);
		if (nottaken_avail) {
			printf("\t%lld not-taken branches\n",count_nottaken);
		}

	}

	if (!quiet) {
		printf("Testing a loop with %lld conditional taken branches (%d times):\n",
			expected_cond,num_runs);
	}

	for(i=0;i<num_runs;i++) {
		PAPI_reset(eventset_taken);
		PAPI_start(eventset_taken);

		ins_result=branches_testcode();

		retval=PAPI_stop(eventset_taken,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_skip( __FILE__, __LINE__, "unimplemented", 0);
		}

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_BR_TKN", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	error=display_error(average,high,low,expected_cond,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		not_expected=1;
		//test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) printf("\n");

	/* Check if using TOTAL instead of CONDITIONAL */
	if (not_expected) {

		error=display_error(average,high,low,expected_total,quiet);

		if ((error > 1.0) || (error<-1.0)) {
			if (!quiet) printf("Instruction count off by more than 1%%\n");
			test_fail( __FILE__, __LINE__, "Error too high", 1 );
		}
		else {
			test_warn(__FILE__,__LINE__,"Using TOTAL BRANCHES as base rather than CONDITIONAL BRANCHES\n",0);
		}
	}

	test_pass( __FILE__ );

	PAPI_shutdown();

	return 0;
}
