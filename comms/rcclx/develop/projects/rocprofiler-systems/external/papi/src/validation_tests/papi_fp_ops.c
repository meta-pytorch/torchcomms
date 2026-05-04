/* This file attempts to test the floating point		*/
/* performance counter PAPI_FP_OPS				*/

/* by Vince Weaver, <vincent.weaver@maine.edu>			*/

/* Note!  There are many many many things that can go wrong	*/
/* when trying to get a sane floating point measurement.	*/


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
	long long high=0,low=0,average=0,expected=1500000;
	double error,double_result;

	long long count,total=0;
	int quiet=0,retval,ins_result;
	int eventset=PAPI_NULL;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("\nTesting the PAPI_FP_OPS event.\n\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create the eventset */
	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* Add FP_OPS event */
	retval=PAPI_add_named_event(eventset,"PAPI_FP_OPS");
	if (retval!=PAPI_OK) {
		if (!quiet) fprintf(stderr,"PAPI_FP_OPS not available!\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_FP_OPS", retval );
	}

	/**************************************/
	/* Test a loop with no floating point */
	/**************************************/
	total=0; high=0; low=0;
	expected=0;

	if (!quiet) {
		printf("Testing a loop with %lld floating point (%d times):\n",
			expected,num_runs);
	}

	for(i=0;i<num_runs;i++) {
		PAPI_reset(eventset);
		PAPI_start(eventset);

		ins_result=branches_testcode();

		retval=PAPI_stop(eventset,&count);

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

	error=display_error(average,high,low,expected,quiet);

	if (average>10) {
		if (!quiet) printf("Unexpected FP event value\n");
		test_fail( __FILE__, __LINE__, "Unexpected FP event", 1 );
	}

	if (!quiet) printf("\n");

	/*******************************************/
	/* Test a single-precision matrix multiply */
	/*******************************************/
	total=0; high=0; low=0;
	expected=flops_float_init_matrix();

	num_runs=3;

	if (!quiet) {
		printf("Testing a matrix multiply with %lld single-precision FP operations (%d times)\n",
			expected,num_runs);
	}

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		double_result=flops_float_matrix_matrix_multiply();

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_TOT_INS", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	if (!quiet) printf("Result %lf\n",double_result);

	average=(total/num_runs);

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) printf("\n");


	/*******************************************/
	/* Test a double-precision matrix multiply */
	/*******************************************/
	total=0; high=0; low=0;
	expected=flops_double_init_matrix();

	num_runs=3;

	if (!quiet) {
		printf("Testing a matrix multiply with %lld double-precision FP operations (%d times)\n",
			expected,num_runs);
	}

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		double_result=flops_double_matrix_matrix_multiply();

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_TOT_INS", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	if (!quiet) printf("Result %lf\n",double_result);

	average=(total/num_runs);

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) printf("\n");

	test_pass( __FILE__ );

	PAPI_shutdown();

	return 0;
}
