/* This file attempts to test the retired instruction event	*/
/* As implemented by PAPI_TOT_INS				*/

/* For more info on the causes of overcount on x86 systems	*/
/* See the ISPASS2013 paper:					*/
/*	"Non-Determinism and Overcount on Modern Hardware	*/
/*		Performance Counter Implementations"		*/

/* by Vince Weaver, <vincent.weaver@maine.edu>			*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"
#include "testcode.h"

#define NUM_RUNS 100


   /* Test a simple loop of 1 million instructions             */
   /* Most implementations should count be correct within 1%   */
   /* This loop in in assembly language, as compiler generated */
   /* code varies too much.                                    */

static void test_million(int quiet) {

	int i,result,ins_result;

	long long count,high=0,low=0,total=0,average=0;
	double error;
	int eventset=PAPI_NULL;

	if (!quiet) {
		printf("\nTesting a loop of 1 million instructions (%d times):\n",
		NUM_RUNS);
	}

	result=PAPI_create_eventset(&eventset);
	if (result!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", result );
	}

	result=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
	if (result!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_TOT_INS\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_TOT_INS", result );
	}

	for(i=0;i<NUM_RUNS;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		ins_result=instructions_million();

		result=PAPI_stop(eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_skip( __FILE__, __LINE__, "unimplemented", 0);
		}

		if (result!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_TOT_INS", result );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/NUM_RUNS;

	error=display_error(average,high,low,1000000ULL,quiet);

	if ((error > 1.0) || (error<-1.0)) {

#if defined(__PPC__)

		if(!quiet) {
			printf("If PPC is off by 50%%, this might be due to\n"
				"\"folded\" branch instructions on PPC32\n");
		}
#endif
		test_fail( __FILE__, __LINE__, "validation", result );

	}
}

/* Test fldcw.  Pentium 4 overcounts this instruction */

static void test_fldcw(int quiet) {

	(void)quiet;

#if defined(__i386__) || (defined __x86_64__)
	int i,result,ins_result;
	int eventset=PAPI_NULL;

	long long count,high=0,low=0,total=0,average=0;
	double error;

	if (!quiet) {
		printf("\nTesting a fldcw loop of 900,000 instructions (%d times):\n",
		NUM_RUNS);
	}

	result=PAPI_create_eventset(&eventset);
	if (result!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", result );
	}

	result=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
	if (result!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "adding PAPI_TOT_INS", result );
	}

	for(i=0;i<NUM_RUNS;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		ins_result=instructions_fldcw();

		result=PAPI_stop(eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			test_fail( __FILE__, __LINE__, "Code unimplemented", 1 );
		}

		if (result!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "Unexpected error on read", 1 );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/NUM_RUNS;

	error=display_error(average,high,low,900000ULL,quiet);

	if ((error > 1.0) || (error<-1.0)) {

		if (!quiet) {
			printf("On Pentium 4 machines, the fldcw instruction counts as 2.\n");
			printf("This will lead to an overcount of 22%%\n");
		}
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}
#endif
}

/* Test rep-prefixed instructions. */
/* HW counters count this as one each, not one per repeat */

static void test_rep(int quiet) {

	(void)quiet;

#if defined(__i386__) || (defined __x86_64__)
	int i,result,ins_result;
	int eventset=PAPI_NULL;

	long long count,high=0,low=0,total=0,average=0;
	double error;

	if(!quiet) {
		printf("\nTesting a 16k rep loop (%d times):\n", NUM_RUNS);
	}

	result=PAPI_create_eventset(&eventset);
	if (result!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", result );
	}

	result=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
	if (result!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "adding PAPI_TOT_INS", result );
	}

	for(i=0;i<NUM_RUNS;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		ins_result=instructions_rep();

		result=PAPI_stop(eventset,&count);

		if (ins_result==CODE_UNIMPLEMENTED) {
			fprintf(stderr,"\tCode unimplemented\n");
			test_fail( __FILE__, __LINE__, "Code unimplemented", 1 );
		}

		if (result!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "Unexpected error on read", 1 );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/NUM_RUNS;

	error=display_error(average,high,low,6002,quiet);

	if ((error > 10.0) || (error<-10.0)) {
		if (!quiet) {
			printf("Instruction count off by more than 10%%\n");
		}
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}
#endif
}

int main(int argc, char **argv) {

	int retval;
	int quiet=0;

	(void)argc;
	(void)argv;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("\nThis test checks that the \"PAPI_TOT_INS\" generalized "
			"event is working.\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	test_million(quiet);
	test_fldcw(quiet);
	test_rep(quiet);

	if (!quiet) printf("\n");

	test_pass( __FILE__ );

	PAPI_shutdown();

	return 0;
}
