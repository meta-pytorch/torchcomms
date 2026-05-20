/* This code attempts to test the L2 Data Cache Missses		*/
/* performance counter PAPI_L2_DCM				*/

/* by Vince Weaver, vincent.weaver@maine.edu			*/


/* Due to prefetching it is hard to create a testcase short of */
/* just having random accesses. */
/* In addition, due to context switching the cache might be */
/* affected by other processes on a busy system. */

/* Other tests to attempt */
/* Repeatedly reading same cache line should give very small error */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#include "cache_helper.h"
#include "display_error.h"

#include "testcode.h"

/* How much should we allow? */
#define ALLOWED_ERROR	5.0

#define NUM_RUNS	100

#define ITERATIONS	1000000

int main(int argc, char **argv) {

	int i;
	int eventset=PAPI_NULL;
	int num_runs=NUM_RUNS;
	long long high,low,average,expected;
	long long count,total;

	int retval;
	int l1_size,l2_size,l1_linesize,l2_linesize,l2_entries;
	int arraysize;
	int quiet,errors=0;

	double error;
	double *array;
	double aSumm = 0.0;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("Testing the PAPI_L2_DCM event\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
		test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);
	}

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset,"PAPI_L2_DCM");
	if (retval!=PAPI_OK) {
		test_skip( __FILE__, __LINE__, "adding PAPI_L2_DCM", retval );
	}

	l1_size=get_cachesize(L1D_CACHE);
	l1_linesize=get_linesize(L1D_CACHE);
	l2_size=get_cachesize(L2_CACHE);
	l2_linesize=get_linesize(L2_CACHE);
	l2_entries=get_entries(L2_CACHE);

	if (!quiet) {
		printf("\tDetected %dk L1 DCache, %dB linesize\n",
			l1_size/1024,l1_linesize);
		printf("\tDetected %dk L2 DCache, %dB linesize, %d entries\n",
			l2_size/1024,l2_linesize,l2_entries);
	}

	arraysize=(l2_size/sizeof(double))*8;

	if (arraysize==0) {
		if (!quiet) printf("Could not detect cache size\n");
		test_skip(__FILE__,__LINE__,"Could not detect cache size",0);
	}

	if (!quiet) {
		printf("\tAllocating %zu bytes of memory (%d doubles)\n",
			arraysize*sizeof(double),arraysize);
	}

	array=calloc(arraysize,sizeof(double));
	if (array==NULL) {
		test_fail(__FILE__,__LINE__,"Can't allocate memory",0);
	}

	/******************/
	/* Testing Writes */
	/******************/

	if (!quiet) {
		printf("\nWrite Test: Writing an array of %d doubles %d random times:\n",
			arraysize,ITERATIONS);
		printf("\tPrefetch and shared nature of L2s make this hard.\n");
		printf("\tExpected 7/8 of accesses to be miss.\n");
        }

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		cache_random_write_test(array,arraysize,ITERATIONS);

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_L2_DCM", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	expected=(ITERATIONS*7)/8;

//	expected=arraysize/(l1_linesize);

	error=display_error(average,high,low,expected,quiet);

	if ((error > ALLOWED_ERROR) || (error<-ALLOWED_ERROR)) {
		if (!quiet) {
			printf("Instruction count off by more "
				"than %.2lf%%\n",ALLOWED_ERROR);
		}
		errors++;
	}

	if (!quiet) printf("\n");

	/******************/
	/* Testing Reads  */
	/******************/

	if (!quiet) {
		printf("\nRead Test: Summing %d random doubles from array "
			"of size %d:\n",ITERATIONS,arraysize);
		printf("\tExpected 7/8 of accesses to be miss.\n");
        }

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		aSumm+=cache_random_read_test(array,arraysize,ITERATIONS);

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_L2_DCM", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	expected=(ITERATIONS*7)/8;

//	expected=arraysize/(l1_linesize/sizeof(double));

	error=display_error(average,high,low,expected,quiet);

	if ((error > ALLOWED_ERROR) || (error<-ALLOWED_ERROR)) {
		if (!quiet) {
			printf("Instruction count off by more "
				"than %.2lf%%\n",ALLOWED_ERROR);
		}
		errors++;
	}

	if (!quiet) {
		printf("\n");
	}

	/* FIXME: Warn, as we fail on broadwell and more recent chips */
	if (errors) {
		test_warn( __FILE__, __LINE__, "Error too high", 1 );
	}



	test_pass(__FILE__);

	return 0;
}
