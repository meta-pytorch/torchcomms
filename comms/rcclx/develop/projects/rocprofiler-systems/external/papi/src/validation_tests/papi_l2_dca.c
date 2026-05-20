/* This code attempts to test the L2 Data Cache Acceesses	*/
/* performance counter PAPI_L2_DCA				*/

/* Notes:							*/
/*	Should this be equivelent to PAPI_L1_DCM?		*/
/*		(on IVY it is)					*/
/*	On Haswell/Broadwell/Skylake this maps to :		*/
/*		L2_RQSTS:ALL_DEMAND_REFERENCES			*/

/*	Should this include *all* L2 accesses or just those	*/
/*		caused by the user?  Prefetch? MESI?		*/



/* by Vince Weaver, vincent.weaver@maine.edu			*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#include "cache_helper.h"
#include "display_error.h"

#include "testcode.h"

#define NUM_RUNS	100

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
		printf("Testing the PAPI_L2_DCA event\n");
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

	retval=PAPI_add_named_event(eventset,"PAPI_L2_DCA");
	if (retval!=PAPI_OK) {
		test_skip( __FILE__, __LINE__, "adding PAPI_L2_DCA", retval );
	}

	l1_size=get_cachesize(L1D_CACHE);
	l1_linesize=get_linesize(L1D_CACHE);
	l2_size=get_cachesize(L2_CACHE);
	l2_linesize=get_linesize(L2_CACHE);
	l2_entries=get_entries(L2_CACHE);

	if ((l2_size==0) || (l2_linesize==0)) {
		if (!quiet) {
			printf("Unable to determine size of L2 cache!\n");
		}
		test_skip( __FILE__, __LINE__, "adding PAPI_L2_DCA", retval );
	}

	if (!quiet) {
		printf("\tDetected %dk L1 DCache, %dB linesize\n",
			l1_size/1024,l1_linesize);
		printf("\tDetected %dk L2 DCache, %dB linesize, %d entries\n",
			l2_size/1024,l2_linesize,l2_entries);
	}

	arraysize=l2_size/sizeof(double);

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
		printf("\nWrite Test: Initializing an array of %d doubles:\n",
			arraysize);
        }

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		cache_write_test(array,arraysize);

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_L2_DCA", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	expected=arraysize/(l2_linesize/sizeof(double));

	if (!quiet) {
		printf("\tShould be roughly "
			"arraysize/L2_linesize/double_size (%d/%d/%zu): "
			"%lld\n\n",
			arraysize,l2_linesize,sizeof(double),
			expected);
	}

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		errors++;
	}

	if (!quiet) printf("\n");

	/******************/
	/* Testing Reads  */
	/******************/

	if (!quiet) {
		printf("\nRead Test: Summing an array of %d doubles:\n",
			arraysize);
        }

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		aSumm+=cache_read_test(array,arraysize);

		retval=PAPI_stop(eventset,&count);

		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"reading PAPI_L2_DCA", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=(total/num_runs);

	expected=arraysize/(l2_linesize/sizeof(double));

	if (!quiet) {
		printf("\tShould be roughly "
			"arraysize/L2_linesize/double_size (%d/%d/%zu): "
			"%lld\n\n",
			arraysize,l2_linesize,sizeof(double),
			expected);
	}

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		errors++;
	}

	if (!quiet) {
		printf("\n");
	}

	/* Warn for now, as we get errors we can't easily */
	/* explain on haswell and more recent Intel chips */
	if (errors) {
		test_warn( __FILE__, __LINE__, "Error too high", 1 );
	}

	test_pass(__FILE__);

	return 0;
}
