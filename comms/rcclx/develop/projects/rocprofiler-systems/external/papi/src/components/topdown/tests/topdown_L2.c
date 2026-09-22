/*
 * Specifically tests that the Level 2 topdown events make sense.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS  8
#define PERC_TOLERANCE  1.5

// fibonacci function to serve as a benchable code section
void __attribute__((optimize("O0"))) fib(int n)
{
	long i, a = 0;
	int b = 1;
	for (i = 0; i < n; i++)
	{
		b = b + a;
		a = b - a;
	}
}

int main(int argc, char **argv)
{
	int i, quiet, retval;
	int EventSet = PAPI_NULL;
	const PAPI_component_info_t *cmpinfo = NULL;
	int numcmp, cid, topdown_cid = -1;
	long long values[NUM_EVENTS];
	double tmp;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet(argc, argv);

	/* PAPI Initialization */
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
	{
		test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", retval);
	}

	if (!quiet)
	{
		printf("Testing topdown component with PAPI %d.%d.%d\n",
			   PAPI_VERSION_MAJOR(PAPI_VERSION),
			   PAPI_VERSION_MINOR(PAPI_VERSION),
			   PAPI_VERSION_REVISION(PAPI_VERSION));
	}

	/*******************************/
	/* Find the topdown component  */
	/*******************************/
	numcmp = PAPI_num_components();
	for (cid = 0; cid < numcmp; cid++)
	{
		if ((cmpinfo = PAPI_get_component_info(cid)) == NULL)
		{
			test_fail(__FILE__, __LINE__, "PAPI_get_component_info failed\n", 0);
		}
		if (!quiet)
		{
			printf("\tComponent %d - %d events - %s\n", cid,
				   cmpinfo->num_native_events,
				   cmpinfo->name);
		}
		if (strstr(cmpinfo->name, "topdown"))
		{
			topdown_cid = cid;

			/* check that the component is enabled */
			if (cmpinfo->disabled)
			{
				printf("Topdown component is disabled: %s\n", cmpinfo->disabled_reason);
				test_fail(__FILE__, __LINE__, "The TOPDOWN component is not enabled\n", 0);
			}
		}
	}

	if (topdown_cid < 0)
	{
		test_skip(__FILE__, __LINE__, "Topdown component not found\n", 0);
	}

	if (!quiet)
	{
		printf("\nFound Topdown Component at id %d\n", topdown_cid);
		printf("\nAdding the level 2 topdown metrics..\n");
	}

	/* Create EventSet */
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
				  "PAPI_create_eventset()", retval);
	}

	/* Add the level 2 topdown metrics */
	/* if we can't, just skip because not all processors support level 2 */
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_HEAVY_OPS_PERC");
	if (retval != PAPI_OK)
	{
		test_skip(__FILE__, __LINE__,
			"Error adding TOPDOWN_HEAVY_OPS_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_LIGHT_OPS_PERC");
	if (retval != PAPI_OK)
	{
		/* if the first L2 event was successfully added though, */
		/* subsequent failures indicate a deeper problem */
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_LIGHT_OPS_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_BR_MISPREDICT_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_BR_MISPREDICT_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_MACHINE_CLEARS_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_MACHINE_CLEARS_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_FETCH_LAT_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_FETCH_LAT_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_FETCH_BAND_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_FETCH_BAND_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_MEM_BOUND_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_MEM_BOUND_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_CORE_BOUND_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_CORE_BOUND_PERC", retval);
	}

	/* stat a loop-based calculation of the sum of the fibonacci sequence */
	/* the workload needs to be fairly large in order to acquire an accurate */
	/* set of measurements */
	PAPI_start(EventSet);
	fib(6000000);
	PAPI_stop(EventSet, values);

	/* run some sanity checks: */
	
	/* first, the sum of all level 2 metric percentages should be 100% */
	tmp = 0;
	for (i=0; i<NUM_EVENTS; i++) {
		tmp += *((double *)(&values[i]));
	}
	if (!quiet)
		printf("L2 metric percentages sum to %.2f%%:\n", tmp);
	if (tmp < 100 - PERC_TOLERANCE || tmp > 100 + PERC_TOLERANCE) {
		test_fail(__FILE__, __LINE__,
			"Level 2 topdown metric percentages did not sum to 100%%\n", 1);
	}

	/* next, check that we are retiring more light ops than heavy ops */
	/* this is a very reasonable expectation for a simple loop performing
	/* scalar add and multiply operations */
	if (!quiet) {
		printf("\tHeavy ops:\t%.1f%%\n", *((double *)(&values[0])));
		printf("\tLight ops:\t%.1f%%\n", *((double *)(&values[1])));

	}
	if (*((double *)(&values[0])) > *((double *)(&values[1]))) {
		test_warn(__FILE__, __LINE__,
			"Heavy ops should be much smaller than light ops", 1);
	}

	/* next, check that the branch mispredictions and machine clears */
	/* are insignificant as this benchmark should have good speculation */
	if (!quiet) {
		printf("\tBranch mispredictions:\t%.1f%%\n", *((double *)(&values[2])));
		printf("\tMachine clears:\t%.1f%%\n", *((double *)(&values[3])));
	}
	if ((*((double *)(&values[2])) + *((double *)(&values[3]))) > 5.0) {
		test_warn(__FILE__, __LINE__,
			"Bad speculation should be insignificant for this workload", 1);
	}

	/* next, check that the fetch latency and bandwidth are insignificant */
	if (!quiet) {
		printf("\tFetch latency:\t%.1f%%\n", *((double *)(&values[4])));
		printf("\tFetch bandwidth:\t%.1f%%\n", *((double *)(&values[5])));
	}
	if ((*((double *)(&values[4])) + *((double *)(&values[5]))) > 10.0) {
		test_warn(__FILE__, __LINE__,
			"Frontend bound should be insignificant for this workload", 1);
	}

	/* finally, check that core bound is greater than memory bound. */
	/* we can expect this because there are no memory loads/stores here */
	if (!quiet) {
		printf("\tMemory bound:\t%.1f%%\n", *((double *)(&values[6])));
		printf("\tCore bound:\t%.1f%%\n", *((double *)(&values[7])));
	}
	if (*((double *)(&values[6])) > *((double *)(&values[7]))) {
		test_warn(__FILE__, __LINE__,
			"The workload should be significantly more core bound than memory bound", 1);
	}

	return 0;
}