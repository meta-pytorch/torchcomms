/*
 * Specifically tests that the Level 1 topdown events make sense.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS  4
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
				test_fail(__FILE__, __LINE__, "Component is not enabled\n", 0);
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
		printf("\nAdding the level 1 topdown metrics..\n");
	}

	/* Create EventSet */
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
				  "PAPI_create_eventset()", retval);
	}

	/* Add the level 1 topdown metrics */
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_RETIRING_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_RETIRING_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_BAD_SPEC_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_BAD_SPEC_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_FE_BOUND_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_FE_BOUND_PERC", retval);
	}
	retval = PAPI_add_named_event(EventSet, "TOPDOWN_BE_BOUND_PERC");
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
			"Error adding TOPDOWN_BE_BOUND_PERC", retval);
	}

	/* stat a loop-based calculation of the sum of the fibonacci sequence */
	/* the workload needs to be fairly large in order to acquire an accurate */
	/* set of measurements */
	PAPI_start(EventSet);
	fib(6000000);
	PAPI_stop(EventSet, values);

	/* run some sanity checks: */
	
	/* first, the sum of all level 1 metric percentages should be 100% */
	tmp = 0;
	for (i=0; i<NUM_EVENTS; i++) {
		tmp += *((double *)(&values[i]));
	}
	if (!quiet)
		printf("L1 metric percentages sum to %.2f%%\n", tmp);
	if (tmp < 100 - PERC_TOLERANCE || tmp > 100 + PERC_TOLERANCE) {
		test_fail(__FILE__, __LINE__,
			"Level 1 topdown metric percentages did not sum to 100%%\n", 1);
	}

	if (!quiet)
		printf("\tRetiring:\t%.1f%%\n", *((double *)(&values[0])));

	/* next, verify that the percentage of bad spec slots is reasonable. */
	/* for this benchmark, we can expect very low rate of bad speculation */
	/* due to the fact that it consists of a simple for loop */
	if (!quiet)
		printf("\tBad spec:\t%.1f%%\n", *((double *)(&values[1])));
	if (*((double *)(&values[1])) > 5.0) {
		test_warn(__FILE__, __LINE__,
			"The percentage of slots affected by bad speculation was unexpectedly high", 1);
	}

	/* finally, make sure the frontend/backend bound percentages make sense */
	/* we should expect this benchmark to be significantly more limited */
	/* by the back end, so check that be bound is larger than the fe bound */
	if (!quiet) {
		printf("\tFrontend bound:\t%.1f%%\n", *((double *)(&values[2])));
		printf("\tBackend bound:\t%.1f%%\n", *((double *)(&values[3])));

	}
	if (*((double *)(&values[2])) > *((double *)(&values[3]))) {
		test_warn(__FILE__, __LINE__,
			"Frontend bound should be significantly smaller than backend bound", 1);
	}

	return 0;
}