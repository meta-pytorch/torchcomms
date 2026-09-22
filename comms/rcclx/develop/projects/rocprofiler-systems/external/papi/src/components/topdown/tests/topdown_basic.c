/*
 * Basic test that just adds all of the topdown events and make sure they dont
 * produce any errors.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

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
	int code, maximum_code = 0;
	char event_name[PAPI_MAX_STR_LEN];
	PAPI_event_info_t event_info;
	int num_events = 0;
	long long *values;

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
		printf("\nListing all events in this component:\n");
	}

	/* Create EventSet */
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK)
	{
		test_fail(__FILE__, __LINE__,
				  "PAPI_create_eventset()", retval);
	}

	/*****************************************************/
	/* Add all the events to an eventset as a basic test */
	/*****************************************************/
	code = PAPI_NATIVE_MASK;
	retval = PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, topdown_cid);

	while (retval == PAPI_OK)
	{
		if (PAPI_event_code_to_name(code, event_name) != PAPI_OK)
		{
			printf("Error translating %#x\n", code);
			test_fail(__FILE__, __LINE__,
					  "PAPI_event_code_to_name", retval);
		}

		if (PAPI_get_event_info(code, &event_info) != PAPI_OK)
		{
			printf("Error getting info for event %#x\n", code);
			test_fail(__FILE__, __LINE__,
					  "PAPI_get_event_info()", retval);
		}

		retval = PAPI_add_event(EventSet, code);
		if (retval != PAPI_OK)
		{
			test_fail(__FILE__, __LINE__,
					  "PAPI_add_event()", retval);
		}

		if (!quiet)
		{
			printf("\tEvent %#x: %s -- %s\n",
				   code, event_name, event_info.long_descr);
		}

		num_events += 1;
		maximum_code = code;
		retval = PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, topdown_cid);
	}
	if (!quiet)
		printf("\n");

	/* ensure there is space for the output values */
	values = calloc(num_events, sizeof(long long));
	if (values == NULL)
	{
		test_fail(__FILE__, __LINE__,
				  "Insufficient memory", retval);
	}

	/* now stat some code to make sure the events work */
	PAPI_start(EventSet);
	fib(6000000);
	PAPI_stop(EventSet, values);

	if (!quiet)
		printf("Values:\n");
	for (i = 0; i < num_events; i++)
	{
		/* ensure the metric percentages are between 0 and 100 */
		if (*((double *)(&values[i])) < 0 || *((double *)(&values[i])) > 100.0)
		{
			test_fail(__FILE__, __LINE__,
					  "Topdown metric was not a valid percentage", retval);
		}

		if (!quiet)
			printf("\t%d:\t%.1lf%%\n", i, *((double *)(&values[i])));
	}

	return 0;
}