/*
 * Copyright (c) 2020 Intel Corp. All rights reserved
 * Contributed by Peinan Zhang  <peinan.zhang@intel.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Based on Intel Software Optimization Guide 2015
 */


/*
 *  This test case reports all available Intel GPU performance metrics.
 *
 *  @brief  list all available metrics groups, all metrics or the metrics in a certain metrics group
 *
 *  @option:
 *	 [-a (all metrics)] [-g <metricGroupName>] [-m <metricName>]
 *
 *	 Example:
 *	  metricGroupName "ComputeBasic"
 *	  metricName	  "ComputeBasic.GpuTime"
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"

#define COMP_NAME	"intel_gpu"

int numDevice = 0;

#define MAX_NUM_METRICS	40
#define MAX_STRLEN		 128 

long long metric_values[MAX_NUM_METRICS];

#define TEST_METRIC  "ComputeBasic.GpuTime"

int
main(int argc, char ** argv) {

	(void)argc;
	(void)argv;

	int	i = 0;
	int	cid = -1;
	int	numMetrics = 0;
	int	numGroups = 0;
	int	selectedAll = 1;
	int	listGroupOnly = 0;
	char *metricName = NULL;
	char *metricGroupName = NULL;
	int	retVal = PAPI_OK;
	PAPI_event_info_t info;
	char ch;

	while ((ch=getopt(argc, argv, "ahm:g:")) != -1) {
		switch(ch) {
			case 'a':
				listGroupOnly = 1;
				if (metricGroupName) {
					printf("list all groups, ignore group name\n");
					metricGroupName = NULL;
				}
				break;
			case 'h': 
				printf("usage:  %s [-a] [-g <metricGroupName>] [-m <metricName>]\n", argv[0]);
				return 0;
			case 'm': 
				metricName = optarg;
				selectedAll = 0;
				break;
			case 'g': 
				if (!listGroupOnly) {
					metricGroupName = optarg;
				} else {
					printf("list all groups, ignore group name\n"); 
				}
				break;
			default:
				break;
		}
	}

	PAPI_component_info_t *aComp =  NULL;

	retVal = PAPI_library_init( PAPI_VER_CURRENT );
	if( retVal != PAPI_VER_CURRENT ) {
		fprintf( stderr, "PAPI_library_init failed\n" );
		exit(-1);
	}

	int numComps  = PAPI_num_components();
	for (i=0; i<numComps; i++) {
		aComp = (PAPI_component_info_t*) PAPI_get_component_info(i);   // get the component info.
		if (aComp && (!strcmp(COMP_NAME, aComp->name))) {
			cid=i;
			break;
		}
	}
	if (i == numComps) {
		printf("Component %s is not supported\n", aComp->name);
		return 1;
	}
 
	printf("Name:  %s\n", aComp->name);
	printf("Description: %s\n", aComp->description);

	if (selectedAll || listGroupOnly) {
		i = 0 | PAPI_NATIVE_MASK; 
		retVal=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );
		numMetrics = 0;
		do {
			memset( &info, 0, sizeof ( info ) );
			retVal =  PAPI_get_event_info( i, &info );	
			if (retVal == PAPI_OK) {
				if (listGroupOnly) {
					if (!metricGroupName || !strstr(info.symbol, metricGroupName)) {
						char *pt = index(info.symbol, '.');
						if (pt) {
							*pt = '\0';
						}
						if (metricGroupName) {
							free(metricGroupName);
						}
						metricGroupName = strdup(info.symbol);
						printf("%s\n", metricGroupName);
						numGroups++;
					}
				} else if ((!metricGroupName) || strstr(info.symbol, metricGroupName)) {
					printf("%s\n\t%s\n", info.symbol, info.long_descr);
					numMetrics++;
				}
			}
			retVal = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid );
		} while (retVal == PAPI_OK);
   
		retVal = PAPI_OK;	
		if (!numMetrics && !numGroups) {
			fprintf(stderr, "Error on enum_cmp_event, abort.\n");
			retVal = PAPI_ENOEVNT;
		}
		if (listGroupOnly) {
			printf("Total %d metric groups are supported\n", numGroups);
		} else {
			printf("Total %d metrics are supported\n", numMetrics);
		}
		PAPI_shutdown();
		return retVal;
	}

	if (!metricName) {
		metricName = TEST_METRIC;
	}
	printf("Query metric by name %s --  ", metricName);
	
	retVal = PAPI_query_named_event(metricName);
	if (retVal != PAPI_OK) {
		printf("does not exist, abort.\n");
		PAPI_shutdown();
		return retVal;
	} else {
		printf("is supported.\n");
	}

	int code = 0;
	retVal = PAPI_event_name_to_code(metricName, &code);
	if (retVal != PAPI_OK) {
		printf("PAPI_event_name_to_code event %s dose not exist, abort.\n", metricName);
		PAPI_shutdown();
		return retVal;
	}
	printf("Named metric %s is enumerated with code 0x%x.\n", metricName, code);
	printf("Query metric by code 0x%x --  ", code);
	retVal = PAPI_query_event(code);
	if (retVal != PAPI_OK) {
		printf("dose not exist, abort.\n");
		PAPI_shutdown();
		return retVal;
	} else {
		printf("is supported.\n");
	}

	PAPI_shutdown();
	return 0;
}
