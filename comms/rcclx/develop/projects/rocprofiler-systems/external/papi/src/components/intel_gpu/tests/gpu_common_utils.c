/* Copyright (c) 2020 Intel Corp. All rights reserved
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
 */

/*
 *  This utility functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>

#include "papi.h"
#include "gpu_common_utils.h"

#if defined(__cplusplus)
extern "C" {
#endif

const char *default_metrics[] = {
				"ComputeBasic.GpuTime",
				"ComputeBasic.GpuCoreClocks",
				"ComputeBasic.AvgGpuCoreFrequencyMHz",
};

int	num_default_metrics = 3;

void
parseMetricList(char *metric_list, InParams *param) {

	int size	= 64;
	int index   = 0;
	if (!metric_list) {
		param->num_metrics = num_default_metrics;
		param->metric_names = (char **)default_metrics;
	} else {
		char **metrics = (char **)calloc(size, sizeof(char *));
		char *token = strtok(metric_list, ",");
		while (token) {
			if (index >= size) {
				 size += 64;
				 metrics = (char **)realloc(metrics, size * sizeof(char *));
			}
			metrics[index++] = token;
			printf("metric[%d]: %s\n", index-1,  metrics[index-1]);
			token = strtok(NULL, ",");
		}
		param->num_metrics = index;
		param->metric_names = metrics;
	}
}

int
parseInputParam(int argc, char **argv, InParams *param) {
	char ch;
	int duration = 3;
	int loops	= 1;
	int reset	= 0;
	char *metric_list = NULL;
	char *app_targets = NULL;
	while ((ch=getopt(argc, argv, "d:l:e:t:m:s")) != -1) {
		switch(ch) {
			case 't':
				app_targets = optarg;
				break;
			case 'd':
				duration = atoi(optarg);
				if ((duration <= 0) || (duration > 3600)) {  //  max 3600 seconds
					 printf("invalid input on dueation [1, 3600], use default 3 sec.\n");
					 duration = 3;
				}
				break;
			case 'l':
				loops = atoi(optarg);
				if ((loops <= 0) || (loops > 0x100000)) {   // max 1M
					 printf("invalid input on loops [1, 1M], use default 1 loop.\n");
					 loops  = 1;
				}
				break;
			case 's':
				reset = 1;
				break;
			case 'm':
				metric_list = strdup(optarg);
				break;
			default: 
				return 1;
		}
	}
	param->duration = duration;
	param->loops = loops;
	param->reset = reset;
	param->app_dev = 0;
	param->app_tile = 0;
	if (app_targets)  {
		char *str = app_targets;
		int i=0;
		if ((str[i]=='d') && (str[i+1] != '\0')) {
			param->app_dev = atoi(&str[++i]);
			while ((str[i] != '\0') && ( str[i] < '0') && (str[i] > '9')) {
				i++;
			}
		}
		if ((str[i] != '\0') && (str[i] == 't') && (str[i+1] != '\0')) {
			param->app_tile = atoi(&str[i+1])+1;
		} 
	}
	parseMetricList(metric_list, param);
	return 0;
}

int
initPAPIGPUComp() {

	PAPI_component_info_t *aComponent =  NULL;
	int cid = -1;

	// init all components including "intel_gpu"
	int retVal = PAPI_library_init( PAPI_VER_CURRENT );
	if( retVal != PAPI_VER_CURRENT ) {
		fprintf( stderr, "PAPI_library_init failed\n" );
		return -1;
	}

	int numComponents  = PAPI_num_components();
	int i = 0;
	for (i=0; i<numComponents && cid<0; i++) {
		// get the component info.
		aComponent = (PAPI_component_info_t*) PAPI_get_component_info(i);   
		if (aComponent == NULL) {
			continue;
		}
		if (strcmp(COMP_NAME, aComponent->name) == 0) {
			cid=i;				// If we found our match, record it.
		} // end search components.
	}
	if (cid < 0) {
		fprintf(stderr, "Failed to find component [%s] in total %i supported components.\n", 
			COMP_NAME, numComponents);
		 PAPI_shutdown();
		 return -1;
	}
	return cid;
}

int
initMetricSet(char **metric_names, int num_metrics, int *eventSet) {
 
	int retVal = PAPI_create_eventset(eventSet);
	if (retVal != PAPI_OK) {
		fprintf(stderr, "Error on PAPI_create_eventset, retVal %d\n", retVal);
		PAPI_shutdown();
		return retVal;
	}

	for (int i=0; i<num_metrics; i++) {
		retVal = PAPI_add_named_event(*eventSet, metric_names[i]);
		if (retVal < 0) {
			fprintf(stderr, "Error on PAPI_add_named_event %s,  retVal %d\n", 
					metric_names[i], retVal);
			break;
		}
	}
	if (retVal != PAPI_OK) {
		PAPI_shutdown();
		return retVal;
	}
	return PAPI_OK;
}

#if defined(__cplusplus)
}
#endif

