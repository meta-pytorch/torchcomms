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
 *  This test case tests time based data collection on ntel GPU performance metrics
 *
 *  @ brief Collect  metric data for a certain time interval, with one or more loops.
 *          By default, the metric data will aggregate overtime for each loop.
 *          When reset, each group reports metric data only for the sepecified time duration
 *
 *  @option:
 *     [-d <time interval in second>][-l <number of loops>] [-s (reset)] [-m <metric list>]
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"

#include "gpu_common_utils.h"

/*
 * This test case tests:
 *  PAPI_start(),  PAPI_read(), PAPI_reset(),  PAPI_stop()
 */

int
main(int argc, char ** argv) {

    int i = 0;
    int retVal = 0;
    int cid = -1;
    int total_metrics = 0;
    int event_set = PAPI_NULL;
    PAPI_event_info_t info;
    InParams  param;

    if (argc < 2) {
        printf("usage: %s -d <duration> [ -l <loops>][-s][-m metric[:device=0][:tile=0]]\n", argv[0]);
        return 0;
    }
    // unset variable ZET_ENABLE_API_TRACING_EXP to enable time base collection
    retVal = putenv("ZET_ENABLE_API_TRACING_EXP=0");
    retVal = parseInputParam(argc, argv, &param);
    if (retVal) {
        printf("Invalid input parameters.\n");
        printf("usage: %s -d <duration> [ -l <loops>][-s][-m metric[:device=0][:tile=0]]\n", argv[0]);
		return 0;
    }
    int num_metrics = param.num_metrics;
    char **metric_names = (char **)(param.metric_names);

    cid = initPAPIGPUComp();
    if (cid < 0) {
        return 1;
    }

    i = 0 | PAPI_NATIVE_MASK; 
    retVal=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on enum_cmp_event for component[ %s ],  abort.\n", COMP_NAME);
        PAPI_shutdown();
        return 1;
    }
    total_metrics = 0;
    do {
        memset( &info, 0, sizeof ( info ) );
        retVal =  PAPI_get_event_info( i, &info );
        if (retVal == PAPI_OK)  {
            total_metrics++;
            retVal = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid );
        }
    } while (retVal == PAPI_OK);

    if ((!total_metrics)) {
        fprintf(stderr, "Error on enum_cmp_event, abort.\n");
        PAPI_shutdown();
        return 1;
    }
    retVal = initMetricSet(metric_names, num_metrics, &event_set);
    if (retVal != PAPI_OK) {
        PAPI_shutdown();
        return 1;
    }
    long long *metric_values = (long long *)calloc(num_metrics, sizeof(long long));
    if (!metric_values) {
        fprintf(stderr, "Memory allocation failed, abort.\n");
        PAPI_shutdown();
        return 1;
    }
    retVal = PAPI_start(event_set);
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on PAPI_start, retVal %d\n", retVal);
        free(metric_values);
        PAPI_shutdown();
        return 1;
    }
    //some work here
    sleep(param.duration);
    param.loops--;
    for (uint32_t i=0; i<param.loops; i++) {
        PAPI_read(event_set, metric_values);
        for (int j=0; j<num_metrics; j++) {
            printf("%-50s ......  %llu\n", metric_names[j], metric_values[j]);
        }
        printf("======\n");
        if (param.reset) {
            PAPI_reset(event_set);
        }
        //some work here
        sleep(param.duration);
    }
    retVal = PAPI_stop(event_set, metric_values);
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on PAPI_stop, retVal %d\n", retVal);
        free(metric_values);
        PAPI_shutdown();
        return 1;
    }
    for (int i=0; i<num_metrics; i++) {
        printf("%-50s ......  %llu\n", metric_names[i], metric_values[i]);
    }
    free(metric_values);
    PAPI_shutdown();
    return 0;
}
