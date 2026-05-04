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
 *          When reset, each group reports metric data only for the sepecified tiem duration
 *
 *  @option:
 *     [-d <time interval in second>][-l <number of loops>] [-s (reset)]
 */

#ifndef _GPU_COMMON_UTILS_H
#define _GPU_COMMON_UTILS_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define MAX_NUM_METRICS    40
#define MAX_STRLEN         128 

#define COMP_NAME       "intel_gpu"

typedef struct _inParams {
    uint32_t duration;
    uint32_t loops;
    uint32_t reset;
    uint32_t app_dev;
    uint32_t app_tile;
    uint32_t num_metrics;
    char **metric_names;
} InParams;


void parseMetricList(char *metric_list,  InParams *param);
int parseInputParam(int argc, char **argv, InParams *param);
int initPAPIGPUComp();
int initMetricSet(char **metric_names, int num_metrics, int *eventSet);


#if defined(__cplusplus)
}
#endif

#endif /* _GPU_COMMON_UTILS_H */

