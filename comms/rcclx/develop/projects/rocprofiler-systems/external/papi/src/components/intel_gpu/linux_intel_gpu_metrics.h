/*
 * linux_intel_gpu_metrics.h:  Intel® Graphics Processing Unit (GPU) Component for PAPI.
 *
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
 */

#ifndef _INTEL_GPU_METRICS_H
#define _INTEL_GPU_METRICS_H

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/* environment varaiable for changing the control */
#define METRICS_SAMPLING_PERIOD   "METRICS_SAMPLING_PERIOD"	   // setting sampling period
#define ENABLE_API_TRACING		"ZET_ENABLE_API_TRACING_EXP"	// for oneAPI Level0 V1.0 +
#define ENABLE_SUB_DEVICE		 "ENABLE_SUB_DEVICE"

#define MINIMUM_SAMPLING_PERIOD  100000
#define DEFAULT_SAMPLING_PERIOD  400000

#define GPU_MAX_COUNTERS  54
#define GPU_MAX_METRICS	  128

void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));


typedef struct _metric_ctl_s {
	uint32_t	 interval;
	uint32_t	 metrics_type;
	int		  mode;
	uint32_t	 loops;
	int		  domain;
} MetricCtlState;


typedef struct _device_context_s {
	uint32_t	 device_code;
	uint32_t	 mgroup_code;
	uint32_t	 num_metrics;
	uint32_t	 metric_code[GPU_MAX_METRICS];
	DEVICE_HANDLE handle;
	uint32_t	 num_reports;
	uint32_t	 num_data_sets;
	uint32_t	*data_set_sidx;
	uint32_t	 data_size;
	MetricData  *data;
} DeviceContext;

typedef struct _metric_context_s {
	int		  cmp_id;
	int		  device_id;
	int		  domain;
	int		  thread_id;
	int		  data_avail;
	uint32_t	 num_metrics;
	uint32_t	 num_reports;
	uint32_t	 num_devices;
	uint32_t	*active_sub_devices;
	uint32_t	*active_devices;
	uint32_t	 metric_idx[GPU_MAX_METRICS];
	uint32_t	 dev_ctx_idx[GPU_MAX_METRICS];
	uint32_t	 subdev_idx[GPU_MAX_METRICS];
	long long	metric_values[GPU_MAX_METRICS];
} MetricContext;



#endif /* _INTEL_GPU_METRICS_H */

