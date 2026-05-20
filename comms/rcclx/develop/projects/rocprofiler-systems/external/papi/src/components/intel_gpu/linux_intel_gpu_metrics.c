/*
 * linux_intel_gpu_metrics.c:  Intel® Graphics Processing Unit (GPU) Component for PAPI.
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>

#include "inc/GPUMetricInterface.h"
#include "linux_intel_gpu_metrics.h"


#define DEFAULT_LOOPS	 0				// collection infinitely until receive stop command
#define DEFAULT_MODE	  METRIC_SUMMARY   // summary report

#define GPUDEBUG	 SUBDBG

// all available devices
static uint32_t num_avail_devices = 0;
static DEVICE_HANDLE *avail_devices;

// devices currently being queried
static uint32_t num_active_devices = 0;
DeviceContext *active_devices;

static MetricInfo	metricInfoList;
static int		   total_metrics		= 0;		  
static int		   global_metrics_type  = TIME_BASED; // default type

papi_vector_t _intel_gpu_vector;

/*!
 * @brief Parser a metric name with qualifiers
 *		  metricName can be
 *			  component:::metrcGroup.metricname:device=xx:tile=yy
 *		  or  metrcGroup.metricname:device=xx:tile=yy
 *  @param   IN   metricName	-- metric name in the above format
 *  @param   OUT  devnum		-- device id
 *  @param   OUT  tilenum	    -- tile id
 */
void
parseMetricName(const char *metricName, int *devnum, int *tilenum) {
	char	 *name	  = strdup(metricName);
	char	 *ptr	  = name;
	char	 *param	  = name;
	uint32_t  dnum	  = 0;
	uint32_t  tnum	  = 0;

	if ((*param != '\0') && (ptr=strstr(param, ":"))) {
		if ((ptr=strstr(param, ":device="))) {
			ptr += 8;
			dnum = atoi(ptr)+1;
		}
		if ((ptr=strstr(param, ":tile="))) {
			ptr += 6;
			tnum = atoi(ptr)+1;
		}
	}
	*devnum = (dnum)?dnum:1;
	*tilenum = (tnum)?tnum:0;  // default tile is 0 for root device
	free(name);
}


/*!
 * @brief   Get handle from device code
 */
DEVICE_HANDLE
getHandle(uint32_t device_code) {
	uint32_t i=0;
	for (i=0; i<num_avail_devices; i++) {
		if (IsDeviceHandle(device_code, avail_devices[i])) {
		   return avail_devices[i];
		}
	}
	return 0;
}

/*!
 * @brief  Add a metrics to a certain metric device
 */
int
addMetricToDevice(uint32_t code, int rootDev) {

	int index = GetIdx(code);
	if  (index >= total_metrics) {
		return -1;
	}
	uint32_t group  = GetGroupIdx(metricInfoList.infoEntries[index].code);
	uint32_t metric = GetMetricIdx(metricInfoList.infoEntries[index].code);
	uint32_t devcode = GetDeviceCode(code);
	if (rootDev) {
		devcode = devcode & ~DMASK;
	}
	GPUDEBUG("addMetricToDevice, code 0x%x, group 0x%x,  metric 0x%x, devcode 0x%x\n",
		  code, group, metric, devcode);
	uint32_t i=0;
	for (i=0; i<num_active_devices; i++)  {
		if  (active_devices[i].device_code == devcode) {
			break;
		}
	}
	if (i >= num_avail_devices) {
		GPUDEBUG("intel_gpu: invalid event code 0x%x\n", code);
		return -1;
	}

	DEVICE_HANDLE hd = getHandle(devcode);
	if (!hd) {
		GPUDEBUG("intel_gpu: Metric is not supported. For multi devices and/or multi-tiles GPUs, "
				"metric name should be qualified with :device=0 and/or :tile=0. "
				"Failed with return code 0x%x \n", PAPI_ENOSUPP);
		return -1;
	}

	// add a new device entry
	DeviceContext *dev = &(active_devices[i]);
	if (i==num_active_devices) {
		dev->device_code = devcode;
		dev->mgroup_code = group;
		dev->handle = hd;
		num_active_devices++;
	}
	// add a new metric to collect on this device
	dev->metric_code[dev->num_metrics++] = metric;
	return i;
}

/*!
 * @brief  Reset metric counts to zero.
 */
void
metricReset( MetricContext *mContext )
{
    for (uint32_t i=0; i<num_avail_devices; i++) {
        uint32_t dev_idx = mContext->active_devices[i];
        if  (dev_idx < num_active_devices) {
            DeviceContext *dev = &active_devices[dev_idx];
            GPUSetMetricControl(dev->handle, METRIC_RESET);
        }
    }
}



/************************* PAPI Functions **********************************/

static int
intel_gpu_init_thread(hwd_context_t *ctx)
{
	GPUDEBUG("Entering intel_gpu_init_thread\n");
	MetricContext *mContext = (MetricContext *)ctx;
	mContext->active_devices = calloc(num_avail_devices, sizeof(uint32_t));
	return PAPI_OK;
}


static int 
intel_gpu_init_component(int cidx)
{
	int retval = PAPI_OK;
	GPUDEBUG("Entering intel_init_component\n");
	if (cidx < 0) {
		return  PAPI_EINVAL;
	}
	char *errStr = NULL;
	memset(_intel_gpu_vector.cmp_info.disabled_reason, 0, PAPI_MAX_STR_LEN);

	if (putenv("ZET_ENABLE_METRICS=1")) {
		errStr = "Set ZET_ENABLE_METRICS=1 failed. Cannot access GPU metrics. ";
		strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
				   errStr, strlen(errStr));
		retval = PAPI_ENOSUPP;
		goto fn_fail;
	}
	if ( _dl_non_dynamic_init != NULL ) {
		errStr = "The intel_gpu component does not support statically linking of libc.";
		strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
				   errStr, strlen(errStr));
		retval = PAPI_ENOSUPP;
		goto fn_fail;
	}
	if (GPUDetectDevice(&avail_devices, &num_avail_devices) || (num_avail_devices==0)) {
		errStr = "The intel_gpu component does not detect metrics device.";
		strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
				   errStr, strlen(errStr));
		retval = PAPI_ENOSUPP;
		goto fn_fail;
	}

	DEVICE_HANDLE handle = avail_devices[0];

	char *envStr = NULL;
	envStr =  getenv(ENABLE_API_TRACING);
	if (envStr != NULL) {
	   if (atoi(envStr) == 1) {
		   global_metrics_type = EVENT_BASED;
	   }
	}

	metricInfoList.numEntries = 0;
	if (GPUGetMetricList(handle, "", global_metrics_type, &metricInfoList)) {
		errStr = "The intel_gpu component failed on get all available metrics.";
		strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
				   errStr, strlen(errStr));
		GPUFreeDevice(handle);
		retval = PAPI_ENOSUPP;
		goto fn_fail;
	}; 
	total_metrics = metricInfoList.numEntries;
	GPUDEBUG("total metrics %d\n", total_metrics);

	_intel_gpu_vector.cmp_info.num_native_events = metricInfoList.numEntries;
	_intel_gpu_vector.cmp_info.num_cntrs		 = GPU_MAX_COUNTERS;
	_intel_gpu_vector.cmp_info.num_mpx_cntrs	 = GPU_MAX_COUNTERS;

	/* Export the component id */
	_intel_gpu_vector.cmp_info.CmpIdx = cidx;

	active_devices = calloc(num_avail_devices, sizeof(DeviceContext));
	num_active_devices = 0;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}

/*!
 * @brief  Setup a counter control state.
 *		 In general a control state holds the hardware info for an EventSet.
 */
static int
intel_gpu_init_control_state( hwd_control_state_t * ctl )
{
	GPUDEBUG("Entering intel_gpu_control_state\n");

	if (!ctl) {
		return PAPI_EINVAL;
	}

	MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
	mCtlSt->metrics_type  = global_metrics_type;

	char *envStr = NULL;
	mCtlSt->interval = DEFAULT_SAMPLING_PERIOD;
	envStr =  getenv(METRICS_SAMPLING_PERIOD);
	if (envStr) {
		mCtlSt->interval = atoi(envStr);
		if (mCtlSt->interval < MINIMUM_SAMPLING_PERIOD) {
			mCtlSt->interval = DEFAULT_SAMPLING_PERIOD;
		}
	}
	// set default mode  (get aggragated value or average value overtime).
	mCtlSt->mode	 = DEFAULT_MODE;
	mCtlSt->loops	= DEFAULT_LOOPS;

	return PAPI_OK;
}

static int
intel_gpu_update_control_state( hwd_control_state_t *ctl,
						 NativeInfo_t *native,
						 int count,
						 hwd_context_t *ctx )
{
	GPUDEBUG("Entering intel_gpu_control_state\n");
	(void)ctl;

	// use local maintained context,
	MetricContext *mContext = (MetricContext *)ctx;

    /* This check accounts for calls to PAPI_cleanup_eventset(). */
	if ( !count )  {
	    metricReset(mContext);
        /* Free devices. */
        for (uint32_t i=0; i<mContext->num_devices; i++) {
            uint32_t dev_idx = mContext->active_devices[i];
            if (dev_idx >= num_active_devices) {
                return PAPI_ENOMEM;
            }
            DeviceContext *dev = &active_devices[dev_idx];
            DEVICE_HANDLE handle = dev->handle;
            if( !handle ) {
                GPUFreeDevice(handle);
            }
            mContext->active_devices[i] = 0;
        }
        mContext->num_devices = 0;
        /* Free metric slots. */
        for (uint32_t midx=0; midx<mContext->num_metrics; midx++) {
            mContext->metric_idx[midx] = 0;
            mContext->metric_values[midx] = 0;
            mContext->dev_ctx_idx[midx] = 0;
            mContext->subdev_idx[midx] = 0;
        }
        mContext->num_metrics = 0;
        return PAPI_OK;
	}

	if ( !native ) {
		return PAPI_OK;
	}

#if defined(_DEBUG)
	for (int i=0; i<count; i++) {
		 GPUDEBUG("\t i=%d, ni_event 0x%x,  ni_papi_code 0x%x, ni_position %d, ni_owners %d   \n", 
				i, native[i].ni_event, native[i].ni_papi_code, 
				native[i].ni_position, native[i].ni_owners);
	}
#endif

	uint32_t nmetrics = mContext->num_metrics;
	uint32_t midx = 0;
	int ni = 0;
	for (ni = 0; ni < count; ni++) {
	   uint32_t index = native[ni].ni_event;
	   // check whether this metric is in the list
	   for (midx=0; midx<nmetrics; midx++) {
		   if (mContext->metric_idx[midx] == index) {
			   GPUDEBUG("metric code %d: already in the list, ignore\n", index); 
			   break;
		   }
	   }
	   if (midx < nmetrics) {
			//  already in the list
			continue;
	   }
	   // whether use root device or subdevice
	   char *envStr = NULL;
	   int useRootDevice = 1;
	   envStr =  getenv(ENABLE_SUB_DEVICE);
	   if (envStr != NULL) {
		   useRootDevice = 0;
	   }
	   int idx = addMetricToDevice(index, useRootDevice);
	   if (idx<0) {
		   return PAPI_ENOSUPP;
	   }
	   mContext->metric_idx[nmetrics] = index;
	   mContext->dev_ctx_idx[nmetrics] = idx;
	   mContext->subdev_idx[nmetrics] = GetSDev(index);
	   GPUDEBUG("add metric[%d] code 0x%x, in device[%d] (event subdev[%d])\n",
		  nmetrics, mContext->metric_idx[nmetrics],
		  mContext->dev_ctx_idx[nmetrics], mContext->subdev_idx[nmetrics]);

	   uint32_t i = 0;
	   for (i=0; i<mContext->num_devices; i++) {
		   if (mContext->active_devices[i] == (uint32_t)idx) {
				// already in the list
				break;
		   }
	   }
	   if (i == mContext->num_devices) {
		   mContext->active_devices[i] = idx;
		   mContext->num_devices++;
	   }
	   native[ni].ni_position = nmetrics;
	   nmetrics++;
	}
	// add this metric
	mContext->num_metrics = nmetrics;
	return PAPI_OK;
}


static int 
intel_gpu_start( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	GPUDEBUG("Entering intel_gpu_start\n");

	MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
	MetricContext *mContext = (MetricContext *)ctx;

	int	 ret	= PAPI_OK;

    metricReset(mContext);

	if (mContext->num_metrics == 0) {
		GPUDEBUG("intel_gpu_start : No metric selected, abort.\n");
		return PAPI_EINVAL;
	}

	char **metrics = calloc(mContext->num_metrics, sizeof(char *));
	if (!metrics) {
		GPUDEBUG("intel_gpu_start : insufficient memory, abort.\n");
		return PAPI_ENOMEM;
	}

	mContext->num_reports = 0;

	for (uint32_t i=0; i<mContext->num_devices; i++) {
		uint32_t dev_idx = mContext->active_devices[i];
		if (dev_idx >= num_active_devices) {
			ret = PAPI_ENOMEM;
			break;
		}
		DeviceContext *dev = &active_devices[dev_idx];
		DEVICE_HANDLE handle = dev->handle;
		if (GPUEnableMetricGroup(handle, "", dev->mgroup_code,
				mCtlSt->metrics_type, mCtlSt->interval, mCtlSt->loops)) {
			GPUDEBUG("intel_gpu_start on EnableMetrics failed, return 0x%x \n", PAPI_ENOSUPP);
			ret = PAPI_ENOMEM;
			break;
		}
	}
	return ret;
}

static int 
intel_gpu_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
	GPUDEBUG("Entering intel_gpu_stop\n");

	MetricContext *mContext = (MetricContext *)ctx;
	MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
	int			 ret	= PAPI_OK;

	for (uint32_t i=0; i<mContext->num_devices; i++) {
		uint32_t dev_idx = mContext->active_devices[i];
		if  (dev_idx < num_active_devices) {
			DeviceContext *dev = &active_devices[dev_idx];
			DEVICE_HANDLE handle = dev->handle;
			if (GPUDisableMetricGroup(handle, mCtlSt->metrics_type)) {
				GPUDEBUG("intel_gpu_stop : failed with ret %d\n", ret);
				ret = PAPI_EINVAL;
			}
		}
	}
	return ret;
}

static int 
intel_gpu_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags )
{
	GPUDEBUG("Entering intel_gpu_read\n");

	(void)flags;

	MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
	MetricContext *mContext = (MetricContext *)ctx;
	MetricData *reports = NULL;
	uint32_t numReports = 0;
	int	ret = PAPI_OK;

	if (!events) {
		return PAPI_EINVAL;
	}

	if (mContext->num_metrics == 0) {
		GPUDEBUG("intel_gpu_read: no metric is selected\n");
		return PAPI_OK;
	}
	for (uint32_t i=0; i<mContext->num_devices; i++) {
		uint32_t dc_idx = mContext->active_devices[i];
		numReports = 0;
		if  (dc_idx >= num_active_devices) {
			 continue;
		}

		DeviceContext *dc = &active_devices[dc_idx];
		DEVICE_HANDLE handle = dc->handle;
		reports = GPUReadMetricData(handle, mCtlSt->mode, &numReports);

		if (!reports) {
			 GPUDEBUG("intel_gpu_read failed on device 0x%x\n", GetDeviceCode(handle));
			 continue;
		} else if (!numReports) {
			 GPUFreeMetricData(reports, numReports);
			 GPUDEBUG("intel_gpu_read: no data available on device 0x%x\n", GetDeviceCode(handle));
			 continue;
		}

		if (dc->data) {
			GPUFreeMetricData(dc->data, dc->num_reports);
		}
		/* take last report, it is expected the numReports is 1 */
		dc->data = &reports[numReports-1];
		dc->num_reports = numReports;
	}
	for (uint32_t i=0; i<mContext->num_metrics; i++) {
		uint32_t dc_idx = mContext->dev_ctx_idx[i];
		DeviceContext *dc = &(active_devices[dc_idx]);
		int index = GetIdx(mContext->metric_idx[i]);
		int start_idx = 0;
		if (!(dc->device_code &DMASK)) {  // root device
			uint32_t dc_sidx = GetSDev(mContext->metric_idx[i]);
			if (dc_sidx > 0) {
				dc_sidx--;  // zero index
			}
			if (dc_sidx < dc->data->numDataSets) {
				start_idx = dc->data->dataSetStartIdx[dc_sidx];
			} else {
			   start_idx = -1;
			}
		}
		if ((start_idx < 0) || !dc->data  || !dc->num_reports) {
			mContext->metric_values[i] = 0;   // no data available
		} else {
			uint32_t midx = GetMetricIdx(metricInfoList.infoEntries[index].code)-1;
			if (!dc->data->dataEntries[midx].type) {
				mContext->metric_values[i] =
			 		(long long)dc->data->dataEntries[start_idx + midx].value.ival;
			} else {
				mContext->metric_values[i] =
			 		(long long)dc->data->dataEntries[start_idx + midx].value.fpval;
			}
		}
	}

	mContext->num_reports = numReports;
	*events = mContext->metric_values;
	return ret;
}

static int
intel_gpu_shutdown_thread( hwd_context_t *ctx )
{
	(void)ctx;
	GPUDEBUG("Entering intel_gpu_shutdown_thread\n" );
	return PAPI_OK;
}

static int
intel_gpu_shutdown_component(void)
{
	GPUDEBUG("Entering intel_gpu_shutdown_component\n");
	for (uint32_t i=0; i<num_avail_devices; i++) {
		DEVICE_HANDLE handle = avail_devices[i];
		GPUFreeDevice(handle);
	}
	return PAPI_OK;
}

/* 
 * reset function will reset the global accumualted metrics values
 */
static int 
intel_gpu_reset( hwd_context_t *ctx, hwd_control_state_t *ctl)
{

	(void)ctl;
	GPUDEBUG("Entering intel_gpu_reset\n");
	MetricContext *mContext = (MetricContext *)ctx;

	metricReset(mContext);

	return PAPI_OK;
}

static int
intel_gpu_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
	GPUDEBUG("Entering intel_gpu_ctl\n");

	(void) ctx;
	(void) code;
	(void) option;
	return PAPI_OK;
}


static int
intel_gpu_set_domain( hwd_control_state_t * ctl, int domain )
{
	GPUDEBUG("Entering intel_gpu_set_domain\n");

	if (!ctl) {
		return PAPI_EINVAL;
	}
	MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
	mCtlSt->domain = domain;
	return PAPI_OK;

}

static int 
intel_gpu_ntv_enum_events(uint32_t *EventCode, int modifier )
{
	GPUDEBUG("Entering intel_gpu_ntv_enum_events\n");
	int index = 0;

	if (!EventCode)  {
		return PAPI_EINVAL;
	}
	switch ( modifier ) {
		case PAPI_ENUM_FIRST:
			*EventCode = 0;
			break;
		case PAPI_ENUM_EVENTS:
			index = GetIdx(*EventCode);
			uint32_t dev_code = GetDeviceCode(*EventCode);
			if ( (index < 0 ) || (index >= (total_metrics-1)) ) {
				return PAPI_ENOEVNT;
			}
			*EventCode = CreateIdxCode(dev_code, index+1);
			break;
		default:
			return PAPI_EINVAL;
	} 
	return PAPI_OK;
}

static int 
intel_gpu_ntv_code_to_name( uint32_t EventCode, char *name, int len )
{
	GPUDEBUG("Entering intel_gpu_ntv_code_to_name\n");
	int  index = GetIdx(EventCode);

	if( ( index < 0 ) || ( index >= total_metrics ) || !name || !len ) {
		return PAPI_EINVAL;
	}

	memset(name, 0, len);
	strncpy_se(name,  len, metricInfoList.infoEntries[index].name, 
		   strlen(metricInfoList.infoEntries[index].name));
	return PAPI_OK;
}

static int 
intel_gpu_ntv_code_to_descr( uint32_t EventCode, char *desc, int len )
{
	GPUDEBUG("Entering intel_gpu_ntv_code_to_descr\n");
	int  index = GetIdx(EventCode);

	if( ( index < 0 ) || ( index >= total_metrics ) || !desc || !len) {
		return PAPI_EINVAL;
	}

	memset(desc, 0, len);
	strncpy_se(desc, len, metricInfoList.infoEntries[index].desc, MAX_STR_LEN-1);
	return PAPI_OK;
}

static int
intel_gpu_ntv_name_to_code( const char *name, uint32_t *event_code)
{
	GPUDEBUG("Entering intel_gpu_ntv_name_to_code\n");
	if( !name || !event_code) {
		return PAPI_EINVAL;
	}

   for (int i=0; i<total_metrics; i++) {
		if (strncmp(metricInfoList.infoEntries[i].name, name, 
			 strlen(metricInfoList.infoEntries[i].name)) == 0) {
		   int devnum   = 0;
		   int tilenum = 0;
		   parseMetricName(name, &devnum, &tilenum);
		   uint32_t code = CreateDeviceCode(1, devnum, tilenum);
		   *event_code = CreateIdxCode(code, i);
		   return PAPI_OK;
		}
	}
	return PAPI_ENOEVNT;
}

static int
intel_gpu_ntv_code_to_info(uint32_t EventCode, PAPI_event_info_t *info)
{
	GPUDEBUG("Entering intel_gpu_ntv_code_to_info\n");
	int  index = GetIdx(EventCode);
	if( ( index < 0 ) || ( index >= total_metrics ) || !info) {
		return PAPI_EINVAL;
	}

	info->event_code = EventCode;
	strncpy_se(info->symbol, PAPI_HUGE_STR_LEN,
			metricInfoList.infoEntries[index].name, MAX_STR_LEN-1);
	// short description could be truncated due to the longer string size of metric name
	strncpy_se(info->short_descr, PAPI_MIN_STR_LEN,
		   metricInfoList.infoEntries[index].name,
		   strlen(metricInfoList.infoEntries[index].name));
	strncpy_se(info->long_descr, PAPI_HUGE_STR_LEN,
		   metricInfoList.infoEntries[index].desc,  MAX_STR_LEN-1);
	info->component_index = _intel_gpu_vector.cmp_info.CmpIdx;
	return PAPI_OK;
}

/* Our component vector */

papi_vector_t _intel_gpu_vector = {
	.cmp_info = {
		/* component information (unspecified values initialized to 0) */
		.name = "intel_gpu",
		.short_name = "intel_gpu",
		.version = "1.0",
		.description = "Intel GPU performance metrics",
		.default_domain = PAPI_DOM_ALL,
		.available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
		.default_granularity = PAPI_GRN_SYS,
		.available_granularities = PAPI_GRN_SYS,
		.num_mpx_cntrs = GPU_MAX_COUNTERS,
		
		/* component specific cmp_info initializations */
		.fast_virtual_timer = 0,
		.attach = 0,
		.attach_must_ptrace = 0,
		.cpu = 0,
		.inherit = 0,
		.cntr_umasks = 0,
	},

	/* sizes of framework-opaque component-private structures */
	.size = {
		.context = sizeof (MetricContext),
		.control_state = sizeof (MetricCtlState),
		.reg_value = sizeof ( int ),
		.reg_alloc = sizeof ( int ),
	},

	/* function pointers in this component */
	.init_thread = intel_gpu_init_thread,
	.init_component	= intel_gpu_init_component,
	.init_control_state	= intel_gpu_init_control_state,
	.update_control_state = intel_gpu_update_control_state,
	.start = intel_gpu_start,
	.stop =	intel_gpu_stop,
	.read =	intel_gpu_read,
	.shutdown_thread = intel_gpu_shutdown_thread,
	.shutdown_component	= intel_gpu_shutdown_component,
	.ctl = intel_gpu_ctl,

	/* these are dummy implementation */
	.set_domain = intel_gpu_set_domain,
	.reset = intel_gpu_reset,

	/* from counter name mapper */
	.ntv_enum_events = intel_gpu_ntv_enum_events,
	.ntv_code_to_name = intel_gpu_ntv_code_to_name,
	.ntv_name_to_code = intel_gpu_ntv_name_to_code,
	.ntv_code_to_descr = intel_gpu_ntv_code_to_descr,
	.ntv_code_to_info = intel_gpu_ntv_code_to_info,
};

