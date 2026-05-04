/*
 * GPUMetricInterface.h:  Intel® Graphics Component for PAPI
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

#ifndef _GPUMETRICINTERFACE_H
#define _GPUMETRICINTERFACE_H

#include <stdio.h>
#include <stdint.h>

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef MAX_STR_LEN
#define MAX_STR_LEN	  256
#endif

//metric group type
#define TIME_BASED   0
#define EVENT_BASED  1

//metric type
#define M_ACCUMULATE  0x0
#define M_AVERAGE	 0x1
#define M_RAW		 0x2

typedef int DEVICE_HANDLE;

typedef char NAMESTR[MAX_STR_LEN];

typedef struct MetricInfo_S MetricInfo;

struct MetricInfo_S {
	uint32_t code;
	uint32_t dataType;
	uint32_t metricType;		// 0 accumulate  1 average, 2 raw (static)
	char name[MAX_STR_LEN];
	char desc[MAX_STR_LEN];
	int numEntries;
	MetricInfo *infoEntries;
};

/* define GPU device info   */
typedef struct DeviceInfo_S {
	DEVICE_HANDLE handle;
	uint32_t	  driverId;
	uint32_t	  deviceId;
	uint32_t	  subdeviceId;
	uint32_t	  numSubdevices;
	uint32_t	  index;
	char		  name[MAX_STR_LEN];
	MetricInfo   *metricGroups;
} DeviceInfo;

typedef struct DataEntry_S {
	uint32_t code;
	uint32_t type;
	union {
	   long long ival;
	   double	fpval;
	} value;
} DataEntry;

/* Metric data for each device
 * Data can contain multiple set, one set per subdevice
 */
typedef struct MetricData_S {
	int		 id;
	uint32_t	grpCode; 
	uint32_t	numDataSets;		 // num of data set per device
	uint32_t	metricCount;		 // metrics per data set
	uint32_t	numEntries;		  // total data entries allocated
	int		*dataSetStartIdx;	 // start index per each data set, -1 mean no data available
	DataEntry  *dataEntries;
}  MetricData;

typedef struct MetricNode_S {
	char	 name[MAX_STR_LEN];
	int	  code;
} MetricNode;


/* index code for a handle
 * resv(4) + drv (4) + dev(4) + subdev(4) +  index(16)
 */
#define DRV_BITS	4
#define DEV_BITS	4
#define SDEV_BITS	4
#define IDX_BITS	16
#define DMASK		0xf
#define DCODE_MASK	0xfff
#define IDX_MASK	0xffff

#define CreateDeviceCode(drv,  dev, sdev)							\
				  ((((drv)&DMASK)<<(SDEV_BITS+DEV_BITS)) |		  	\
		  (((dev)&DMASK)<<SDEV_BITS) |					   			\
		  ((sdev)&DMASK))

/* this macro is used for creating index code for device and event.
 * The idx can be device index or event idx
 */
#define CreateIdxCode(devcode, idx)	 ((((devcode)&DCODE_MASK)<<IDX_BITS)|((idx)&IDX_MASK))

#define GetDeviceCode(handle)	   (((handle) >> IDX_BITS) & DCODE_MASK)
#define GetIdx(handle)			  ((handle) & IDX_MASK)
#define GetSDev(handle)			 (((handle) >> IDX_BITS) & DMASK)
#define GetDev(handle)			  (((handle) >> (IDX_BITS+SDEV_BITS)) & DMASK)
#define GetDrv(handle)			  (((handle) >> (IDX_BITS+SDEV_BITS+DEV_BITS)) & DMASK)
#define IsDeviceHandle(devcode, handle)	((devcode) == (((handle)>>IDX_BITS)&DCODE_MASK))

/*
 * metric code:  group(8) + metrics(8)
 */
#define METRIC_BITS		  8
#define METRIC_GROUP_MASK	0xfff00
#define METRIC_MASK		  0x00ff

#define CreateGroupCode(mGroupId)		 (((mGroupId+1)<<METRIC_BITS) & METRIC_GROUP_MASK)
#define CreateMetricCode(mGroupId, mId)   ((CreateGroupCode(mGroupId)) | ((mId+1)&METRIC_MASK))
#define GetGroupIdx(code)				 (((code) & METRIC_GROUP_MASK) >> METRIC_BITS)
#define GetMetricIdx(code)				((code) & METRIC_MASK)


#define MAX_METRICS	   128

#define MAX_NUM_REPORTS   20

/* collection modes */
#define METRIC_SAMPLE	 0x1
#define METRIC_SUMMARY	0x2
#define METRIC_RESET	  0x4

MetricInfo *GPUAllocMetricInfo(uint32_t count);
void GPUFreeMetricInfo(MetricInfo *info, uint32_t count);
MetricData *GPUAllocMetricData(uint32_t count, uint32_t numSets, uint32_t numMetrics);
void GPUFreeMetricData(MetricData *data, uint32_t count);

extern void
strncpy_se(char *dest, size_t destSize,  char *src, size_t count);


/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUDetectDevice(DEVICE_HANDLE **handle, uint32_t *numDevice);
 *
 * @brief	   Detect the named device which has performance monitoring feature availale.
 *
 * @param	   OUT  handle	- a array of handle, each for an instance of the device
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUDetectDevice(DEVICE_HANDLE **handle, uint32_t *numDevice);


/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUGetDeviceInfo(DEVICE_HANDLE handle, MetricInfo *data);
 *
 * @brief	   Get the device properties, which is mainly in <name, value> format
 *
 * @param	   IN   handle - handle to the selected device
 * @param	   OUT  data   - the property data
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUGetDeviceInfo(DEVICE_HANDLE handle, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUPrintDeviceInfo(DEVICE_HANDLE handle, FILE *stream);
 *
 * @brief	   Get the device properties, which is mainly in <name, value> format
 *
 * @param	   IN   handle - handle to the selected device
 * @param	   IN   stream   - IO stream to print
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUPrintDeviceInfo(DEVICE_HANDLE handle, FILE *stream);


/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUGetMetricGroups(DEVICE_HANDLE handle, uint32_t mtype, MetricInfo *data);
 *
 * @brief	   list available metric groups for the selected type
 *
 * @param	   IN   handle - handle to the selected device
 * @param	   IN   mtype  - metric group type
 * @param	   OUT  data   - metric data
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUGetMetricGroups(DEVICE_HANDLE handle, uint32_t mtype, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUGetMetricList(DEVICE_HANDLE handle, 
 *								   char *groupName, uint32_t mtype, MetricInfo *data);
 *
 * @brief	   list available metrics in the named group.
 *			  If name is "", list all available metrics in all groups
 *
 * @param	   IN   handle	  - handle to the selected device
 * @param	   IN   groupName   - metric group name. "" means all groups.
 * @param	   In   mtype	   - metric type <TIME_BASED, EVENT_BASED>
 * @param	   OUT  data   - metric data
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUGetMetricList(DEVICE_HANDLE handle, char *groupName, uint32_t mtype, MetricInfo *data);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetrics(DEVICE_HANDLE handle, char *metricGroupName,
 *						  uint32_t stype,  uint32_t period, uint32_t numReports)
 *
 * @brief	   enable named metricgroup to collection.
 *
 * @param	   IN   handle			 - handle to the selected device
 * @param	   IN   metricGroupName	- metric group name
 * @param	   IN   metricGroupCode	- metric group code
 * @param	   IN   mtype			  - metric type <TIME_BASED, EVENT_BASED>
 * @param	   IN   period			 - collection timer period
 * @param	   IN   numReports		 - number of reports. Default collect all available metrics
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUEnableMetricGroup(DEVICE_HANDLE handle, 
		char *metricGroupName, uint32_t metricGroupCode, uint32_t mtype,
				uint32_t period, uint32_t numReports);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetrics(DEVICE_HANDLE handle, char **metricNameList,
 *						  unsigned numMetrics, uint32_t stype,
 *						  uint32_t period, uint32_t numReports)
 *
 * @brief	   enable named metrics on a metric group to collection.
 *
 * @param	   IN   handle			 - handle to the selected device
 * @param	   IN   metricNameList	 - a list of metric names to be collected
 * @param	   IN   numMetrics		 - number of metrics to be collected
 * @param	   IN   mtype			  - metric type <TIME_BASED, EVENT_BASED>
 * @param	   IN   period			 - collection timer period
 * @param	   IN   numReports		 - number of reports. Default collect all available metrics
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUEnableMetrics(DEVICE_HANDLE handle, char **metricNameList,
					 uint32_t numMetrics, uint32_t mtype,
					 uint32_t period, uint32_t numReports);

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUDisableMetrics(DEVICE_HANDLE handle, uint32_t mtype);
 *
 * @brief	   disable current metric collection
 *
 * @param	   IN   handle		   - handle to the selected device
 * @param	   IN   mtype			- a metric group type
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUDisableMetricGroup(DEVICE_HANDLE handle, uint32_t mtype);


/*------------------------------------------------------------------------- */
/*!
 * @fn int GPUStart(DEVICE_HANDLE handle);
 *
 * @brief	   start collection
 *
 * @param	   IN   handle - handle to the selected device
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUStart(DEVICE_HANDLE handle);


/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUStop(DEVICE_HANDLE handle,  MetricData **data, int *reportCounts);
 *
 * @brief	   stop collection
 *
 * @param	   IN   handle		 - handle to the selected device
 * @param	   OUT  data		   - returned metric data array
 * @param	   OUT  reportCounts   - returned metric data array size
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUStop(DEVICE_HANDLE handle);



/* ------------------------------------------------------------------------- */
/*!
 * @fn Metricdata *GPUReadMetricData(DEVICE_HANDLE handle, uint32_t mode, uint32_t *reportCounts);
 *
 * @brief	   read metric data
 *
 * @param	   IN   handle		 - handle to the selected device
 * @param	   IN   mode		   - collection mode (sample, summary, reset)
 * @param	   OUT  reportCounts   - returned metric data array size
 *
 * @return	  data				- returned metric data array
 */
MetricData *GPUReadMetricData(DEVICE_HANDLE handle, uint32_t mode, uint32_t *reportCounts);

/* ------------------------------------------------------------------------- */
/*!
 * @fn Metricdata *GPUSetMetricControl(DEVICE_HANDLE handle, uint32_t mode);
 *
 * @brief	   set  control for metric data collection
 *
 * @param	   IN   handle		 - handle to the selected device
 * @param	   IN   mode		   - collection mode (sample, summary, reset)
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUSetMetricControl(DEVICE_HANDLE handle, uint32_t mode);


/* ------------------------------------------------------------------------- */
/*!
 * @fn void GPUFreeDevice(DEVICE_HANDLE handle);
 *
 * @brief	   free the resouce related this device handle
 *
 * @param	   IN   handle	   - handle to the selected device
 *
 * @return	  0 -- success,  otherwise, error code
 */
//void GPUFreeDevice();
void GPUFreeDevice(DEVICE_HANDLE handle);

#if defined(__cplusplus)
}
#endif

#endif
