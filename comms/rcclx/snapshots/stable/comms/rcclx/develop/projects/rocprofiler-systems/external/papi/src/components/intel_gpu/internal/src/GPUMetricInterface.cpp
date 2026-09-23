/*
 * GPUMetricInterface.cpp:  Intel® Graphics Component for PAPI
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
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>

#include "GPUMetricInterface.h"
#include "GPUMetricHandler.h"

#ifdef __cplusplus
extern "C" {
#endif 

//#define _DEBUG 1

#define  DebugPrintError(format, args...)	fprintf(stderr, format, ## args)

#if defined(_DEBUG)
#define DebugPrint(format, args...)		  fprintf(stderr, format, ## args)
#else
#define DebugPrint(format, args...)		  {do { } while(0);}
#endif


#define UNINITED	0
#define INITED	  1
#define ENABLED	 2

#define MAX_ENTRY	256
#define MAX_HANDLES	16

/* global device information*/
static DeviceInfo *gDeviceInfo = nullptr;
static uint32_t	   gNumDeviceInfo = 0;

// maintail available handlers, one per subdevice
static GPUMetricHandler **gHandlerTable	= nullptr;
static uint32_t	 gNumHandles = 0;
static DEVICE_HANDLE *gHandles = nullptr;

static int runningSessions = 0;

// lock to make sure API is thread-safe
static std::mutex  infLock;

/*  build the key from <driver, device, handler, index> */
static
GPUMetricHandler *getHandler(DEVICE_HANDLE handle) {
	 uint32_t index = GetIdx(handle);
	 if (index >= gNumHandles) {
		return nullptr;
	 }
	 return gHandlerTable[index];
}

/*
 * wrapper function for safe strncpy
 */
void 
strncpy_se(char *dest, size_t destSize,  char *src, size_t count)  {
	if (dest && src) {
		size_t toCopy = (count <= strlen(src))? count:strlen(src);
		if (toCopy < destSize) {
			memcpy(dest, src, toCopy);
			dest[toCopy] = '\0';
		}
	}
}


/*============================================================================
 * Below are Wrapper interface functions
 *============================================================================*/

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUDetectDevice(DEVICE_HANDLE **handle, uint32_t *num_device);
 *
 * @brief	   Detect and init GPU device which has performance metrics availale.
 *
 * @param	   OUT  handles   - a array of handle, each for an instance of the device
 * @param	   OUT  numDevice - total number of device detected
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUDetectDevice(DEVICE_HANDLE **handles, uint32_t *numDevices)
{
	int ret = 0;
	int retError = 1;

	infLock.lock();

	if (gNumHandles) {
		*handles =   gHandles;
		*numDevices =  gNumHandles;
		infLock.unlock();
		return ret;
	}

	ret = GPUMetricHandler::InitMetricDevices(&gDeviceInfo, &gNumDeviceInfo, &gNumHandles);
	if (!gDeviceInfo || !gNumDeviceInfo)  {
		DebugPrintError("InitMetricDevices failed,"
					   	" device does not exist or cannot find dependent libraries, abort\n");
		infLock.unlock();
		return retError;
	}

	gHandlerTable = (GPUMetricHandler **)calloc(gNumHandles, sizeof(GPUMetricHandler *));
	gHandles = (DEVICE_HANDLE *)calloc(gNumHandles, sizeof(DEVICE_HANDLE));
	uint32_t index = 0;
	uint32_t dcode = 0;
	for (uint32_t i=0; i<gNumDeviceInfo; i++) {
		/* for root device */
		dcode = CreateDeviceCode(gDeviceInfo[i].driverId, gDeviceInfo[i].deviceId, 0);
		dcode = CreateIdxCode(dcode, index);
		gHandlerTable[index] = GPUMetricHandler::GetInstance(gDeviceInfo[i].driverId,
										gDeviceInfo[i].deviceId, 0);
		gHandles[index++] = dcode;
		if (gDeviceInfo[i].numSubdevices) {
			for (uint32_t j=0; j<gDeviceInfo[i].numSubdevices; j++)  {
				dcode = CreateDeviceCode(gDeviceInfo[i].driverId, gDeviceInfo[i].deviceId, (j+1));
				dcode = CreateIdxCode(dcode, index);
				gHandlerTable[index] = GPUMetricHandler::GetInstance(gDeviceInfo[i].driverId, 
				gDeviceInfo[i].deviceId, j+1);
				gHandles[index++] = dcode;
			}
		}
	}
	*handles = gHandles;
	*numDevices = gNumHandles;

	infLock.unlock();
	return ret;	
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn void GPUFreeDevice(DEVICE_HANDLE handle);
 *
 * @brief	   free the resouce related this device handle
 *
 * @param	   IN   handle	   - handle to the selected device
 *
 */
void GPUFreeDevice(DEVICE_HANDLE handle)
{
	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUFreeDevice failed, handle is not valid or device is not initiated!\n");
		return;
	}
	getHandler(handle)->DestroyMetricDevice();
}


/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUEnableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName
 *				 unsigned int mtype,  unsigned int period, unsigned int numReports)
 *
 * @brief	   add named metric group tby name or code o collection.
 *
 * @param	   IN   handle			  - handle to the selected device
 * @param	   IN   metricGroupName	 - a metric group name
 * @param	   IN   metricGroupCode	 - a metric group code
 * @param	   IN   mtype			   - metric type <TIME_BASED,  EVENT_BASED>
 * @param	   IN   period			  - collection timer period
 * @param	   IN   numReports		  - number of reports. Default collect all available metrics
 *
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUEnableMetricGroup(DEVICE_HANDLE handle, char *metricGroupName, uint32_t metricGroupCode,
		unsigned int mtype, unsigned int period, unsigned int numReports)
{
	int ret = 0;
	int retError = 1;
	unsigned int groupCode  = 0;

	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUEnableMetricGroup: device handle is not initiated!\n");
		return retError;
	}
	GPUMetricHandler *curMetricHandle = getHandler(handle);
	uint32_t curMetricGroupCode = curMetricHandle->GetCurGroupCode();

	if (metricGroupName && strlen(metricGroupName))  {
		ret=curMetricHandle->GetMetricCode(metricGroupName, "", mtype, &groupCode, nullptr);
		if (ret) {
			DebugPrintError("GPUEnableMetricGroup: metric group %s is not supported, abort!\n",
				 metricGroupName);
			return ret;
		}
	} else {
		groupCode = metricGroupCode;
	}
	if (curMetricGroupCode) {   // already in collecting a metric group
		if (groupCode == curMetricGroupCode)  {
			runningSessions++;
			return ret;
		} else {
			DebugPrintError("GPUEnableMetricGroup failed, reason:"
					  " tried to collect more than one metric groups at the smae time."
					  " Collection abort!\n");
			return retError;
		}
	}

	infLock.lock();
	int enabled = 0;
	ret = curMetricHandle->EnableMetricGroup(groupCode, mtype, &enabled);
	if (!ret && !enabled) {
		if (mtype == TIME_BASED) {
			ret = curMetricHandle->EnableTimeBasedStream(period, numReports);
		} else {
			ret = curMetricHandle->EnableEventBasedQuery();
		}
	}

	if (!ret) {
		runningSessions++;
	};
	infLock.unlock();

	return ret;
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUDisableMetricGroup(DEVICE_HANDLE handle, unsigned int mtype);
 *
 * @brief	   disable a metric group configured for the device
 *
 * @param	   IN   handle			- handle to the selected device
 * @param	   IN   mtype			 - a metric group type
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUDisableMetricGroup(DEVICE_HANDLE handle, unsigned mtype)
{

	int ret = 0;
	int retError = 1;

	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUDisableMetricGroup: device handle is not initiated!\n");
		return retError;
	}
	infLock.lock();
	runningSessions--;
	if (runningSessions == 0)  {
	   if (mtype == TIME_BASED) {
		   GPUMetricHandler *curMetricHandler = getHandler(handle);
		   curMetricHandler->DisableMetricGroup();
		}
	}
	infLock.unlock();
	return ret;
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn int GPUSetMetricControl(DEVICE_HANDLE handle, unsigned int mode);
 *
 * @brief	   set controls for metroc collection
 *
 * @param	   IN   handle			- handle to the selected device
 * @param	   IN   mode			- a metric collection control mode
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUSetMetricControl(DEVICE_HANDLE handle, unsigned int mode)
{
	int ret			   = 0;
	int retError		  = 1;
	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUStop: device handle is not initiated!\n");
		return retError;
	}

	infLock.lock();
	getHandler(handle)->SetControl(mode);
	infLock.unlock();
	return ret;
}

/* ----------------------------------------------------------------------------------------- */
/*!
 * @fn		 MetricData *GPUReadMetricData(DEVICE_HANDLE handle, 
 *								int mode, unsigned int *reportCounts);
 *
 * @brief	   read metric data
 *
 * @param	   IN   handle		  - handle to the selected device
 * @param	   IN   mode		  - reprot data mode,  METRIC_SUMMARY, METIC_SAMPLE
 * @param	   OUT  reportCounts  - returned metric data array size
 *
 * @return	  data				- returned metric data array
 */
MetricData *GPUReadMetricData(DEVICE_HANDLE handle, unsigned int mode, unsigned int *reportCounts)
{
	MetricData   *reportData = nullptr;

	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUReadMetricData: device handle is not initiated!\n");
		return nullptr;
	}
	unsigned int numReports = 0;
	infLock.lock();
	reportData = getHandler(handle)->GetMetricData(mode, &numReports);
	infLock.unlock();
	if (!reportData) {
		DebugPrintError("Failed on GPUReadMetricData\n");
	   *reportCounts = 0;
		return nullptr;
	}
	if (!numReports) {
		DebugPrintError("GPUReadMetricData: No metric is collected\n");
	   *reportCounts = 0;
		GPUFreeMetricData(reportData, numReports);
		return nullptr;
	}
	DebugPrint("GPUReadMetricData:  GetMetricData numreport %d\n", numReports);

#if defined(_DEBUG)
	for (int i=0; i<(int)numReports;  i++) {
		DebugPrint("reportData[%d], metrics %d\n", i, reportData[i].metricCount);
		for (int j=0; j<(int)reportData[i].numDataSets; j++) {
			int sidx = reportData[i].dataSetStartIdx[j];
			if (sidx < 0) {
				 continue;
			}
			for (int k=0; k<(int)reportData[i].metricCount; k++) {
				DebugPrint("record[%d], dataSet[%d], metric [%d]: code 0x%x, ",
							i, j, k, reportData[i].dataEntries[sidx+k].code);
				if (reportData[i].dataEntries[sidx+k].type) {
					DebugPrint("value %lf \n", reportData[i].dataEntries[sidx+k].value.fpval);
				} else {
					DebugPrint("value %llu\n", reportData[i].dataEntries[sidx+k].value.ival);
				}
			}
		}
	}
#endif
	*reportCounts = (int)numReports;
	return reportData;
}


/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data);
 *
 * @brief	   list all available metric groups for the selected type 
 * @param	   IN   handle - handle to the selected device
 * @param	   IN   mtype  - metric group type
 * @param	   OUT  data   - metric data
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUGetMetricGroups(DEVICE_HANDLE handle, unsigned int mtype, MetricInfo *data)
{
	int ret			   = 0;
	int retError		  = 1;

	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUGetMetricGroups: device handle is not initiated!\n");
		return retError;
	}
	infLock.lock();
	ret = getHandler(handle)->GetMetricInfo(mtype, data);
	infLock.unlock();
	if (ret) { 
		DebugPrintError("GPUGetMetricGroups failed, return %d\n", ret);
	}
#if defined(_DEBUG)
	for (int i=0; i<data->numEntries; i++) {
		DebugPrint("GPUGetMetricGroups: metric group[%d]: %s, code 0x%x, numEntries %d\n",
			i, data->infoEntries[i].name, data->infoEntries[i].code,
			data->infoEntries[i].numEntries);
	}
#endif

	return ret;
}


/* ------------------------------------------------------------------------- */
/*!
 * @fn		  int GPUGetMetricList(DEVICE_HANDLE handle,
 *								   char *groupName, unsigned int mtype, MetricInfo *data);
 *
 * @brief	   list available metrics in the named group. 
 *			  If name is "", list all available metrics in all groups
 *
 * @param	   IN   handle	  - handle to the selected device
 * @param	   IN   groupName - metric group name. "" means all groups.
 * @param	   In   mtype	  - metric type <TIME_BASED, EVENT_BASED>
 * @param	   OUT  data	  - metric data
 *
 * @return	  0 -- success,  otherwise, error code
 */
int GPUGetMetricList(DEVICE_HANDLE handle, char *groupName, unsigned mtype, MetricInfo *data)
{
	int ret	  = 0;
	int retError = 1;

	if (!handle || !getHandler(handle)) {
		DebugPrintError("GPUGetMetricList: device handle (0x%x) is not initiated!\n", handle);
		return retError;
	}

	if (groupName == nullptr) {
		return retError;
	}

	GPUMetricHandler *mHandler = getHandler(handle);

	if (strlen(groupName) > 0) {
		ret = mHandler->GetMetricInfo(groupName, mtype, data);
	} else {
		ret = mHandler->GetMetricInfo("", mtype, data);
	}
	if (ret) {
		DebugPrintError("GPUGetMetrics [%s] failed, return %d\n", groupName, ret);
	}
	return ret;
}

/************************************************************************************************
 * Memory allocate/free functions are defiend to allow C code caller to free up the memory space
 ************************************************************************************************/
/* ------------------------------------------------------------------------- */
/*!
 * @fn		  MetricInfo *GPUAllocMetricInfo(uint32_t count)
 *
 * @brief	   allocate memory for metrics info
 * @param	   IN   count	  - number of entries of MetricInfo
 *
 * @return	   array of MetricInfo
 */
MetricInfo *
GPUAllocMetricInfo(uint32_t count) {
	return (MetricInfo *)calloc(count, sizeof(MetricInfo));
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  void GPUFreeMetricInfo(MetricInfo *info, uint32_t count);
 *
 * @brief	   free  memory space for metrics info
 * @param	   IN   count	  - number of entries of MetricInfo
 *
 */
void
GPUFreeMetricInfo(MetricInfo *info, uint32_t count) {
	if (!info  || !count) {
		return;
	}
	for (uint32_t i=0; i<count; i++) {
		if (info[i].infoEntries) {
			  free(info[i].infoEntries);
		 }
	}
	free(info);
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  MetricData * GPUAllocMetricData(uint32_t count, 
 *									uint32_t numSets, uint32_t numMetrics) {
 *
 * @brief	   allocate memory space for metrics data
 * @param	   IN   count	    - number of entries of MetricData
 * @param	   IN   numSets	    - number of metrics sets
 * @param	   IN   numMetrics  - number of metrics per set
 *
 * @return	   array of MetricData, nullptr if out of space.
 */
MetricData *
GPUAllocMetricData(uint32_t count, uint32_t numSets, uint32_t numMetrics) {
	MetricData *data = (MetricData *)calloc(count, sizeof(MetricData));
	if (!data) {
		return data;
	}
	uint32_t numEntries  = numSets * numMetrics;
	for (uint32_t i=0; i<count; i++) {
		data[i].numDataSets = numSets;
		data[i].metricCount = numMetrics;
		data[i].numEntries = numEntries;
		data[i].dataSetStartIdx = (int *)calloc(numSets, sizeof(int));
		data[i].dataEntries = (DataEntry *)calloc(numEntries, sizeof(DataEntry));
		if (!data[i].dataSetStartIdx || !data[i].dataEntries) {
			GPUFreeMetricData(data, count);
			return nullptr;
		}
	}
	return data;
}

/* ------------------------------------------------------------------------- */
/*!
 * @fn		  void GPUFreeMetricInfo(MetricData *data, uint32_t count);
 *
 * @brief	   free  memory space for metrics data
 * @param	   IN   count	  - number of entries of MetricInfo
 */
void
GPUFreeMetricData(MetricData *data, uint32_t count) {
	if (!data  || !count) {
		return;
	}
	for (uint32_t i=0; i<count; i++) {
		if (data[i].dataSetStartIdx)  free(data[i].dataSetStartIdx);
		if (data[i].dataEntries)  free(data[i].dataEntries);
	}
	free(data);
}

#if defined(__cplusplus)
}
#endif
