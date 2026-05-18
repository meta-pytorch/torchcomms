/*
 * GPUMetricHandler.h:  Intel® Graphics Component for PAPI
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

#ifndef _GPUMETRICHANDLER_H
#define _GPUMETRICHANDLER_H

#pragma once

#include <stdio.h>
#include <string.h>
#include <list>
#include <string>
#include <fstream>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>

#include "level_zero/zet_api.h"

#include "GPUMetricInterface.h"

/* collection status */
#define COLLECTION_IDLE	  0
#define COLLECTION_INIT	  1
#define COLLECTION_CONFIGED  2
#define COLLECTION_ENABLED   3
#define COLLECTION_DISABLED  4
#define COLLECTION_COMPLETED 5

using namespace std;
class GPUMetricHandler;

typedef struct TMetricNode_S 
{
	uint32_t code;		   // 0 mean invalid
	uint32_t metricId;
	uint32_t metricGroupId;
	uint32_t metricDomainId;
	int	metricType;
	zet_metric_properties_t props;
	zet_metric_handle_t handle;
} TMetricNode;

typedef struct TMetricGroupNode_S
{
	uint32_t code;		 // 0 mean invalid
	uint32_t numMetrics;
	TMetricNode *metricList;
	int *opList;	   // list of metric operation
	zet_metric_group_properties_t props;
	zet_metric_group_handle_t handle;
}TMetricGroupNode;


typedef struct TMetricGroupInfo_S {
	uint32_t				 numMetricGroups;
	uint32_t				 numMetrics;
	uint32_t				 maxMetricsPerGroup;
	uint32_t				 domainId;
	TMetricGroupNode		*metricGroupList;
}TMetricGroupInfo;


typedef struct QueryData_S {
	string									kernName;
	zet_metric_query_handle_t				metricQuery;
	ze_event_handle_t						event;
} QueryData;

typedef struct QueryState_S {
	atomic<uint32_t>						kernelId{0};
	std::mutex								lock;
	std::map<ze_kernel_handle_t, string>	nameMap;
	zet_metric_query_pool_handle_t			queryPool;
	ze_event_pool_handle_t					eventPool;
	GPUMetricHandler					   *handle;
	vector<QueryData>						queryList;
} QueryState;

typedef struct InstanceData {
	uint32_t    kernelId;
	QueryState *queryState;
	zet_metric_query_handle_t metricQuery;
} InstanceData;


class GPUMetricHandler
{
public:
	static int InitMetricDevices(DeviceInfo **deviceInfoList, uint32_t *numDeviceInfo, 
                     uint32_t *totalDevices);
	static GPUMetricHandler* GetInstance(uint32_t driverId, uint32_t deviceId, 
                     uint32_t subdeviceId);
	~GPUMetricHandler();
	void DestroyMetricDevice();
	int  EnableMetricGroup(uint32_t metricGroupCode, uint32_t mtype, int *status);
	int	 EnableMetricGroup(const char *metricGroupName,  uint32_t mtype, int *status);
	int	 EnableTimeBasedStream(uint32_t timePeriod, uint32_t numReports);
	int	 EnableEventBasedQuery();
	void DisableMetricGroup();
	int	 GetMetricInfo(int type, MetricInfo *data);
	int	 GetMetricInfo(const char * name, int type, MetricInfo *data);
	int  GetMetricCode(const char *mGroupName, const char *metricName,	uint32_t mtype, 
						uint32_t *mGroupCode, uint32_t *metricCode);
	MetricData   *GetMetricData(uint32_t  mode, uint32_t *numReports);
	int	 SetControl(uint32_t mode);
	uint32_t GetCurGroupCode();


private:
	GPUMetricHandler(uint32_t driverid, uint32_t deviceid, uint32_t subdeviceid);
	GPUMetricHandler(GPUMetricHandler const&);
	void  operator=(GPUMetricHandler const&);
	static int  InitMetricGroups(ze_device_handle_t device, TMetricGroupInfo *mgroups);
	string  GetDeviceName(ze_device_handle_t device);
	uint8_t	*ReadStreamData(size_t *rawDataSize);
	uint8_t	*ReadQueryData(QueryData &data, size_t *rawDataSize, ze_result_t *retStatus);
	void	GenerateMetricData(uint8_t *rawData, size_t rawDataSize, uint32_t  mode);
	void	ProcessMetricDataSet(uint32_t dataSetId, zet_typed_value_t* typedDataList,
					uint32_t startIdx, uint32_t dataSize,
					uint32_t metricCount, uint32_t  mode);

private: // Fields
	static GPUMetricHandler*  m_handlerInstance;
	static vector<int>		  driverList;
	static vector<ze_device_handle_t> m_deviceList;
	static vector<GPUMetricHandler>	  handlerList;

	int m_driverId;
	int m_deviceId;
	int m_subdeviceId;

	std::mutex	m_lock;
	string		m_dataDumpFileName;
	string		m_dataDumpFilePath;
	fstream		m_dataDump;

	// current state
	ze_driver_handle_t  m_driver;
	ze_context_handle_t	m_context;
	ze_device_handle_t  m_device;
	TMetricGroupInfo  *m_groupInfo;
	uint32_t m_domainId;
	int		 m_groupId;
	uint32_t m_groupType;
	QueryState *m_queryState;

	ze_event_pool_handle_t m_eventPool;
	ze_event_handle_t m_event;

	zet_metric_streamer_handle_t m_metricStreamer;
	zet_metric_query_pool_handle_t m_queryPool;
	zet_tracer_exp_handle_t	m_tracer;

	volatile int	m_status;
	uint32_t		m_numDevices;
	uint32_t		m_numDataSet;
	MetricData		*m_reportData;
	uint32_t		*m_reportCount;
};

/* this struct maintains the <driver, device> information */
typedef struct TMetricDevice_S {
	uint32_t			 driverId;
	uint32_t			 deviceId;
	uint32_t			 metricHandlerIndex;
	uint32_t			 numSubdevices;
	TMetricGroupInfo	*groupList;
	char				*devName;
	ze_driver_handle_t  *ze_driver;
	ze_device_handle_t  *ze_device;
} TMetricDevice;

/* This struct maintains the <driver, device, subdevice> information
 * This is the based to access the GPUMetricHandler method
 */
typedef struct TMetricDeviceHandler_S {
	uint32_t			 driverId;
	uint32_t			 deviceId;
	uint32_t			 subdeviceId;
	uint32_t			 deviceListIndex;
	GPUMetricHandler	*handler;
} TMetricDeviceHandler;

#endif

