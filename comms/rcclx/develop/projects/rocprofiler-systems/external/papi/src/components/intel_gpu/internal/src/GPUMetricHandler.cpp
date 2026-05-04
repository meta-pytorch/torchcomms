/*
 * GPUMetricHandler.cpp:   Intel® Graphics Component for PAPI
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

#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdarg>
#include <stdexcept>
#include <semaphore.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include <vector>
#include <map>
#include <mutex>

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <level_zero/ze_ddi.h>
#include <level_zero/zet_ddi.h>

#include "GPUMetricHandler.h"

#define  DebugPrintError(format, args...)	 fprintf(stderr, format, ## args)

//#define _DEBUG 1

#if defined(_DEBUG)
#define DebugPrint(format, args...)		   fprintf(stderr, format, ## args)
#else 
#define DebugPrint(format, args...)		   {do {} while(0); }
#endif

#define MAX_REPORTS		 32768
#define MAX_KERNERLS	 1024

#define CHECK_N_RETURN_STATUS(status, retVal)	{if (status) return retVal; }
#define CHECK_N_RETURN(status)				   {if (status) return;}

#define ENABLE_RAW_LOG	"ENABLE_RAW_LOG"


/************************************************************************/
/* Helper function													  */
/************************************************************************/
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

ze_pfnInit_t								zeInitFunc;
ze_pfnDriverGet_t							zeDriverGetFunc;
ze_pfnDriverGetProperties_t					zeDriverGetPropertiesFunc;
ze_pfnDeviceGet_t							zeDeviceGetFunc;
ze_pfnDeviceGetSubDevices_t					zeDeviceGetSubDevicesFunc;
ze_pfnDeviceGetProperties_t					zeDeviceGetPropertiesFunc;
ze_pfnEventPoolCreate_t						zeEventPoolCreateFunc;
ze_pfnEventPoolDestroy_t					zeEventPoolDestroyFunc;
ze_pfnContextCreate_t						zeContextCreateFunc;
ze_pfnEventCreate_t							zeEventCreateFunc;
ze_pfnEventDestroy_t						zeEventDestroyFunc;
ze_pfnEventHostSynchronize_t				zeEventHostSynchronizeFunc;

zet_pfnMetricGroupGet_t						zetMetricGroupGetFunc;
zet_pfnMetricGroupGetProperties_t			zetMetricGroupGetPropertiesFunc;
zet_pfnMetricGroupCalculateMetricValues_t   zetMetricGroupCalculateMetricValuesFunc;
zet_pfnMetricGroupCalculateMultipleMetricValuesExp_t  zetMetricGroupCalculateMultipleMetricValuesExpFunc;
zet_pfnMetricGet_t							zetMetricGetFunc;
zet_pfnMetricGetProperties_t				zetMetricGetPropertiesFunc;
zet_pfnContextActivateMetricGroups_t		zetContextActivateMetricGroupsFunc;
zet_pfnMetricStreamerOpen_t					zetMetricStreamerOpenFunc;
zet_pfnMetricStreamerClose_t				zetMetricStreamerCloseFunc;
zet_pfnMetricStreamerReadData_t				zetMetricStreamerReadDataFunc;
zet_pfnMetricQueryPoolCreate_t				zetMetricQueryPoolCreateFunc;
zet_pfnMetricQueryPoolDestroy_t				zetMetricQueryPoolDestroyFunc;
zet_pfnMetricQueryCreate_t					zetMetricQueryCreateFunc;
zet_pfnMetricQueryDestroy_t					zetMetricQueryDestroyFunc;
zet_pfnMetricQueryReset_t					zetMetricQueryResetFunc;
zet_pfnMetricQueryGetData_t					zetMetricQueryGetDataFunc;
zet_pfnTracerExpCreate_t					zetTracerExpCreateFunc;
zet_pfnTracerExpDestroy_t					zetTracerExpDestroyFunc;
zet_pfnTracerExpSetPrologues_t				zetTracerExpSetProloguesFunc;
zet_pfnTracerExpSetEpilogues_t				zetTracerExpSetEpiloguesFunc;
zet_pfnTracerExpSetEnabled_t				zetTracerExpSetEnabledFunc;
zet_pfnCommandListAppendMetricQueryBegin_t  zetCommandListAppendMetricQueryBeginFunc;
zet_pfnCommandListAppendMetricQueryEnd_t	zetCommandListAppendMetricQueryEndFunc;

#define DLL_SYM_CHECK(handle, name, type)				\
	do {												\
		name##Func = (type) dlsym(handle, #name);		\
		if (dlerror() != nullptr) {						\
			DebugPrintError("failed: %s\n", #name);		\
		   return 1;									\
		}												\
   } while (0)

static int
functionInit(void *dllHandle)
{
	int ret  = 0;
	DLL_SYM_CHECK(dllHandle, zeInit, ze_pfnInit_t);
	DLL_SYM_CHECK(dllHandle, zeDriverGet, ze_pfnDriverGet_t);
	DLL_SYM_CHECK(dllHandle, zeDriverGetProperties, ze_pfnDriverGetProperties_t);
	DLL_SYM_CHECK(dllHandle, zeDeviceGet, ze_pfnDeviceGet_t);
	DLL_SYM_CHECK(dllHandle, zeDeviceGetSubDevices, ze_pfnDeviceGetSubDevices_t);
	DLL_SYM_CHECK(dllHandle, zeDeviceGetProperties, ze_pfnDeviceGetProperties_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGroupGet, zet_pfnMetricGroupGet_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGroupGetProperties, zet_pfnMetricGroupGetProperties_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGet, zet_pfnMetricGet_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGetProperties, zet_pfnMetricGetProperties_t);
	DLL_SYM_CHECK(dllHandle, zetContextActivateMetricGroups, zet_pfnContextActivateMetricGroups_t);
	DLL_SYM_CHECK(dllHandle, zeEventPoolCreate, ze_pfnEventPoolCreate_t);
	DLL_SYM_CHECK(dllHandle, zeEventPoolDestroy, ze_pfnEventPoolDestroy_t);
	DLL_SYM_CHECK(dllHandle, zeContextCreate, ze_pfnContextCreate_t);
	DLL_SYM_CHECK(dllHandle, zeEventCreate, ze_pfnEventCreate_t);
	DLL_SYM_CHECK(dllHandle, zeEventDestroy, ze_pfnEventDestroy_t);
	DLL_SYM_CHECK(dllHandle, zeEventHostSynchronize, ze_pfnEventHostSynchronize_t);
	DLL_SYM_CHECK(dllHandle, zetMetricStreamerOpen, zet_pfnMetricStreamerOpen_t);
	DLL_SYM_CHECK(dllHandle, zetMetricStreamerClose, zet_pfnMetricStreamerClose_t);
	DLL_SYM_CHECK(dllHandle, zetMetricStreamerReadData, zet_pfnMetricStreamerReadData_t);
	DLL_SYM_CHECK(dllHandle, zetMetricQueryPoolCreate, zet_pfnMetricQueryPoolCreate_t);
	DLL_SYM_CHECK(dllHandle, zetMetricQueryPoolDestroy, zet_pfnMetricQueryPoolDestroy_t);
	DLL_SYM_CHECK(dllHandle, zetMetricQueryCreate, zet_pfnMetricQueryCreate_t);
	DLL_SYM_CHECK(dllHandle, zetMetricQueryDestroy, zet_pfnMetricQueryDestroy_t);
	DLL_SYM_CHECK(dllHandle, zetMetricQueryGetData, zet_pfnMetricQueryGetData_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGroupCalculateMetricValues, 
				zet_pfnMetricGroupCalculateMetricValues_t);
	DLL_SYM_CHECK(dllHandle, zetMetricGroupCalculateMultipleMetricValuesExp, 
				zet_pfnMetricGroupCalculateMultipleMetricValuesExp_t);
	DLL_SYM_CHECK(dllHandle, zetTracerExpCreate, zet_pfnTracerExpCreate_t);
	DLL_SYM_CHECK(dllHandle, zetTracerExpDestroy, zet_pfnTracerExpDestroy_t);
	DLL_SYM_CHECK(dllHandle, zetTracerExpSetPrologues, zet_pfnTracerExpSetPrologues_t);
	DLL_SYM_CHECK(dllHandle, zetTracerExpSetEpilogues, zet_pfnTracerExpSetEpilogues_t);
	DLL_SYM_CHECK(dllHandle, zetTracerExpSetEnabled, zet_pfnTracerExpSetEnabled_t);
	DLL_SYM_CHECK(dllHandle, zetCommandListAppendMetricQueryBegin, 
				zet_pfnCommandListAppendMetricQueryBegin_t);
	DLL_SYM_CHECK(dllHandle, zetCommandListAppendMetricQueryEnd, 
				zet_pfnCommandListAppendMetricQueryEnd_t);
	return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn	static void kernelCreateCB(
 *					 ze_command_list_append_launch_kernel_params_t* params,
 *					 ze_result_t result, void* pUserData, void** instanceData)
 *
 * @brief callback function called to add kernel into name mapping.
 *		It is called when exiting kerenl creation.
 *
 * @param IN	params		  -- kerne parametes
 * @param IN	result		  -- kernel launch result
 * @param INOUT pUserData	  -- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 */
static void kernelCreateCB(ze_kernel_create_params_t *params,
				ze_result_t result, void *pUserData, void **instanceData)
{
	(void)instanceData;
	if (result != ZE_RESULT_SUCCESS) {
		return;
	}
	QueryState *queryState =  reinterpret_cast<QueryState *>(pUserData);
	queryState->lock.lock();
	ze_kernel_handle_t kernel = **(params->pphKernel);
	queryState->nameMap[kernel] = (*(params->pdesc))->pKernelName;
	CHECK_N_RETURN(queryState==nullptr);
	queryState->lock.unlock();
	return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	static void kernelDestroyCB(
 *					 ze_command_list_append_launch_kernel_params_t* params,
 *					 ze_result_t result, void* pUserData, void** instanceData)
 *
 * @brief callback function to remove kernel from maintained mapping
 *		It is called when exiting kerenl destroy.
 *
 * @param IN	params		  -- kerne parametes
 * @param IN	result		  -- kernel launch result
 * @param INOUT pUserData	  -- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 */
static void
kernelDestroyCB(ze_kernel_destroy_params_t *params,
		ze_result_t result, void *pUserData, void **instanceData)
{
	(void)instanceData;
	if (result != ZE_RESULT_SUCCESS) {
		return;
	}
	QueryState *queryState =  reinterpret_cast<QueryState *>(pUserData);
	CHECK_N_RETURN(queryState==nullptr);
	queryState->lock.lock();
	ze_kernel_handle_t kernel = *(params->phKernel);
	if (queryState->nameMap.count(kernel) != 1) {
		queryState->lock.unlock();
		return;
	}
	queryState->nameMap.erase(kernel);
	queryState->lock.unlock();
	return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn	static void metricQueryBeginCB(
 *					 ze_command_list_append_launch_kernel_params_t* params,
 *					 ze_result_t result, void* pUserData, void** instanceData)
 *
 * @brief callback to append "query begin" request into command list before kernel launches.
 *
 * @param IN	params		-- kerne parametes
 * @param IN	result		-- kernel launch result
 * @param INOUT pUserData	-- pointer to the location which stores global data
 * @param OUT   instanceData  -- pointer to the location which stores the data for this query
 *
 * @return
 *
 */
static void
metricQueryBeginCB(
	ze_command_list_append_launch_kernel_params_t* params,
	ze_result_t result, void* pUserData, void** instanceData)
{
	(void)result;
	(void)instanceData;

	QueryState *queryState =  reinterpret_cast<QueryState *>(pUserData);
	CHECK_N_RETURN(queryState==nullptr);

	// assign each kernel an id for reference.
	uint32_t kernId = queryState->kernelId.fetch_add(1, std::memory_order_acq_rel);
	if (kernId >=  MAX_KERNERLS) {
		*instanceData = nullptr;
		return;
	}
	ze_result_t status = ZE_RESULT_SUCCESS;
	zet_metric_query_handle_t metricQuery = nullptr;
	status = zetMetricQueryCreateFunc(queryState->queryPool, kernId, &metricQuery);
	CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);
	ze_command_list_handle_t commandList = *(params->phCommandList);
	CHECK_N_RETURN(commandList== nullptr);
	status = zetCommandListAppendMetricQueryBeginFunc(commandList, metricQuery);
	CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

	// maintain the query for current kernel
	InstanceData* data = new InstanceData;
	CHECK_N_RETURN(data==nullptr);
	data->kernelId = kernId;
	data->metricQuery = metricQuery;
	data->queryState = queryState;
	*instanceData = reinterpret_cast<void*>(data);

	return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	static void metricQueryEndCB(
 *					ze_command_list_append_launch_kernel_params_t* params,
 *					ze_result_t result, void* pUserData, void** instanceData)
 *
 * @brief callback to append "query end" request into command list after kernel completes
 *
 * @param IN	params		-- kerne parametes
 * @param IN	result		-- kernel launch result
 * @param INOUT pUserData	-- pointer to the location which stores global data
 * @param INOUT instanceData  -- pointer to the location which stores the data for this query
 *
 */
static void
metricQueryEndCB(
	ze_command_list_append_launch_kernel_params_t* params,
	ze_result_t result, void* pUserData, void** instanceData)
{
	(void)result;
	InstanceData* data = reinterpret_cast<InstanceData*>(*instanceData);
	CHECK_N_RETURN(data==nullptr);
	QueryState *queryState =  reinterpret_cast<QueryState *>(pUserData);
	CHECK_N_RETURN(queryState==nullptr);
	ze_command_list_handle_t commandList = *(params->phCommandList);
	CHECK_N_RETURN(commandList==nullptr);

	ze_event_desc_t eventDesc;
	eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
	eventDesc.index = data->kernelId;
	eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
	eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

	ze_event_handle_t event = nullptr;
	ze_result_t status	  = ZE_RESULT_SUCCESS;

	status = zeEventCreateFunc(queryState->eventPool, &eventDesc, &event);
	CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

	status = zetCommandListAppendMetricQueryEndFunc(commandList, data->metricQuery, event,
											0, nullptr);
	CHECK_N_RETURN(status!=ZE_RESULT_SUCCESS);

	queryState->lock.lock();
	ze_kernel_handle_t kernel = *(params->phKernel);
	if (queryState->nameMap.count(kernel) == 1) {
		QueryData queryData = { queryState->nameMap[kernel],
							   data->metricQuery, event };
		queryState->queryList.push_back(queryData);
	}
	queryState->lock.unlock();
	return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn	   int getMetricType(const char *desc, zet_metric_type_t type )
 *
 * @brief	Find the metric type for report (accumulate, average or raw) 
 *			based on metric type and description.
 *
 * @param	IN desc  -- metric description
 * @param	IN type  -- metric type
 *
 * @ return  metric report tyep, <0: accumulate, 1:averge, 2: static>
 */

static int
getMetricType(char *desc, zet_metric_type_t metric_type) {
	if (metric_type == ZET_METRIC_TYPE_THROUGHPUT) {
		return M_ACCUMULATE;
	}
	if (metric_type == ZET_METRIC_TYPE_RATIO) {
		return M_AVERAGE;
	}
	char *ptr = nullptr;
	if ((metric_type == ZET_METRIC_TYPE_EVENT)  ||
		(metric_type == ZET_METRIC_TYPE_DURATION))  {
		if ((ptr = strstr(desc, "percentage")) ||
			(ptr = strstr(desc, "Percentage")) ||
			(ptr = strstr(desc, "Average"))	||
			(ptr = strstr(desc, "average")))   {
			return M_AVERAGE;
		}
		return M_ACCUMULATE;
	}
	return M_RAW;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn		 void * typedValue2Value(zet_typed_value_t data, std::fstream *foutstream, 
 *									 int outflag, int *dtype, long long *iVal,  
 *									 double *fpVal, int isLast)
 *
 *  @brief	Convert typed value to long long or double.  if foutstream is open, log the data
 *
 *  @param	In data		  - typed data
 *  @param	IN foutstream - output stream to dump raw data  (used for debug)
 *  @param	IN outflag	  - 1 for output to stdout. 0 for disable output to stdout
 *  @param	OUT dtype	  - data type,  0: long long,  1: double
 *  @param	OUT ival	  - converted long long integer value
 *  @param	OUT dval	  - converted double value
 *  @param	IN isLast  	  - true if the data is the last value in a metric group
 *
 */
static void
typedValue2Value(zet_typed_value_t data, std::fstream *foutstream, int outflag, int *dtype, long long *iVal,  double *fpVal, int isLast)
{
	 *dtype = 0;

	static int count = 0;
	switch( data.type )
	{
		case ZET_VALUE_TYPE_UINT32:
		case ZET_VALUE_TYPE_BOOL8:
			 if (foutstream->is_open()) {
				*foutstream << data.value.ui32 << ",";
			 }
			 if (outflag) {
				cout << data.value.ui32 << ",";
			 }
			 *iVal = (long long)data.value.ui32;
			 break;
		case ZET_VALUE_TYPE_UINT64:
			 if (foutstream->is_open()) {
				*foutstream << data.value.ui64 << ",";
			 }
			 if (outflag) {
				cout << data.value.ui64 << ",";
			 }
			 *iVal = (long long)data.value.ui64;
			 break;
		case ZET_VALUE_TYPE_FLOAT32:
			 if (foutstream->is_open()) {
				*foutstream << data.value.fp32 << ",";
			 }
			 if (outflag) {
				cout << data.value.fp32 << ",";
			 }
			 *dtype = 1;
			 *fpVal = (double)data.value.fp32;
			 break;
		case ZET_VALUE_TYPE_FLOAT64:
			 if (foutstream->is_open()) {
				*foutstream << data.value.fp64 << ",";
			 }
			 if (outflag) {
				cout << data.value.fp64 << ",";
			 }
			 *dtype = 1;
			 *fpVal = (double)data.value.fp64;
			 break;
		default:
			 break;
	}
	count++;
	if (isLast) {
		if (foutstream->is_open()) {
			*foutstream << endl;
		}
		if (outflag) {
			cout << endl;
		}
	}

}

/***************************************************************************/
/* GPUMericHandler class												   */
/***************************************************************************/

using namespace std;

GPUMetricHandler* GPUMetricHandler::m_handlerInstance = nullptr;

static vector<DeviceInfo> g_deviceInfo;
static map<uint32_t, GPUMetricHandler *>  g_metricHandlerMap;
static uint32_t	 g_stdout = 0;
static std::mutex g_lock;


/*--------------------------------------------------------------------------*/
/*!
 * GPUMetricHandler constructor
 */
GPUMetricHandler::GPUMetricHandler(uint32_t driverId, uint32_t deviceId, uint32_t subdeviceId)
	 : m_driverId(driverId), m_deviceId(deviceId), m_subdeviceId(subdeviceId)
{
	m_driver  = nullptr;
	m_device  = nullptr;
	m_context = nullptr;
	m_domainId	= 0;
	m_groupId = -1;
	m_dataDumpFileName	= "";
	m_dataDumpFilePath	= "";
	m_status  = COLLECTION_IDLE;
	m_numDevices  = 0;
	m_numDataSet  = 0;
	m_reportData  = nullptr;
	m_reportCount = nullptr;
	m_eventPool	  = nullptr;
	m_event		  = nullptr;
	m_metricStreamer  = nullptr;
	m_tracer	 = nullptr;
	m_queryPool  = nullptr;
	m_groupType  = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
}

/*-------------------------------------------------------------------------------------------*/
/*!
 * ~GPUMetricHandler
 */
GPUMetricHandler::~GPUMetricHandler()
{
	DestroyMetricDevice();
	m_handlerInstance = nullptr;
}


/*-------------------------------------------------------------------------------------------*/
/*!
 * fn	  GPUMetricHandler * GPUMetricHandler::GetInstance(uint32_t driverId, 
 *										uint32_t deviceId, uint32_t subdeviceId)
 * 
 * @brief  Get an instance of GPUMetrichander object
 *
 * @param  IN deviceId    - given driver id
 * @param  IN deviceId    - given device id
 * @param  IN subdeviceId - given subdevice id
 *
 * @return GPUMetricHandler object if such device exists, otherwise return nullptr;
 */
GPUMetricHandler *
GPUMetricHandler::GetInstance(uint32_t driverId, uint32_t deviceId, uint32_t subdeviceId)
{
	GPUMetricHandler *handler = nullptr;
	if (!g_deviceInfo.size())  {
		DeviceInfo *info = nullptr;
		uint32_t numDevs  = 0;
		uint32_t totalAvailDevs  = 0;
		GPUMetricHandler::InitMetricDevices(&info, &numDevs, &totalAvailDevs);
	}

	uint32_t key = CreateDeviceCode(driverId, deviceId, subdeviceId);
	auto it = g_metricHandlerMap.find(key);
	if (it == g_metricHandlerMap.end()) {
		DebugPrintError("Device <%d, %d, %d> is not a valid metrics device\n", 
				driverId, deviceId, subdeviceId);
	} else {
		 handler = (GPUMetricHandler *)it->second;
	}
	return handler;
}


/*-------------------------------------------------------------------------------------------*/
/*!
 * fn	  int GPUMetricHandler::InitMetricDevice((DeviceInfo **deviceInfo, uint32_t *numDevices,
 *                                 uint32_t *totalDevices)
 * 
 * @brief  Discover and initiate GPU Metric Device
 *
 * @param  OUT deviceInfo   -- a list of DeviceInfo objects for devices
 * @param  OUT numDevice    -- number of DeviceInfo objects for devices
 * @param  OUT totalDevice  -- total available devices including root devices and subdevices.
 *
 * @return Status.  0 for success, otherwise 1. 
 */
int GPUMetricHandler::InitMetricDevices(DeviceInfo **deviceInfo,  uint32_t *numDevices, 
					uint32_t *totalDevices)
{
	uint32_t driverCount = 0;
	int ret			  = 0;
	int retError		 = 1;
	ze_result_t status   = ZE_RESULT_SUCCESS;
	*numDevices = 0;
	*totalDevices = 0;
	*deviceInfo = nullptr;

	g_lock.lock();
	if (g_deviceInfo.size()) {
		*numDevices = g_deviceInfo.size();
		*deviceInfo = g_deviceInfo.data();
		g_lock.unlock();
		return 0;
	}

	void *dlHandle = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dlHandle) {
		DebugPrintError("dlopen libze_loader.so failed\n");
		g_lock.unlock();
		return retError;
	}

	ret = functionInit(dlHandle);
	if (ret) {
		DebugPrintError("Failed in finding functions in libze_loader.so\n");
		g_lock.unlock();
		return ret;
	}
	status = zeInitFunc(ZE_INIT_FLAG_GPU_ONLY);
	if (status == ZE_RESULT_SUCCESS) {
		status = zeDriverGetFunc(&driverCount, nullptr);
	}
	if (status != ZE_RESULT_SUCCESS || driverCount == 0) {
		g_lock.unlock();
		return retError;
	}
	vector<ze_driver_handle_t> driverList(driverCount, nullptr);
	status = zeDriverGetFunc(&driverCount, driverList.data());
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

	char *envStr =  getenv(ENABLE_RAW_LOG);
	if (envStr) {
	   g_stdout = atoi(envStr);
	}
	vector<DeviceInfo *> deviceInfoList;
	uint32_t dcount = 0;

	for (uint32_t i = 0; i < driverCount; ++i) {
		uint32_t deviceCount = 0;
		status = zeDeviceGetFunc(driverList[i], &deviceCount, nullptr);
		if (status != ZE_RESULT_SUCCESS || deviceCount == 0) {
			continue;
		}
		vector<ze_device_handle_t> deviceList(deviceCount, nullptr);
		status = zeDeviceGetFunc(driverList[i], &deviceCount, deviceList.data());
		if (status != ZE_RESULT_SUCCESS || deviceCount == 0) {
			continue;
		}

		for (uint32_t j = 0; j < deviceCount; ++j) {
			ze_device_properties_t props;
			status = zeDeviceGetPropertiesFunc(deviceList[j], &props);
			CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);
			if ((props.type != ZE_DEVICE_TYPE_GPU) || (strstr(props.name, "Intel") == nullptr)) {
				continue;
			}
			uint32_t subdeviceCount = 0;
			status = zeDeviceGetSubDevicesFunc(deviceList[j], &subdeviceCount, nullptr);
			if (status != ZE_RESULT_SUCCESS) {
				continue;
			}
			DeviceInfo node;
			node.driverId = i+1;
			node.deviceId = j+1;
			node.subdeviceId = 0;
			node.handle = 0;
			node.index = 0;
			node.numSubdevices = subdeviceCount;
			strncpy_se(node.name, MAX_STR_LEN, props.name, strlen(props.name));
			g_deviceInfo.push_back(node);
			// create metric groups for this device
			GPUMetricHandler *handler = new GPUMetricHandler((i+1), (j+1), 0);
			handler->m_driver = driverList.at(i);
			handler->m_device = deviceList.at(j);
			handler->m_numDevices = (subdeviceCount)?subdeviceCount:1;

			TMetricGroupInfo *mgroups = nullptr;
			mgroups = new TMetricGroupInfo();
			ret = InitMetricGroups(deviceList.at(j), mgroups);
			if (ret) {
				 continue;
			}
			handler->m_groupInfo = mgroups;
			uint32_t key = CreateDeviceCode((i+1), (j+1), 0);
			g_metricHandlerMap[key] = handler;
			dcount++;
			DebugPrint("detected device: <drv:%d, dev:%d>, [%s], subdevCount %d, m_dev %p\n",
				 i, j,  props.name, subdeviceCount, handler->m_device);
			if (subdeviceCount) {
				vector<ze_device_handle_t> subdeviceList(subdeviceCount, nullptr);
				status = zeDeviceGetSubDevicesFunc(deviceList[j], &subdeviceCount,
						  subdeviceList.data());
				if (status != ZE_RESULT_SUCCESS) {
					continue;
				}
				for (uint32_t k = 0; k < subdeviceCount; ++k) {
					ze_device_properties_t subprops;
					status = zeDeviceGetPropertiesFunc(subdeviceList[k], &subprops);
					GPUMetricHandler *handler = new GPUMetricHandler((i+1), (j+1), (k+1));
					handler->m_driver = driverList.at(i);
					handler->m_device = subdeviceList.at(k);
					handler->m_numDevices = 1;
					uint32_t key = CreateDeviceCode((i+1), (j+1), (k+1));
					DebugPrint("detected subdevice: <drv:%d, dev:%d, subdev:%d>, "
							"key 0x%x, name %s, m_device %p\n",
							i, j, k, key,  props.name, handler->m_device);
					g_metricHandlerMap[key] = handler;
					handler->m_groupInfo = mgroups;
					dcount++;
				}
			}
		}
	}
	*numDevices = g_deviceInfo.size();
	*deviceInfo = g_deviceInfo.data();
	*totalDevices = dcount;
#if defined(_DEBUG)
	for (uint32_t i=0; i<*numDevices; i++) {
		DeviceInfo node = g_deviceInfo.at(i);
		DebugPrint("dev[%d]: drv %d, dev %d subdev %d\n", 
				i, node.driverId, node.deviceId, node.subdeviceId);
	}
#endif

	g_lock.unlock();
	if (!*numDevices) {
		DebugPrintError("No Intel GPU metric device detected.");
		return 1;
	}

	return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * fn	  void GPUMetricHandler::DestroyMetricDevice()
 *
 * @brief  Clean up metric device
 */
void GPUMetricHandler::DestroyMetricDevice()
{
	DebugPrint("DestroyMetricDevice\n");

	m_device = nullptr;
	m_groupInfo = nullptr;

	if (m_reportData) {
		for (uint32_t i=0; i<m_numDevices; i++) {
			if (m_reportData[i].dataEntries) {
				delete m_reportData[i].dataEntries;
			}
		}
		delete m_reportData;
	}
	if (m_reportCount) {
		delete m_reportCount;
	}
	if (m_event) {
		zeEventDestroyFunc(m_event);
	}
	if (m_eventPool) {
		zeEventPoolDestroyFunc(m_eventPool);
	}
	if (m_metricStreamer) {
		zetMetricStreamerCloseFunc(m_metricStreamer);
	}
	if (m_tracer) {
		zetTracerExpDestroyFunc(m_tracer);
	}
	if (m_queryPool) {
		zetMetricQueryPoolDestroyFunc(m_queryPool);
	}
	m_driver = nullptr;
	m_device = nullptr;
}








/*------------------------------------------------------------------------------*/
/*!
 * @fn	  int GPUMetricHandler::InitMetricGroups(ze_device_handle_t device,
 *						TMetricGroupInfo *mgroups)
 *
 * @brief	 Initiate Metric Group
 *
 * @param	 IN device  -- device handle
 * @param	 INOUT mgroups -- metric group info to fill out
 *
 * @return	Status.  0 for success, otherwise 1. 
 */
int GPUMetricHandler::InitMetricGroups(ze_device_handle_t device, TMetricGroupInfo *mgroups)
{

	int ret		 = 0;
	int retError	= 1;
	if (!device) {
		return 1;
	}

	ze_result_t status	 = ZE_RESULT_SUCCESS;
	uint32_t   groupCount = 0;
	uint32_t   numMetrics = 0;
	uint32_t   eventBasedCount	= 0;
	uint32_t   timeBasedCount	 = 0;
	uint32_t   maxMetricsPerGroup = 0;
	TMetricGroupNode *groupList = nullptr;

	status = zetMetricGroupGetFunc(device, &groupCount, nullptr);
	if (status != ZE_RESULT_SUCCESS || groupCount == 0) {
		std::cout << "[WARNING] No metrics found" << std::endl;
		return retError;
	}
	groupList = new TMetricGroupNode[groupCount];
	CHECK_N_RETURN_STATUS((groupList==nullptr), 1);
	zet_metric_group_handle_t* groupHandles = new zet_metric_group_handle_t[groupCount];
	CHECK_N_RETURN_STATUS((groupHandles==nullptr), 1);
	status = zetMetricGroupGetFunc(device, &groupCount, groupHandles);
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

	for (uint32_t gid = 0; gid < groupCount; ++gid) {
		zet_metric_group_properties_t groupProps = {};
		status = zetMetricGroupGetPropertiesFunc(groupHandles[gid], &groupProps);
		CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

		uint32_t metricCount = groupProps.metricCount;
		if (metricCount > maxMetricsPerGroup) {
			maxMetricsPerGroup = metricCount;
		}
		groupList[gid].code = CreateGroupCode(gid);
		groupList[gid].props = groupProps;
		groupList[gid].handle = groupHandles[gid];
		groupList[gid].metricList = new TMetricNode[metricCount];
		CHECK_N_RETURN_STATUS(( groupList[gid].metricList ==nullptr), 1);

		DebugPrint("group[%d]: name %s, desc %s\n", gid, groupProps.name, groupProps.description);

		zet_metric_handle_t*  metricHandles = new zet_metric_handle_t[metricCount];
		CHECK_N_RETURN_STATUS((metricHandles ==nullptr), 1);
		status = zetMetricGetFunc(groupHandles[gid], &metricCount, metricHandles);
		CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);

		for (uint32_t mid = 0; mid < metricCount; ++mid) {
			zet_metric_properties_t metricProps = {};
			status = zetMetricGetPropertiesFunc(metricHandles[mid], &metricProps);
			CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), 1);
			groupList[gid].metricList[mid].props = metricProps;
			groupList[gid].metricList[mid].handle = metricHandles[mid];
			groupList[gid].metricList[mid].code = CreateMetricCode(gid,mid);
			groupList[gid].metricList[mid].metricGroupId = gid;
			groupList[gid].metricList[mid].metricId = mid;
			groupList[gid].metricList[mid].metricType =
			getMetricType(metricProps.description, metricProps.metricType);
			DebugPrint("   metric[%d][%d] name %s, desc %s, metric_type %d\n", 
					gid, mid, metricProps.name, metricProps.description,
					metricProps.metricType);
		}
		numMetrics += metricCount;
		if (groupList[gid].props.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) {
			eventBasedCount += metricCount;
		} else {
			timeBasedCount += metricCount;
		}			
		delete [] metricHandles;
	}
	DebugPrint("init metric groups return:  groupCount %d, metric %d, TBS %d, EBS %d\n",
			groupCount, numMetrics, timeBasedCount, eventBasedCount);
	delete [] groupHandles;
	mgroups->metricGroupList	= groupList;
	mgroups->numMetricGroups	= groupCount;
	mgroups->numMetrics		 = numMetrics;
	mgroups->maxMetricsPerGroup = maxMetricsPerGroup;

	return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	 int GPUMetricHandler::GetMetricInfo(int type, MetricInfo *data)
 * 
 * @brief  Get available metric info
 *
 * @param  IN  type    - metric group type,  0 for timed-based,  1 for query based
 * @param  INOUT data  - pointer to the MetricInfo data contains a list of metrics
 *
 * @reutrn		   - 0 if success. 
 */
int GPUMetricHandler::GetMetricInfo(int type, MetricInfo *data)
{
	uint32_t i	= 0;
	int	 ret  = 0;
	int	 retError = 1;
	uint32_t  stype	= 0;

	if(!m_device) {
		DebugPrintError("MetricsDevice not opened\n");
		return retError;
	}
	if (!data) {
		DebugPrintError("GetMetricInfo: invalid out data\n");
		return retError;
	}

	// get all metricGroups
	if (type) {
		stype = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED;
	} else {
		stype = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
	}
	data->code = 0;
	data->infoEntries = GPUAllocMetricInfo(m_groupInfo->numMetricGroups);
	CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
	int index = 0;
	TMetricGroupNode  *groupList = m_groupInfo->metricGroupList;
	for (i = 0; i < m_groupInfo->numMetricGroups; i++ ) {
		if (groupList[i].props.samplingType & stype) {
			strncpy_se(data->infoEntries[index].name, MAX_STR_LEN,
					groupList[i].props.name,
					strlen(groupList[i].props.name));
			strncpy_se(data->infoEntries[index].desc, MAX_STR_LEN,
					groupList[i].props.description,
					strlen(groupList[i].props.description));
			data->infoEntries[index].numEntries = groupList[i].props.metricCount;
			data->infoEntries[index].dataType = groupList[i].props.samplingType;
			data->infoEntries[index++].code = groupList[i].code;
		}
		data->numEntries = index;
	}
	return ret;
}

/*------------------------------------------------------------------------------*/
/**
 * @fn	 int GPUMetricHandler::GetMetricInfo(const char *name, int type, MetricInfo *data)
 *
 * @brief  Get available metrics in a certain metrics group. 
 *		 If metric group is not specified, get all available metrics from all metric groups.
 *
 * @param  IN  name  -- metric group name.  If nullptr or empty, means all metric groups
 * @param  IN  type  -- metric group type,  0 for timed-based,  1 for query based
 * @param  INOUT data  -- pointer to the MetricInfo data contains a list of metrics
 *
 * @reutrn		   -- 0 if success. 
 */
int GPUMetricHandler::GetMetricInfo(const char *name, int type, MetricInfo *data)
{
	uint32_t i = 0;
	uint32_t numMetrics = 0;
	int ret	 = 0;
	int retError = 1;
	 uint32_t  stype = (type)? ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED 
					: ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
	if(!m_device) {
		DebugPrintError("MetricsDevice not opened\n");
		return retError;
	}
	if (!data) {
		DebugPrintError("GetMetricInfo: invalid out data\n");
		return retError;
	}

	// get all metricGroups
	numMetrics = m_groupInfo->numMetrics;
	int selectedAll = 0;
	if (!name || !strlen(name)) {
		data->infoEntries = GPUAllocMetricInfo(numMetrics);
		CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
		data->code = 0;
		selectedAll = 1;
	}
	int index = 0;
	TMetricGroupNode *mGroup;
	for (i = 0; i < m_groupInfo->numMetricGroups; i++ ) {
		mGroup = &(m_groupInfo->metricGroupList[i]);
		if (!(mGroup->props.samplingType & stype) || strlen(mGroup->props.name) == 0) {
			continue;
		}
		if (!selectedAll) {
			if (strncmp(name, mGroup->props.name, MAX_STR_LEN) != 0) {
				 continue;
			}
			numMetrics = mGroup->props.metricCount;
			data->infoEntries = GPUAllocMetricInfo(numMetrics);
			CHECK_N_RETURN_STATUS((data->infoEntries==nullptr), 1);
			m_groupId = i;
			data->code = mGroup->code;
			index = 0;
		}
		for (uint32_t j=0; j<mGroup->props.metricCount; j++) {
			 strncpy_se((char *)(data->infoEntries[index].name), MAX_STR_LEN,
					   mGroup->props.name, strlen(mGroup->props.name));
			 size_t mnLen = strlen(data->infoEntries[index].name);
			 if ((mnLen+3) < MAX_STR_LEN) {
				data->infoEntries[index].name[mnLen++] = '.';
				strncpy_se((char *)&(data->infoEntries[index].name[mnLen]), (MAX_STR_LEN-mnLen),
				mGroup->metricList[j].props.name,
				strlen(mGroup->metricList[j].props.name));
			 }
			 strncpy_se(data->infoEntries[index].desc, MAX_STR_LEN,
					mGroup->metricList[j].props.description,
					strlen(mGroup->metricList[j].props.description));
			 data->infoEntries[index].code = mGroup->metricList[j].code;
			 data->infoEntries[index].numEntries = 0;
			 data->infoEntries[index].dataType = mGroup->metricList[j].props.resultType;
			 data->infoEntries[index].metricType = mGroup->metricList[j].metricType;
			 index++;
		 }
		 if (!selectedAll) {
			 break;
		 }
	}
	if ( !selectedAll && (i == m_groupInfo->numMetricGroups)) {
		DebugPrintError( "GetMetricInfo: metricGroup %s is not found, abort\n", name);
		return retError;
	} 
	data->numEntries = index;
	return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn	   int GPUMetricHandler::GetMetricCode(
 *					   const char *mGroupName, const char *metricName, uint32_t mtype,
 *						uint32_t *mGroupCode,  uint32_t *metricCode)
 * 
 * @ brief   Get metric code and metric group code for a given valid metric group name and 
 *           metric name.
 *
 * @param	IN  mGroupName  - Metric group name
 * @param	IN  metricName  - metric name in the 
 * @param	IN  mtype	    - metric type:  < time_based,  event_based>
 * @param	OUT mGroupCode  - metric group code,  0 if the metric group dose not exist
 * @param	OUT metricCode  - metric code,  0 if the metric dose not exist
 *
 * @return   Status,  0 if success, 1 if no such metric or metric group exist.
 */
int GPUMetricHandler::GetMetricCode(
	 const char *mGroupName, const char *metricName, uint32_t mtype,
	 uint32_t *mGroupCode,  uint32_t *metricCode)
{
	int ret	   = 0;
	int retError  = 1;

	DebugPrint( "GetMetricCode: metricGroup %s, metric %s == ", mGroupName, metricName);

	int metricType = (mtype)?ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED
				: ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
	TMetricGroupNode *groupList = m_groupInfo->metricGroupList;
	for (uint32_t i=0; i< m_groupInfo->numMetricGroups; i++) {
		if (groupList[i].code												 &&
			(metricType & groupList[i].props.samplingType)					 &&
			(strncmp(mGroupName,  groupList[i].props.name, MAX_STR_LEN) == 0 )) {
			 *mGroupCode = groupList[i].code;
			if (!metricName || (strlen(metricName)==0) || !metricCode) {
				DebugPrint( " mGroupCode 0x%x \n", *mGroupCode);
				return ret;
			}
			for (uint32_t j=0; j< groupList[i].props.metricCount; j++) {
				if (strncmp(metricName, groupList[i].metricList[j].props.name,MAX_STR_LEN)==0) {
					*metricCode = groupList[i].metricList[j].code;
					DebugPrint( "  mGroupCode 0x%x , metricCode 0x%x\n", *mGroupCode, *metricCode);
					return ret;
				}
			}
		}
	}
	return retError;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	  int GPUMetricHandler::EnableMetricGroup(
 *				 const char *metricGroupName, uint32_t mtype, int *enableSt)
 *
 * @brief   Enable a named metric group for collection
 *
 * @param   IN metricGroupName - the metric group name
 * @param   IN mtype		   - metric group type
 * @param   OUT enableSt      - if the metric group already enaled, otherwise 0
 *
 * @return  Status, 0 for success.
 */
int
GPUMetricHandler::EnableMetricGroup(const char *metricGroupName, uint32_t mtype, int *enableSt)
{
	int		  ret		= 0;
	int		  retError   = 1;
	uint32_t groupCode = 0;

	if ((ret=GetMetricCode(metricGroupName, "", mtype, &groupCode, nullptr))) {
		DebugPrintError("MetricGroup %s is not found\n", metricGroupName);
		return retError;
	}
	return EnableMetricGroup(groupCode, mtype, enableSt);
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	  int GPUMetricHandler::EnableMetricGroup(
 *				 const char *metricGroupName, uint32_t mtype, int *enableSt)
 *
 * @brief   Enable a named metric group for collection
 *
 * @param   IN metricGroupName - the metric group name
 * @param   IN mtype		   - metric group type
 * @param   OUT enableSt       - 1 if the metrics group is already enabled, otherwise 0.
 *
 * @return  Status, 0 for success.
 */
int
GPUMetricHandler::EnableMetricGroup(uint32_t groupCode, uint32_t mtype, int *enableSt) {

	int		  ret		= 0;
	int		  retError   = 1;
	ze_result_t  status	 = ZE_RESULT_SUCCESS;

	*enableSt = 0;
	if ((groupCode-1) >= m_groupInfo->numMetricGroups)  {
		DebugPrintError("MetricGroup code 0x%x is not found\n", groupCode);
		return retError;
	}

	m_lock.lock();
	if ( m_status == COLLECTION_CONFIGED) {
		DebugPrint( "EnableMetricGroup: already in enable status\n");
		*enableSt = 1;
		m_lock.unlock();
		return ret;
	}

	m_groupId = groupCode-1;
	if (mtype == EVENT_BASED) {
		m_groupType	= ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED;
	} else {
		m_groupType	= ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED;
	}
	DebugPrint("EnableMetricGroup: code 0x%x, type 0x%x\n", groupCode, m_groupType);

	TMetricGroupNode  *groupList =  m_groupInfo->metricGroupList;
	uint32_t metricCount = groupList[m_groupId].props.metricCount;
	if (!metricCount) {
		DebugPrintError("MetricGroup dose not have metrics\n");
		m_lock.unlock();
		return retError;
	}
	if (!m_context) {
		ze_context_desc_t ctxtDesc = {
			ZE_STRUCTURE_TYPE_CONTEXT_DESC,
			nullptr, 
			0
		}; 
		zeContextCreateFunc(m_driver, &ctxtDesc, &m_context);
	}

	// activate metric group
	zet_metric_group_handle_t mGroup = groupList[m_groupId].handle;
	status = zetContextActivateMetricGroupsFunc(m_context, m_device, 1, &mGroup);
	if (status != ZE_RESULT_SUCCESS) {
		DebugPrintError("ActivateMetricGroup code 0x%x failed.\n", groupCode);
		return retError;
	}

	m_eventPool	= nullptr;
	m_event		= nullptr;
	m_tracer	   = nullptr;
	// create buffer for report data
	if (!m_reportData) {
		m_reportData = new MetricData[m_numDevices];
		m_reportCount = new uint32_t[m_numDevices];
		CHECK_N_RETURN_STATUS((m_reportData==nullptr), 1);
	}
	for (uint32_t i=0; i<m_numDevices; i++) {
		m_reportCount[i]= 0;
		m_reportData[i].grpCode	  = m_groupId + 1;
		m_reportData[i].numEntries   = metricCount;
		m_reportData[i].dataEntries  = new DataEntry[metricCount];
		CHECK_N_RETURN_STATUS((m_reportData[i].dataEntries==nullptr), 1);
		for( uint32_t j = 0; j < metricCount; j++ ) {
			m_reportData[i].dataEntries[j].type = 0;
			m_reportData[i].dataEntries[j].value.ival = 0;
			m_reportData[i].dataEntries[j].code = groupList[m_groupId].metricList[j].code;
		}
	}
	m_status = COLLECTION_CONFIGED;
	m_lock.unlock();
	return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	  int GPUMetricHandler::EnableTimeBasedStream(
 *					 uint32_t timePeriod, uint32_t numReports)
 *
 * @brief   Enable a named metric group for collection
 *
 * @param   IN  timePeriod	- The timer period for sampling
 * @param   OUT numReports	- total number of sample to be collected (not used).
 *							  Default is to collect all sample record available
 *
 * @return  Status, 0 for success.
 */
int GPUMetricHandler::EnableTimeBasedStream(uint32_t timePeriod, uint32_t numReports)
{
	int		  ret		= 0;
	int		  retError   = 1;
	ze_result_t  status	 = ZE_RESULT_SUCCESS;

	if (m_groupId < 0)  {
		DebugPrintError("No metrics enabled. Data collection abort\n");
		return retError;
	}
	m_lock.lock();
	if (m_status  == COLLECTION_ENABLED) {
		DebugPrint( "EnableTimeBaedStream: already enabled\n");
		m_lock.unlock();
		return ret;
	}

	ze_event_pool_desc_t  eventPoolDesc;
	eventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
	eventPoolDesc.pNext = nullptr;
	eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
	eventPoolDesc.count = 100;
	status = zeEventPoolCreateFunc(m_context, &eventPoolDesc, 1, &m_device, &m_eventPool);
	ze_event_desc_t eventDesc;
	eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
	eventDesc.index   = 0;
	eventDesc.signal  = ZE_EVENT_SCOPE_FLAG_HOST;
	eventDesc.wait	= ZE_EVENT_SCOPE_FLAG_HOST;
	if (status == ZE_RESULT_SUCCESS) {
		status = zeEventCreateFunc(m_eventPool, &eventDesc, &m_event);
	}
	if (status == ZE_RESULT_SUCCESS) {
		zet_metric_streamer_desc_t  metricStreamerDesc;
		metricStreamerDesc.stype = ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC;
		if (numReports == 0) {
			metricStreamerDesc.notifyEveryNReports = MAX_REPORTS;
		} else {
			metricStreamerDesc.notifyEveryNReports = numReports;
		}
		metricStreamerDesc.samplingPeriod	  = timePeriod;
		zet_metric_group_handle_t mGroup = m_groupInfo->metricGroupList[m_groupId].handle;
		status = zetMetricStreamerOpenFunc(m_context, m_device, mGroup,
									&metricStreamerDesc, m_event, &m_metricStreamer);
		if (!m_metricStreamer) {
			DebugPrintError("zetMetricStreamerOpen: failed with null streamer on device [%p]\n", m_device);
		}
	}
	if (status == ZE_RESULT_SUCCESS) {
		ret = 0;
		m_status = COLLECTION_ENABLED;
	} else {
		DebugPrintError("EnableTimeBasedStream: failed on device [%p], status 0x%x\n", 
						m_device, status);
		status = zetContextActivateMetricGroupsFunc(m_context, m_device, 0, nullptr);
		m_status = COLLECTION_INIT;
		ret = 1;
	}
	m_lock.unlock();
	return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	  int GPUMetricHandler::EnableEventBasedQuery()
 *
 * @brief   Enable metric query on enabled metrics
 *
 * @return  Status, 0 for success.
 */
int GPUMetricHandler::EnableEventBasedQuery()
{
	ze_result_t  status	 = ZE_RESULT_SUCCESS;
	int		  ret		= 0;
	int		  retError   = 0;

	if (m_groupId < 0)  {
		DebugPrintError("No metrics enabled. Data collection abort\n");
		return retError;
	}
	m_lock.lock(); 
	if (m_status  == COLLECTION_ENABLED) {
		DebugPrint( "EnableEventBaedQuery: already enabled\n");
		m_lock.unlock();
		return ret;
	}

	zet_metric_group_handle_t mGroup = m_groupInfo->metricGroupList[m_groupId].handle;
	m_queryState = new QueryState;
	zet_metric_query_pool_desc_t metricQueryPoolDesc;
	if (m_context) {
		ze_context_desc_t ctxtDesc = {
			ZE_STRUCTURE_TYPE_CONTEXT_DESC,
			nullptr,
			0
		};
		zeContextCreateFunc(m_driver, &ctxtDesc, &m_context);
	}
	metricQueryPoolDesc.stype = ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
	metricQueryPoolDesc.type = ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE;
	metricQueryPoolDesc.count = MAX_KERNERLS;
	status = zetMetricQueryPoolCreateFunc(m_context, m_device, mGroup,
			&metricQueryPoolDesc, &m_queryPool);
	if (status == ZE_RESULT_SUCCESS) {
		m_queryState->queryPool = m_queryPool;
		ze_event_pool_desc_t  eventPoolDesc;
		eventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		eventPoolDesc.flags= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		eventPoolDesc.count =  MAX_KERNERLS;

		// create event to wait
		status = zeEventPoolCreateFunc(m_context, &eventPoolDesc, 1, &m_device, &m_eventPool);
	}
	zet_tracer_exp_desc_t tracerDesc;
	tracerDesc.stype = ZET_STRUCTURE_TYPE_TRACER_EXP_DESC;
	if (status == ZE_RESULT_SUCCESS) {
		m_queryState->eventPool = m_eventPool;
		m_queryState->handle = this;
		tracerDesc.pUserData = m_queryState;
		status = zetTracerExpCreateFunc(m_context, &tracerDesc, &m_tracer);
	}
	zet_core_callbacks_t prologCB = {};
	zet_core_callbacks_t epilogCB = {};

	if (status == ZE_RESULT_SUCCESS) {
		epilogCB.Kernel.pfnCreateCb = kernelCreateCB;
		epilogCB.Kernel.pfnDestroyCb = kernelDestroyCB;
		prologCB.CommandList.pfnAppendLaunchKernelCb = metricQueryBeginCB;
		epilogCB.CommandList.pfnAppendLaunchKernelCb = metricQueryEndCB;

		status = zetTracerExpSetProloguesFunc(m_tracer, &prologCB);
	}
	if (status == ZE_RESULT_SUCCESS) {
		status = zetTracerExpSetEpiloguesFunc(m_tracer, &epilogCB);
	}
	if (status == ZE_RESULT_SUCCESS) {
		status = zetTracerExpSetEnabledFunc(m_tracer, true);
	}
	if (status == ZE_RESULT_SUCCESS) {
		m_status = COLLECTION_ENABLED;
		ret  = 0;
	} else {
		DebugPrintError("EnableEventBasedQuery: failed with status 0x%x, abort.\n", status);
		status = zetContextActivateMetricGroupsFunc(m_context, m_device, 0, nullptr);
		m_status = COLLECTION_INIT;
		ret  = retError;
	}
	m_lock.unlock();
	return ret;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn		 void GPUMetricHandler::DisableMetricGroup()
 * @brief	  Disable current metric group
 *			 After metric group disabled, cannot read data anymore
 * 
 */
void
GPUMetricHandler::DisableMetricGroup()
{
	DebugPrint("enter DisableMetricGroup()\n");

	m_lock.lock();
	if ((m_status != COLLECTION_ENABLED) && (m_status != COLLECTION_CONFIGED)) {
	   m_lock.unlock();
	   return;
	}
	m_status = COLLECTION_DISABLED;

	zetContextActivateMetricGroupsFunc(m_context,  m_device, 0, nullptr);
	m_lock.unlock();
	return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	 MetricData * GPUMetricHandler::GetMetricData(
 *					   uint32_t  mode, uint32_t *numReports)
 *
 * @brief  Read raw event data and calculate metrics.
 *         Return an array of MetricData as the overall aggregated metrics data for a device.
		   One MetricData pre device or subdevice.
 *
 *  @param In  mode			 - report mode, summary or samples
 *  @param OUT numReports	 - total number of sample reports
 *
 *  @return					 - metric data array, one MetricData per report.
 */
MetricData *
GPUMetricHandler::GetMetricData(uint32_t mode, uint32_t *numReports)
{
	uint8_t	*rawBuffer  = nullptr;
	size_t	rawDataSize = 0;
	MetricData *reportArray = nullptr;
	*numReports = 0;

	m_lock.lock();
	if  (m_groupType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED) {
		rawBuffer = ReadStreamData(&rawDataSize);
		if (rawBuffer) {
			GenerateMetricData(rawBuffer, rawDataSize,  mode);
			delete [] rawBuffer;
		}
	} else {
		m_queryState->lock.lock();
		vector<QueryData>  qList;
		for (auto query : m_queryState->queryList) {
			rawDataSize = 0;
			rawBuffer = nullptr;
			ze_result_t status = ZE_RESULT_SUCCESS;
			rawBuffer = ReadQueryData(query, &rawDataSize, &status);
			if (status != ZE_RESULT_SUCCESS) {
				DebugPrintError("query read failed: status 0x%x\n", status);
			}
			if (status == ZE_RESULT_NOT_READY) {
				qList.push_back(query);
			} 
			if ((status == ZE_RESULT_SUCCESS) && rawBuffer) {
				GenerateMetricData(rawBuffer, rawDataSize,  mode);
				delete [] rawBuffer;
			} 
		}
		m_queryState->queryList.clear();
		for (auto query : qList) {
			 m_queryState->queryList.push_back(query);
		}		
		m_queryState->lock.unlock();
	}

	// only SUMMARY mode
	TMetricGroupNode  *groupList = m_groupInfo->metricGroupList;
	TMetricNode *metricList = groupList[m_groupId].metricList;
	uint32_t metricCount = groupList[m_groupId].props.metricCount;

	reportArray = GPUAllocMetricData(1, m_numDataSet, metricCount);
	if (!reportArray)  {
		m_lock.unlock();
		return reportArray;
	}
	*numReports = 1;
	MetricData  *reportData  = &(reportArray[0]);

	int index = 0;
	for (uint32_t i=0; i<m_numDataSet; i++) {
		if (!m_reportCount[i]) {
			reportData->dataSetStartIdx[i] = -1;
			continue;
		}
		reportData->dataSetStartIdx[i] = index;
		for (uint32_t j=0; j < metricCount; j++, index++) {
			reportData->dataEntries[index].code = m_reportData[i].dataEntries[j].code;
			reportData->dataEntries[index].type = m_reportData[i].dataEntries[j].type;
			if (metricList[j].metricType != M_AVERAGE) {
				reportData->dataEntries[index].value.ival =
						m_reportData[i].dataEntries[j].value.ival;
			} else {
				if (m_reportData[i].dataEntries[j].type) {  // calculate avg
					if (m_reportData[i].dataEntries[j].value.fpval != 0.0) {
						reportData->dataEntries[index].value.fpval =
							  (m_reportData[i].dataEntries[j].value.fpval)/m_reportCount[i];
					}
				} else {
					reportData->dataEntries[index].value.ival =
							 (m_reportData[i].dataEntries[j].value.ival)/m_reportCount[i];
				}
			}
		}
	}
	m_lock.unlock();
	return reportData;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	int GPUMetricHandler::GetCurGroupCode()
 *
 * @brief Return the metric group code of current activated metric group
 *
 * @return Out   -- metric group code
 */
uint32_t
GPUMetricHandler::GetCurGroupCode() {
	return (uint32_t)(m_groupId + 1);
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	int GPUMetricHandler::SetControl(uint32_t mode)
 * 
 * @brief Set control 
 *
 * @param IN mode  -- control mode to set
 */
int
GPUMetricHandler::SetControl(uint32_t mode) {

	int ret = 0;
	m_lock.lock();
	if (mode & METRIC_RESET) {
		if (m_numDevices && m_reportData) {
			for (uint32_t i=0; i<m_numDevices; i++) {
				MetricData *data = &m_reportData[i];
				for (uint32_t j=0; j<data->numEntries; j++) {
					data->dataEntries[j].value.ival = 0;
				}
				m_reportCount[i] = 0;
			}
		}
	}
	m_lock.unlock();
	return ret;
}

/*------------------------------------------------------------------------------*/
/*!
 * fn	   string  GPUMetricHandler::GetDeviceName(ze_device_handle_t device)
 * 
 * @brief   Get metric device name
 *
 * @parapm  In device   Device handler
 *
 * @return  The device name
 */
string  GPUMetricHandler::GetDeviceName(ze_device_handle_t device)
{
	ze_result_t status = ZE_RESULT_SUCCESS;
	ze_device_properties_t props;
	status = zeDeviceGetPropertiesFunc(device, &props);
	if (status == ZE_RESULT_SUCCESS) {
		return props.name;
	}
	return "";
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	void GPUMetricHandler::GenerateMetricData(
 *					   uint8_t *rawBuffer, size_t rawDataSize, uint32_t mode) 
 *
 * @brief Calculate metric data from raw event data, the result will be aggrated into global data
 *
 * @param IN  rawBuffer	    - buffer for raw event data
 * @param IN  rawDataSize   - the size of raw event data in the buffer
 * @param IN  mode		    - report mode
 *
 */
void
GPUMetricHandler::GenerateMetricData(
	uint8_t	 *rawBuffer,
	size_t	 rawDataSize,
	uint32_t mode
)
{
	if (!rawDataSize) {
		return;
	}

	ze_result_t status = ZE_RESULT_SUCCESS;
	TMetricGroupNode  *groupList = m_groupInfo->metricGroupList;
	zet_metric_group_handle_t mGroup = groupList[m_groupId].handle;
	uint32_t metricCount = groupList[m_groupId].props.metricCount;
	uint32_t numSets = 0;
	uint32_t numValues = 0;

	status = zetMetricGroupCalculateMultipleMetricValuesExpFunc(mGroup,
			ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
			(size_t)rawDataSize, (uint8_t*)rawBuffer,
			&numSets, &numValues, nullptr,  nullptr);

	if ((status != ZE_RESULT_SUCCESS) || !numSets || !numValues) {
		DebugPrint("Metrics calculation,  failed on allocating memory space.\n");
		return;
	}
	std::vector<uint32_t> metricSetCounts(numSets);
	std::vector<zet_typed_value_t> metricValues(numValues);

	status = zetMetricGroupCalculateMultipleMetricValuesExpFunc(mGroup,
			ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
			(size_t)rawDataSize, (uint8_t*)rawBuffer,
			&numSets, &numValues,
		metricSetCounts.data(),  metricValues.data());
	if (status != ZE_RESULT_SUCCESS)  {
		DebugPrintError("Failed on metrics calculation\n");
		return;
	}
	if (!numSets || !numValues) {
		DebugPrintError("No metrics calculated.\n");
		return;
	}
	zet_typed_value_t *typedDataList = metricValues.data();
	m_numDataSet = numSets;

	uint32_t index = 0;
	for (uint32_t i=0; i<numSets; i++) {
		ProcessMetricDataSet(i, typedDataList, index, metricSetCounts[i], metricCount, mode);
		index += metricSetCounts[i];
	}
	return;
}

/*------------------------------------------------------------------------------*/
/*!
 * @fn	void GPUMetricHandler::ProcessMetricDataSet(uint32_t dataSetId, 
 *				zet_typed_value_t *typedData, uint32_t startIdx, uint32_t metricDataSize,
 *				uint32_t metricCount, uint32_t mode)
 *
 * @brief Process data set which corresponding to a device/subdevice
 *
 * @param IN  dataSetId	    - given data set index
 * @param IN  typedData	    - typed data array
 * @param IN  startIdxd	    - start index in the set
 * @param IN  metricDataSize- given metric data size
 * @param IN  metricCount	- metric count in the group
 * @param IN  mode		    - control mode, i.e, RESET after read or not.
 *
 */
void
GPUMetricHandler::ProcessMetricDataSet(
	uint32_t		   dataSetId,
	zet_typed_value_t *typedData,
	uint32_t		   startIdx,
	uint32_t		   metricDataSize,
	uint32_t		   metricCount,
	uint32_t		   mode
)
{
	TMetricGroupNode  *groupList = m_groupInfo->metricGroupList;
	TMetricNode *metricList = groupList[m_groupId].metricList;

	if (dataSetId > m_numDevices) {
		return;
	}

	MetricData *reportData = &(m_reportData[dataSetId]);
	uint32_t reportCounts = metricDataSize/metricCount;
	if (mode & METRIC_RESET) {
		for (uint32_t j=0; j<metricCount; j++) {
			reportData->dataEntries[j].value.ival = 0;
		}
		m_reportCount[dataSetId] = 0;
	}

	DebugPrint("data[%d], metricDataSize %d, reportCounts %d, metricCount %d\n", 
		(int)dataSetId, (int)metricDataSize, (int)reportCounts, (int)metricCount);

	// log metric names
	if (g_stdout) {
		for (uint32_t j=0; j<metricCount; j++) {
			 printf("%s, ", metricList[j].props.name);
		}
		printf("\n");
		printf("metricDataSize 0x%x, reportCounts %d\n", metricDataSize, m_reportCount[dataSetId]);
		printf("\n");
	}
	uint32_t  sidx = startIdx;
	for (uint32_t i=0; i < reportCounts; i++, sidx+= metricCount) {
		for (uint32_t j=0; j<metricCount; j++) {
			 int dtype = 0;
			 long long iVal = 0; 
			 double fpVal = 0.0; 

			 typedValue2Value(typedData[sidx + j],
							 &m_dataDump, g_stdout, &dtype, &iVal, &fpVal, (j==(metricCount-1)));
			 if (metricList[j].metricType != M_RAW) {
				 reportData->dataEntries[j].type = dtype;
				 if (!dtype) {
					 reportData->dataEntries[j].value.ival += iVal;
				 } else {
					 reportData->dataEntries[j].value.fpval += fpVal;
				 }
			 } else {
				// static value, only need to take last one
				if (i== (reportCounts -1)) {
					if (!dtype)  {
						reportData->dataEntries[j].value.ival = iVal;
					} else {
						reportData->dataEntries[j].value.fpval = fpVal;
					}
				 }
			 }
		}
	}
	m_reportCount[dataSetId] += reportCounts;
	return;
}


/*------------------------------------------------------------------------------*/
/*!
 * @fn		uint8_t * GPUMetricHandler::ReadStreamData(size_t *rawDataSize)
 *
 * @brief	  read raw time based sampling data
 *
 *  @param OUT rawDataSize	- total byte size of raw data read.
 *
 *  @return	buffer pointer  - buffer contains the raw data, nulltpr if failed.
 */
uint8_t *
GPUMetricHandler::ReadStreamData(size_t *rawDataSize)
{
	ze_result_t status  = ZE_RESULT_SUCCESS;
	size_t	  rawSize = 0;
	uint8_t	*rawBuffer = nullptr;

   *rawDataSize = 0;
	//read raw data
	status = zeEventHostSynchronizeFunc(m_event, 50000 /* wait delay in nanoseconds */);
 
	status = zetMetricStreamerReadDataFunc(m_metricStreamer, UINT32_MAX,
										  &rawSize, nullptr);
	if (status !=  ZE_RESULT_SUCCESS) {
		DebugPrintError("ReadStreamData failed, status 0x%x, rawSize %d\n", status, (int)rawSize);
		return nullptr;
	}

	rawBuffer = new uint8_t[rawSize];
	CHECK_N_RETURN_STATUS((rawBuffer==nullptr), nullptr);

	if (!rawSize) {
	   *rawDataSize = rawSize;
		return rawBuffer;
	}
	status = zetMetricStreamerReadDataFunc(m_metricStreamer, UINT32_MAX,
								   &rawSize, (uint8_t *)rawBuffer);
	if (status !=  ZE_RESULT_SUCCESS) {
		DebugPrintError("ReadStreamData failed, status 0x%x\n", status);
		delete [] rawBuffer;
		return nullptr;
	}
	if (!rawSize)  {
		// thi may not be an error, especially in multi-thread case.
		DebugPrint("No Raw Data available. This could be collection time too short "
				"or buffer overflow. Please increase the sampling period\n");
	}

   *rawDataSize = rawSize;
	return rawBuffer;
}

/*------------------------------------------------------------------------------*/
/*!
 *  @fn		uint8_t * GPUMetricHandler::ReadQueryData(
 *							QueryData &data,
 *							size_t *rawDataSize,
 *							ze_result_t * status)
 *
 *  @brief	 read raw query based sampling data
 *
 *  @param IN  data	          - given query data
 *  @param OUT rawDataSize	  - total # of bytes of raw data read.
 *  @param OUT retStatus	  - return status
 *
 *  @return	buffer pointer  - buffer contains the raw data, nulltpr if failed.
 */
uint8_t *
GPUMetricHandler::ReadQueryData(QueryData &data, size_t *rawDataSize, ze_result_t *retStatus)
{

	size_t	  rawSize = 0;
   *rawDataSize = 0;

	std::map<std::string, MetricData> dataMap;
	ze_result_t status = ZE_RESULT_SUCCESS;
	status = zeEventHostSynchronizeFunc(data.event, 1000);
	*retStatus = status;
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);
	status = zeEventDestroyFunc(data.event);
	*retStatus = status;
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

	status = zetMetricQueryGetDataFunc(data.metricQuery, &rawSize, nullptr);
	*retStatus = status;
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

	uint8_t *rawBuffer = new uint8_t[rawSize];
	CHECK_N_RETURN_STATUS((rawBuffer==nullptr), nullptr);
	status = zetMetricQueryGetDataFunc(data.metricQuery, &rawSize, rawBuffer);
	*retStatus = status;
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

	status = zetMetricQueryDestroyFunc(data.metricQuery);
	*retStatus = status;
	CHECK_N_RETURN_STATUS((status!=ZE_RESULT_SUCCESS), nullptr);

	*rawDataSize = rawSize;
	
	return rawBuffer;
}


