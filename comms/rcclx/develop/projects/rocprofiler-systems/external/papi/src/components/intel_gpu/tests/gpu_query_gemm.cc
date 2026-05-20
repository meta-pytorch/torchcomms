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
 */

/*
 *  This test case tests event based (query) data collection on Intel GPU performance metrics
 *
 *  @ brief Collect  metric data for offload kernel "gemm"
 *
 */
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include  <level_zero/ze_api.h>

#include "gpu_common_utils.h"

#if defined(ENABLE_PAPI)
#include "papi.h" 
const char *env_str = "ZET_ENABLE_API_TRACING_EXP=1";
#endif


#define A_VALUE 0.128f
#define B_VALUE 0.256f
#define MAX_EPS 1.0e-4f

#define NSEC_IN_SEC  1000000000

#define MAX_STR_LEN     128
#define MAX_BUFF_SIZE   256

#define ALIGN 64

using namespace std;

#define CHECK_N_MSG_EXIT(status, msg, retVal) {                                         \
        if (status) {                                                                   \
            if (retVal) { fprintf(stderr, "%s, PAPI return status %d\n", msg, retVal);  \
            } else  { fprintf(stderr, "%s\n", msg); }                                   \
            PAPI_shutdown();                                                            \
            exit(-1); } }

inline string GetExecutablePath() {
    char buffer[MAX_BUFF_SIZE] = { 0 };
    ssize_t status = readlink("/proc/self/exe", buffer, MAX_BUFF_SIZE);
    assert(status > 0);
    string path(buffer);
    return path.substr(0, path.find_last_of("/\\") + 1);
}

inline 
vector<uint8_t> LoadBinaryFile(const string& path) {
    vector<uint8_t> binary;
    ifstream stream(path, ios::in | ios::binary);
    if (!stream.good()) {
        return binary;
    }
    size_t size = 0;
    stream.seekg(0, ifstream::end);
    size = static_cast<size_t>(stream.tellg());
    stream.seekg(0, ifstream::beg);
    if (size == 0) {
      return binary;
    }
    binary.resize(size);
    stream.read(reinterpret_cast<char *>(binary.data()), size);
    return binary;
}

inline string GetDeviceName(ze_device_handle_t device) {
    assert(device != nullptr);
    ze_result_t status = ZE_RESULT_SUCCESS;
    ze_device_properties_t props;
    status = zeDeviceGetProperties(device, &props);
    assert(status == ZE_RESULT_SUCCESS);
    return props.name;
}

inline void GetIntelDeviceAndDriver(ze_device_type_t type,
		                    uint32_t dev_idx, uint32_t tile_idx,
                                    ze_device_handle_t& device,
                                    ze_driver_handle_t& driver) {
    ze_result_t status = ZE_RESULT_SUCCESS;
  
    uint32_t driver_count = 0;
    status = zeDriverGet(&driver_count, nullptr);
    if (status != ZE_RESULT_SUCCESS || driver_count == 0) {
        return;
    }
    vector<ze_driver_handle_t> driver_list(driver_count, nullptr);
    status = zeDriverGet(&driver_count, driver_list.data());
    assert(status == ZE_RESULT_SUCCESS);
  
    for (uint32_t i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        status = zeDeviceGet(driver_list[i], &device_count, nullptr);
        if (status != ZE_RESULT_SUCCESS || device_count == 0) {
            continue;
        }
        vector<ze_device_handle_t> device_list(device_count, nullptr);
        status = zeDeviceGet(driver_list[i], &device_count, device_list.data());
        assert(status == ZE_RESULT_SUCCESS);
 
        for (uint32_t j = 0; j < device_count; ++j) {
            ze_device_properties_t props;
            status = zeDeviceGetProperties(device_list[j], &props);
            assert(status == ZE_RESULT_SUCCESS);
  
            if (props.type == type && strstr(props.name, "Intel") != nullptr) {
                if (dev_idx != j) {
                    continue;
				}
				uint32_t subdevice_count = 0;
				zeDeviceGetSubDevices(device_list[j], &subdevice_count, nullptr);
				// select a tile on a  multipl tiles device
				if (subdevice_count && tile_idx && (tile_idx <= subdevice_count)) {
					vector<ze_device_handle_t> subdevice_list(subdevice_count, nullptr);
					status = zeDeviceGetSubDevices(device_list[j], &subdevice_count,
							subdevice_list.data());
							device=subdevice_list.at(tile_idx-1);
				} else {
					device = device_list[j];
				}
                driver = driver_list[i];
                break;
            }
        }
    }
    return;
}

static void 
RunKernel(ze_kernel_handle_t kernel,
                         ze_device_handle_t device,
                         ze_context_handle_t context, 
                         const vector<float>& a,
                         const vector<float>& b,
                         vector<float>& c,
                         int size) 
{
    assert(kernel != nullptr);
    assert(device != nullptr);
    assert(context != nullptr);
    
    assert(size > 0);
    int array_size = size * size;
    assert(a.size() == static_cast<size_t>(array_size));
    assert(b.size() == static_cast<size_t>(array_size));
    assert(c.size() == static_cast<size_t>(array_size));

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t group_size[3] = { 0 };
    status = zeKernelSuggestGroupSize(kernel, size, size, 1,
             &(group_size[0]), &(group_size[1]), &(group_size[2]));
    assert(status == ZE_RESULT_SUCCESS);

    if ((size % group_size[0]) != 0 || (size % group_size[1]) != 0) {
        cout << "Non-uniform workgroups are not supported" << endl;
        return;
    }
    void* dev_a = nullptr;
    ze_device_mem_alloc_desc_t alloc_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    status = zeMemAllocDevice(context, &alloc_desc, size * size * sizeof(float),
                              ALIGN, device, &dev_a);
    assert(status == ZE_RESULT_SUCCESS);
    void* dev_b = nullptr;
    status = zeMemAllocDevice(context, &alloc_desc, size * size * sizeof(float),
                              ALIGN, device, &dev_b);
    assert(status == ZE_RESULT_SUCCESS);
    void* dev_c = nullptr;
    status = zeMemAllocDevice(context, &alloc_desc, size * size * sizeof(float),
                              ALIGN, device, &dev_c);
    assert(status == ZE_RESULT_SUCCESS);        

    status = zeKernelSetGroupSize(kernel, group_size[0],
                                  group_size[1], group_size[2]);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeKernelSetArgumentValue(kernel, 0, sizeof(dev_a), &dev_a);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeKernelSetArgumentValue(kernel, 1, sizeof(dev_a), &dev_b);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeKernelSetArgumentValue(kernel, 2, sizeof(dev_a), &dev_c);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeKernelSetArgumentValue(kernel, 3, sizeof(size), &size);
    assert(status == ZE_RESULT_SUCCESS);
    ze_command_list_desc_t cmd_list_desc = {
          ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    ze_command_list_handle_t cmd_list = nullptr;
    status = zeCommandListCreate(context, device, &cmd_list_desc, &cmd_list);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListAppendMemoryCopy(cmd_list, dev_a, a.data(),
                                           size * size * sizeof(float),
                                           nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListAppendMemoryCopy(cmd_list, dev_b, b.data(),
                                           size * size * sizeof(float),
                                           nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
        ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, 1};
    ze_event_pool_handle_t event_pool = nullptr;
    status = zeEventPoolCreate(context, &event_pool_desc,
                               0, nullptr, &event_pool);
    assert(status == ZE_RESULT_SUCCESS);
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0,
        ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t event = nullptr;
    zeEventCreate(event_pool, &event_desc, &event);
    assert(status == ZE_RESULT_SUCCESS);

    ze_group_count_t dim = { size / group_size[0],
                             size / group_size[1],
                             1 };
    status = zeCommandListAppendLaunchKernel(cmd_list, kernel, &dim,
                                             event, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListAppendMemoryCopy(cmd_list, c.data(), dev_c,
                                           size * size * sizeof(float),
                                           nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeCommandListClose(cmd_list);
    assert(status == ZE_RESULT_SUCCESS);

    ze_command_queue_desc_t cmd_queue_desc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_handle_t cmd_queue = nullptr;
    status = zeCommandQueueCreate(context, device, &cmd_queue_desc, &cmd_queue);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeCommandQueueSynchronize(cmd_queue, UINT32_MAX);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeCommandQueueDestroy(cmd_queue);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeCommandListDestroy(cmd_list);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeMemFree(context, dev_a);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeMemFree(context, dev_b);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeMemFree(context, dev_c);
    assert(status == ZE_RESULT_SUCCESS);

    ze_device_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    status = zeDeviceGetProperties(device, &props);
    assert(status == ZE_RESULT_SUCCESS);

    ze_kernel_timestamp_result_t timestamp{};
    status = zeEventQueryKernelTimestamp(event, &timestamp);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeEventDestroy(event);
    assert(status == ZE_RESULT_SUCCESS);
    double time = static_cast<double>(
        (timestamp.context.kernelEnd - timestamp.context.kernelStart) *
        props.timerResolution);
    time /= NSEC_IN_SEC;
    cout << "Matrix multiplication time: " << time << " sec" << endl;
}

static void Compute(ze_device_handle_t device,
                    ze_driver_handle_t driver,
                    const vector<float>& a,
                    const vector<float>& b,
                    vector<float>& c,
                    int size, int repeat_count)
{

    assert(device != nullptr && driver != nullptr);
    assert(size > 0 && repeat_count > 0);

    string module_name = "gemm.spv";
    vector<uint8_t> binary = LoadBinaryFile(
          GetExecutablePath() + module_name);
    if (binary.size() == 0) {
        cout << "Unable to find module " << module_name << endl;
        return;
    }
    ze_result_t status = ZE_RESULT_SUCCESS;
    ze_context_handle_t context = nullptr;
    ze_context_desc_t context_desc = {
        ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};

    status = zeContextCreate(driver, &context_desc, &context);
    assert(status == ZE_RESULT_SUCCESS);

    ze_module_desc_t module_desc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
        ZE_MODULE_FORMAT_IL_SPIRV, static_cast<uint32_t>(binary.size()),
        binary.data(), nullptr, nullptr};
    ze_module_handle_t module = nullptr;
    status = zeModuleCreate(context, device, &module_desc, &module, nullptr);
    assert(status == ZE_RESULT_SUCCESS && module != nullptr);

    ze_kernel_desc_t kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "GEMM"};
    ze_kernel_handle_t kernel = nullptr;
    status = zeKernelCreate(module, &kernel_desc, &kernel);
    assert(status == ZE_RESULT_SUCCESS);

    for (int i = 0; i < repeat_count; ++i) {
        RunKernel(kernel, device, context, a, b, c, size);
    }
    status = zeKernelDestroy(kernel);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeModuleDestroy(module);
    assert(status == ZE_RESULT_SUCCESS);
}

int 
main(int argc, char* argv[]) {
    int retVal;
    InParams param;
    ze_result_t status = ZE_RESULT_SUCCESS;
    int size = 1024;

    int i = 1;
    if (argc > 1) {
        size = atoi(argv[i]);
        //support max 32K matrix size, enaure size * size is a valid 32bit integer
        if (size) {
           if ((size <=32)  || (size > 0x8000)) {
                cout << "input matrix size is invalid or too large (>32K), abort, "
					 << "using defaut 1024." << endl;
                size = 1024;
			}
			i++;
        }
    }
    int repeat_count = 4;
    if ((argc > 2 ) && (i > 1)) {
        repeat_count = atoi(argv[i]);
        if ((repeat_count <= 0) || (repeat_count > 0x100000)) {
            repeat_count = 4;
        }
		i++;
    }
    retVal = parseInputParam(argc, argv, &param);
    if (retVal) {
		printf("Invalid input parameters.\n");
		printf("usage: %s [-m metric[:device=0][:tile=0]]\n", argv[0]);
		return 1;
    }

#if defined(ENABLE_PAPI)
    // init PAPI intel_gpu component with selected metrics
    int    event_set      = PAPI_NULL;
    int    num_metrics    = 0;
    char **metric_names   = NULL;

    int cid         = -1;
    retVal = putenv((char *)env_str);
    if (retVal) {
		cout << "setting EXT_ENABLE_API_TRACING_EXP=1 failed. " 
			 << "Not able to run query based data collection. " << endl;
		return 1;
    }
    cid = initPAPIGPUComp();
    if (cid < 0) {
       return 1;
    }
    metric_names = (char **)(param.metric_names);
    num_metrics = param.num_metrics;
    retVal = initMetricSet(metric_names, num_metrics, &event_set);
    if (retVal != PAPI_OK) {
         PAPI_shutdown();
         return 1;
    }
#else
    status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    assert(status == ZE_RESULT_SUCCESS);
#endif

    ze_device_handle_t device = nullptr;
    ze_driver_handle_t driver = nullptr;
    GetIntelDeviceAndDriver(ZE_DEVICE_TYPE_GPU, param.app_dev, param.app_tile, device, driver);
    if (device == nullptr || driver == nullptr) {
        cout << "Unable to find target device" << endl;
        return 0;
    }

    cout << "Level Zero Matrix Multiplication (matrix size: " << size <<
      " x " << size << ", repeats " << repeat_count << " times)" << endl;
    cout << "Target device: " << GetDeviceName(device) <<
      endl;

    vector<float> a(size * size, A_VALUE);
    vector<float> b(size * size, B_VALUE);
    vector<float> c(size * size, 0.0f);


#if defined(ENABLE_PAPI)
    // enable tracing before offload start 
    auto start_all = chrono::steady_clock::now();
    long long *metric_values = (long long *)calloc(num_metrics, sizeof(long long));
    CHECK_N_MSG_EXIT((metric_values == NULL), "Error on allocating memory.", PAPI_ENOMEM);
    retVal = PAPI_start(event_set);
    CHECK_N_MSG_EXIT((retVal!=PAPI_OK), "Error on PAPI_start", retVal);
#endif

    auto start = chrono::steady_clock::now();
    Compute(device, driver, a, b, c, size, repeat_count);
    auto end = chrono::steady_clock::now();
    chrono::duration<float> time = end - start;
    cout << "Total execution time: " << time.count() << " sec" << endl;

#if defined(ENABLE_PAPI)
    // data ready when offload finish
    retVal = PAPI_stop(event_set, metric_values);
    CHECK_N_MSG_EXIT((retVal!=PAPI_OK), "Error on PAPI_stop", retVal);
    for (int i=0; i<num_metrics; i++) {
        printf("%-50s ......  %llu\n", metric_names[i], metric_values[i]);
    }
    free(metric_values);
    auto end_all = chrono::steady_clock::now();
    chrono::duration<float> time_all = end_all - start_all;
    cout << "Total time: " << time_all.count() << " sec" << endl;
#endif

    return 0;
}
