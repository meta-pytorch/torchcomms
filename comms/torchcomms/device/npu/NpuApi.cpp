#include "comms/torchcomms/device/npu/NpuApi.hpp"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <sstream>
#include <stdexcept>
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {

npu_result_t DefaultNpuApi::setDevice(int device) {
  try {
    ::c10_npu::set_device(device);
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_VALUE;
  }
}

npu_result_t DefaultNpuApi::getDeviceProperties(
    npuDeviceProp* prop,
    int device) {
  if (!prop) {
    return NPU_ERROR_INVALID_VALUE;
  }

  // Get device name
  // ACL does not provide a simple "get device name" API here; use a
  // descriptive default name instead.
  snprintf(prop->name, sizeof(prop->name), "Ascend NPU %d", device);

  // Set device before getting memory info
  auto result = setDevice(device);
  if (result != NPU_SUCCESS) {
    return result;
  }

  size_t free_mem = 0;
  size_t total_mem = 0;
  if (aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem) != ACL_SUCCESS) {
    return NPU_ERROR_INVALID_VALUE;
  }
  prop->totalGlobalMem = total_mem;

  if (aclGetDeviceCapability(
          device, ACL_DEVICE_INFO_AI_CORE_NUM, &prop->cubeCoreNum) !=
      ACL_SUCCESS) {
    return NPU_ERROR_INVALID_VALUE;
  }
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::memGetInfo(size_t* free, size_t* total) {
  if (!free || !total) {
    return NPU_ERROR_INVALID_VALUE;
  }

  if (aclrtGetMemInfo(ACL_HBM_MEM, free, total) != ACL_SUCCESS) {
    *total = 0;
    *free = 0;
    return NPU_ERROR_INVALID_VALUE;
  }
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::getDeviceCount(int* count) {
  if (!count) {
    return NPU_ERROR_INVALID_VALUE;
  }

  try {
    *count = static_cast<int>(::c10_npu::device_count());
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_VALUE;
  }
}

npu_result_t DefaultNpuApi::streamCreateWithPriority(
    npuStream_t& stream,
    unsigned int flags,
    int priority) {
  (void)flags;
  try {
    bool is_high_priority = priority != 0;
    auto device_index = ::c10_npu::current_device();
    stream = ::c10_npu::getStreamFromPool(is_high_priority, device_index);
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_VALUE;
  }
}

npu_result_t DefaultNpuApi::streamDestroy(const npuStream_t& stream) {
  (void)stream;
  // Stream is managed by torch_npu
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::streamWaitEvent(
    const npuStream_t& stream,
    npuEvent_t& event,
    unsigned int flags) {
  (void)flags;
  try {
    event.block(stream);
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_HANDLE;
  }
}

npuStream_t DefaultNpuApi::getCurrentNPUStream(int device_index) {
  return ::c10_npu::getCurrentNPUStream(device_index);
}

npu_result_t DefaultNpuApi::streamSynchronize(const npuStream_t& stream) {
  try {
    stream.synchronize();
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_HANDLE;
  }
}

npu_result_t DefaultNpuApi::streamIsCapturing(
    npuStream_t stream,
    npuStreamCaptureStatus* pCaptureStatus) {
  if (!pCaptureStatus) {
    return NPU_ERROR_INVALID_VALUE;
  }

  // NPU/ACL doesn't support stream capture
  *pCaptureStatus = npuStreamCaptureStatusNone;
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::streamGetCaptureInfo(
    npuStream_t stream,
    npuStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  if (!pCaptureStatus) {
    return NPU_ERROR_INVALID_VALUE;
  }

  *pCaptureStatus = npuStreamCaptureStatusNone;
  if (pId) {
    *pId = 0;
  }
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::malloc(void** devPtr, size_t size) {
  if (!devPtr) {
    return NPU_ERROR_INVALID_VALUE;
  }

  if (size == 0) {
    *devPtr = nullptr;
    return NPU_SUCCESS;
  }

  if (aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    *devPtr = nullptr;
    return NPU_ERROR_OUT_OF_MEMORY;
  }
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::free(void* devPtr) {
  if (!devPtr) {
    return NPU_SUCCESS;
  }

  return aclrtFree(devPtr) == ACL_SUCCESS ? NPU_SUCCESS
                                          : NPU_ERROR_INVALID_VALUE;
}

npu_result_t DefaultNpuApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    npuStream_t stream) {
  if (!dst || !src) {
    return NPU_ERROR_INVALID_VALUE;
  }

  if (count == 0) {
    return NPU_SUCCESS;
  }

  return aclrtMemcpyAsync(
      dst, count, src, count, ACL_MEMCPY_DEVICE_TO_DEVICE, stream.stream());
}

npu_result_t DefaultNpuApi::eventCreate(npuEvent_t& event) {
  try {
    event = ::c10_npu::NPUEvent();
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_VALUE;
  }
}

npu_result_t DefaultNpuApi::eventCreateWithFlags(
    npuEvent_t& event,
    unsigned int flags) {
  try {
    event = ::c10_npu::NPUEvent(flags);
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_VALUE;
  }
}

npu_result_t DefaultNpuApi::eventDestroy(npuEvent_t& event) {
  (void)event;
  // NPUEvent is RAII, nothing to do
  return NPU_SUCCESS;
}

npu_result_t DefaultNpuApi::eventRecord(
    npuEvent_t& event,
    const npuStream_t& stream) {
  try {
    event.record(stream);
    return NPU_SUCCESS;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_HANDLE;
  }
}

npu_result_t DefaultNpuApi::eventQuery(const npuEvent_t& event) {
  try {
    return event.query() ? NPU_SUCCESS : NPU_ERROR_NOT_READY;
  } catch (const std::exception&) {
    return NPU_ERROR_INVALID_HANDLE;
  }
}

// Graph Operations (Unsupported)
npu_result_t DefaultNpuApi::userObjectCreate(
    npuUserObject_t* object_out,
    void* ptr,
    npuHostFn_t destroy,
    unsigned int initialRefcount,
    unsigned int flags) {
  // NPU/ACL doesn't support user objects
  return NPU_ERROR_UNSUPPORTED;
}

npu_result_t DefaultNpuApi::graphRetainUserObject(
    npuGraph_t graph,
    npuUserObject_t object,
    unsigned int count,
    unsigned int flags) {
  // Currently, NPU/ACL doesn't support graphs
  return NPU_ERROR_UNSUPPORTED;
}

npu_result_t DefaultNpuApi::streamGetCaptureInfo_v2(
    npuStream_t stream,
    npuStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    npuGraph_t* graph_out,
    const npuGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
  // Currently, NPU/ACL doesn't support graphs
  return NPU_ERROR_UNSUPPORTED;
}

// Error Handling
const char* DefaultNpuApi::getErrorString(npu_result_t error) {
  // ACL provides aclGetRecentErrMsg() for detailed errors
  // For now, return basic error descriptions
  switch (error) {
    case ACL_SUCCESS:
      return "success";
    case ACL_ERROR_INVALID_PARAM:
      return "invalid parameter";
    case ACL_ERROR_INVALID_RESOURCE_HANDLE:
      return "invalid handle";
    case ACL_ERROR_RT_MEMORY_ALLOCATION:
      return "memory allocation failed";
    case ACL_ERROR_RT_FEATURE_NOT_SUPPORT:
      return "feature not supported";
    default:
      return aclGetRecentErrMsg();
  }
}

} // namespace torch::comms
