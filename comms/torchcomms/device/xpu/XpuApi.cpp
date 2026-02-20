#include "comms/torchcomms/device/xpu/XpuApi.hpp"
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sstream>
#include <stdexcept>
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {

xpu_result_t DefaultXpuApi::setDevice(int device) {
  try {
    ::c10::xpu::set_device(device);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to set XPU device to " << device << ": "
                  << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::getDeviceProperties(
    xpuDeviceProp* prop,
    int device) {
  if (!prop) {
    return XPU_ERROR_INVALID_VALUE;
  }

  try {
    sycl::device sycl_device = ::c10::xpu::get_raw_device(device);

    // Get device name
    std::string device_name = sycl_device.get_info<sycl::info::device::name>();
    strncpy(prop->name, device_name.c_str(), 255);
    prop->name[255] = '\0';

    // Get memory info
    prop->totalGlobalMem =
        sycl_device.get_info<sycl::info::device::global_mem_size>();

    if (!sycl_device.has(sycl::aspect::ext_intel_free_memory)) [[unlikely]] {
      TC_LOG(WARNING)
          << "Free memory queries are unsupported on this SYCL device; using total global memory as the free-memory estimate.";
    }

    // Get compute capabilities
    auto max_work_group_size =
        sycl_device.get_info<sycl::info::device::max_work_group_size>();
    auto max_work_item_sizes =
        sycl_device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    auto max_compute_units =
        sycl_device.get_info<sycl::info::device::max_compute_units>();

    prop->multiProcessorCount = max_compute_units;
    prop->maxThreadsPerBlock = max_work_group_size;
    prop->maxThreadsDim[0] = max_work_item_sizes[0];
    prop->maxThreadsDim[1] = max_work_item_sizes[1];
    prop->maxThreadsDim[2] = max_work_item_sizes[2];

    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to get XPU device properties for device " << device
                  << ": " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::memGetInfo(size_t* free, size_t* total) {
  if (!free || !total) {
    return XPU_ERROR_INVALID_VALUE;
  }

  try {
    int device = ::c10::xpu::current_device();
    sycl::device& sycl_device = ::c10::xpu::get_raw_device(device);

    *total = sycl_device.get_info<sycl::info::device::global_mem_size>();
    if (sycl_device.has(sycl::aspect::ext_intel_free_memory)) [[likely]] {
      *free =
          sycl_device.get_info<sycl::ext::intel::info::device::free_memory>();
    } else [[unlikely]] {
      *free = *total;
    }

    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to get XPU memory info: " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::getDeviceCount(int* count) {
  if (!count) {
    return XPU_ERROR_INVALID_VALUE;
  }

  try {
    *count = ::c10::xpu::device_count();
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to get XPU device count: " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::streamCreateWithPriority(
    xpuStream_t& stream,
    [[maybe_unused]] unsigned int flags,
    int priority) {
  try {
    stream = ::c10::xpu::getStreamFromPool(priority);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to create XPU stream with priority " << priority
                  << ": " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::streamDestroy(
    [[maybe_unused]] const xpuStream_t& stream) {
  // Stream is managed by PyTorch, nothing to do
  return XPU_SUCCESS;
}

xpu_result_t DefaultXpuApi::streamWaitEvent(
    const xpuStream_t& stream,
    xpuEvent_t& event,
    [[maybe_unused]] unsigned int flags) {
  try {
    event.block(stream);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to make XPU stream wait on event: " << e.what();
    return XPU_ERROR_INVALID_HANDLE;
  }
}

xpuStream_t DefaultXpuApi::getCurrentXPUStream(int device_index) {
  return ::c10::xpu::getCurrentXPUStream(device_index);
}

xpu_result_t DefaultXpuApi::streamSynchronize(const xpuStream_t& stream) {
  try {
    stream.queue().wait_and_throw();
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to synchronize XPU stream: " << e.what();
    return XPU_ERROR_INVALID_HANDLE;
  }
}

// Remove [[maybe_unused]] and implement this function once stream capture is
// supported.
xpu_result_t DefaultXpuApi::streamIsCapturing(
    [[maybe_unused]] const xpuStream_t& stream,
    xpuStreamCaptureStatus* pCaptureStatus) {
  if (!pCaptureStatus) {
    return XPU_ERROR_INVALID_VALUE;
  }

  // XPU/SYCL doesn't support stream capture
  *pCaptureStatus = xpuStreamCaptureStatusNone;
  return XPU_ERROR_UNSUPPORTED;
}

// Remove [[maybe_unused]] and implement this function once stream capture is
// supported.
xpu_result_t DefaultXpuApi::streamGetCaptureInfo(
    [[maybe_unused]] const xpuStream_t& stream,
    xpuStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  if (!pCaptureStatus) {
    return XPU_ERROR_INVALID_VALUE;
  }

  *pCaptureStatus = xpuStreamCaptureStatusNone;
  if (pId) {
    *pId = 0;
  }
  return XPU_ERROR_UNSUPPORTED;
}

xpu_result_t DefaultXpuApi::malloc(void** devPtr, size_t size) {
  if (!devPtr) {
    return XPU_ERROR_INVALID_VALUE;
  }

  if (size == 0) {
    *devPtr = nullptr;
    return XPU_SUCCESS;
  }

  try {
    // Use SYCL's malloc_device
    sycl::context& ctx = ::c10::xpu::get_device_context();
    int device = ::c10::xpu::current_device();
    sycl::device& dev = ::c10::xpu::get_raw_device(device);

    *devPtr = sycl::malloc_device(size, dev, ctx);

    if (!*devPtr) {
      return XPU_ERROR_OUT_OF_MEMORY;
    }

    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to allocate " << size
                  << " bytes on XPU: " << e.what();
    return XPU_ERROR_OUT_OF_MEMORY;
  }
}

xpu_result_t DefaultXpuApi::free(void* devPtr) {
  if (!devPtr) {
    return XPU_SUCCESS;
  }

  try {
    sycl::context& ctx = ::c10::xpu::get_device_context();
    sycl::free(devPtr, ctx);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to free XPU memory: " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    const xpuStream_t& stream) {
  if (!dst || !src) {
    return XPU_ERROR_INVALID_VALUE;
  }

  if (count == 0) {
    return XPU_SUCCESS;
  }

  try {
    stream.queue().memcpy(dst, src, count);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to perform asynchronous memory copy on XPU: "
                  << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::eventCreate(xpuEvent_t& event) {
  try {
    event = ::at::xpu::XPUEvent(false); // No timing
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to create XPU event: " << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::eventCreateWithFlags(
    xpuEvent_t& event,
    unsigned int flags) {
  try {
    bool enable_timing = (flags & 0x1) != 0;
    event = ::at::xpu::XPUEvent(enable_timing);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to create XPU event with flags " << flags << ": "
                  << e.what();
    return XPU_ERROR_INVALID_VALUE;
  }
}

xpu_result_t DefaultXpuApi::eventDestroy(
    [[maybe_unused]] const xpuEvent_t& event) {
  // Event is RAII, nothing to do
  return XPU_SUCCESS;
}

xpu_result_t DefaultXpuApi::eventRecord(
    xpuEvent_t& event,
    const xpuStream_t& stream) {
  try {
    event.record(stream);
    return XPU_SUCCESS;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to record XPU event: " << e.what();
    return XPU_ERROR_INVALID_HANDLE;
  }
}

xpu_result_t DefaultXpuApi::eventQuery(const xpuEvent_t& event) {
  try {
    bool is_complete = event.query();
    return is_complete ? XPU_SUCCESS : XPU_ERROR_NOT_READY;
  } catch (const std::exception& e) {
    TC_LOG(ERROR) << "Failed to query XPU event: " << e.what();
    return XPU_ERROR_INVALID_HANDLE;
  }
}

// Graph Operations (Unsupported)
xpu_result_t DefaultXpuApi::userObjectCreate(
    [[maybe_unused]] xpuUserObject_t* object_out,
    [[maybe_unused]] void* ptr,
    [[maybe_unused]] xpuHostFn_t destroy,
    [[maybe_unused]] unsigned int initialRefcount,
    [[maybe_unused]] unsigned int flags) {
  // XPU/SYCL doesn't support user objects
  return XPU_ERROR_UNSUPPORTED;
}

// Remove [[maybe_unused]] and implement this function once graph support is
// available.
xpu_result_t DefaultXpuApi::graphRetainUserObject(
    [[maybe_unused]] xpuGraph_t graph,
    [[maybe_unused]] xpuUserObject_t object,
    [[maybe_unused]] unsigned int count,
    [[maybe_unused]] unsigned int flags) {
  // Currently, XPU/SYCL doesn't support graphs
  return XPU_ERROR_UNSUPPORTED;
}

// Remove [[maybe_unused]] and implement this function once graph support is
// available.
xpu_result_t DefaultXpuApi::streamGetCaptureInfo_v2(
    [[maybe_unused]] const xpuStream_t& stream,
    [[maybe_unused]] xpuStreamCaptureStatus* captureStatus_out,
    [[maybe_unused]] unsigned long long* id_out,
    [[maybe_unused]] xpuGraph_t* graph_out,
    [[maybe_unused]] const xpuGraphNode_t** dependencies_out,
    [[maybe_unused]] size_t* numDependencies_out) {
  // Currently, XPU/SYCL doesn't support graphs
  return XPU_ERROR_UNSUPPORTED;
}

// Error Handling
const char* DefaultXpuApi::getErrorString(xpu_result_t error) {
  switch (error) {
    case XPU_SUCCESS:
      return "success";
    case XPU_ERROR_INVALID_VALUE:
      return "invalid value";
    case XPU_ERROR_NOT_READY:
      return "not ready";
    case XPU_ERROR_INVALID_HANDLE:
      return "invalid handle";
    case XPU_ERROR_OUT_OF_MEMORY:
      return "out of memory";
    case XPU_ERROR_UNSUPPORTED:
      return "unsupported feature";
    default:
      return "unknown error";
  }
}

} // namespace torch::comms
