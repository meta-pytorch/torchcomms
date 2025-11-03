#include "comms/torchcomms/device/XpuApi.hpp"
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace comms {

xpu_result_t DefaultXpuApi::setDevice(int device) {
    try {
        ::c10::xpu::set_device(device);
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
        return XPU_ERROR_INVALID_VALUE;
    }
}

xpu_result_t DefaultXpuApi::getDeviceProperties(xpuDeviceProp* prop, int device) {
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
        prop->totalGlobalMem = sycl_device.get_info<sycl::info::device::global_mem_size>();
        
        // Set version info (XPU doesn't have major/minor version like CUDA)
        prop->major = 1;
        prop->minor = 0;
        
        // Get compute capabilities
        auto max_work_group_size = sycl_device.get_info<sycl::info::device::max_work_group_size>();
        auto max_work_item_sizes = sycl_device.get_info<sycl::info::device::max_work_item_sizes<3>>();
        auto max_compute_units = sycl_device.get_info<sycl::info::device::max_compute_units>();
        
        prop->multiProcessorCount = max_compute_units;
        prop->maxThreadsPerBlock = max_work_group_size;
        prop->maxThreadsDim[0] = max_work_item_sizes[0];
        prop->maxThreadsDim[1] = max_work_item_sizes[1];
        prop->maxThreadsDim[2] = max_work_item_sizes[2];
        
        // Max grid size - not directly available in SYCL, use reasonable defaults
        prop->maxGridSize[0] = 2147483647;
        prop->maxGridSize[1] = 65535;
        prop->maxGridSize[2] = 65535;
        
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
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
        *free = *total; // SYCL doesn't provide free memory query
        
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
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
        return XPU_ERROR_INVALID_VALUE;
    }
}

xpu_result_t DefaultXpuApi::streamCreateWithPriority(
    xpuStream_t& stream,
    unsigned int flags,
    int priority) {
    try {
        // Map priority: priority < 0 = high, priority >= 0 = normal
        bool isHighPriority = (priority < 0);
        stream = ::c10::xpu::getStreamFromPool(isHighPriority);
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
        return XPU_ERROR_INVALID_VALUE;
    }
}

xpu_result_t DefaultXpuApi::streamDestroy(const xpuStream_t& stream) {
    // Stream is managed by PyTorch, nothing to do
    return XPU_SUCCESS;
}

xpu_result_t DefaultXpuApi::streamWaitEvent(
    const xpuStream_t& stream,
    xpuEvent_t& event,
    unsigned int flags) {
    try {
        event.block(stream);
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
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
        return XPU_ERROR_INVALID_HANDLE;
    }
}

xpu_result_t DefaultXpuApi::streamIsCapturing(
    const xpuStream_t& stream,
    xpuStreamCaptureStatus* pCaptureStatus) {
    if (!pCaptureStatus) {
        return XPU_ERROR_INVALID_VALUE;
    }
    
    // XPU/SYCL doesn't support stream capture
    *pCaptureStatus = xpuStreamCaptureStatusNone;
    return XPU_SUCCESS;
}

xpu_result_t DefaultXpuApi::streamGetCaptureInfo(
    const xpuStream_t& stream,
    xpuStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
    if (!pCaptureStatus) {
        return XPU_ERROR_INVALID_VALUE;
    }
    
    *pCaptureStatus = xpuStreamCaptureStatusNone;
    if (pId) {
        *pId = 0;
    }
    return XPU_SUCCESS;
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
        return XPU_ERROR_INVALID_VALUE;
    }
}


xpu_result_t DefaultXpuApi::eventCreate(xpuEvent_t& event) {
    try {
        event = ::at::xpu::XPUEvent(false); // No timing
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
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
        return XPU_ERROR_INVALID_VALUE;
    }
}

xpu_result_t DefaultXpuApi::eventDestroy(const xpuEvent_t& event) {
    // Event is RAII, nothing to do
    return XPU_SUCCESS;
}

xpu_result_t DefaultXpuApi::eventRecord(xpuEvent_t& event, const xpuStream_t& stream) {
    try {
        event.record(stream);
        return XPU_SUCCESS;
    } catch (const std::exception& e) {
        return XPU_ERROR_INVALID_HANDLE;
    }
}

xpu_result_t DefaultXpuApi::eventQuery(const xpuEvent_t& event) {
    try {
        bool is_complete = event.query();
        return is_complete ? XPU_SUCCESS : XPU_ERROR_NOT_READY;
    } catch (const std::exception& e) {
        return XPU_ERROR_INVALID_HANDLE;
    }
}

// Graph Operations (Unsupported)
xpu_result_t DefaultXpuApi::userObjectCreate(
    xpuUserObject_t* object_out,
    void* ptr,
    xpuHostFn_t destroy,
    unsigned int initialRefcount,
    unsigned int flags) {
    // XPU/SYCL doesn't support user objects
    return XPU_ERROR_UNSUPPORTED;
}

xpu_result_t DefaultXpuApi::graphRetainUserObject(
    xpuGraph_t graph,
    xpuUserObject_t object,
    unsigned int count,
    unsigned int flags) {
    // XPU/SYCL doesn't support graphs
    return XPU_ERROR_UNSUPPORTED;
}

xpu_result_t DefaultXpuApi::streamGetCaptureInfo_v2(
    const xpuStream_t& stream,
    xpuStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    xpuGraph_t* graph_out,
    const xpuGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
    if (captureStatus_out) {
        *captureStatus_out = xpuStreamCaptureStatusNone;
    }
    if (id_out) {
        *id_out = 0;
    }
    if (graph_out) {
        *graph_out = nullptr;
    }
    if (dependencies_out) {
        *dependencies_out = nullptr;
    }
    if (numDependencies_out) {
        *numDependencies_out = 0;
    }
    
    return XPU_SUCCESS;
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

} // namespace comms
} // namespace torch
