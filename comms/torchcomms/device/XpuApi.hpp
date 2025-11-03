#pragma once

#include <sycl/sycl.hpp>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>

namespace torch {
namespace comms {

using xpuStream_t = ::c10::xpu::XPUStream;
using xpuEvent_t = ::at::xpu::XPUEvent;

struct xpuDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
};

// Graph-related types (placeholder - unsupported in XPU)
using xpuGraph_t = void*;
using xpuGraphNode_t = void*;
using xpuUserObject_t = void*;
using xpuHostFn_t = void(*)(void*);

// Stream capture status (not supported in XPU)
enum xpuStreamCaptureStatus {
    xpuStreamCaptureStatusNone = 0,
};

// Error code type
using xpu_result_t = int32_t;
constexpr xpu_result_t XPU_SUCCESS = 0;
constexpr xpu_result_t XPU_ERROR_INVALID_VALUE = 1;
constexpr xpu_result_t XPU_ERROR_NOT_READY = 2;
constexpr xpu_result_t XPU_ERROR_INVALID_HANDLE = 3;
constexpr xpu_result_t XPU_ERROR_OUT_OF_MEMORY = 4;
constexpr xpu_result_t XPU_ERROR_UNSUPPORTED = 5;

#define XPU_CHECK(xpu_api, call, err_str)                                 \
  do {                                                                    \
    xpu_result_t status = call;                                           \
    if (status != XPU_SUCCESS) {                                          \
      std::stringstream ss;                                               \
      ss << err_str << ": " << xpu_api->getErrorString(status) << " at "  \
         << __FILE__ << ":" << __LINE__;                                  \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

/**
 * Abstract interface for XPU API operations.
 * This allows for dependency injection and testing by providing
 * a way to override XPU API calls.
 */
class XpuApi {
 public:
  virtual ~XpuApi() = default;

  // Device management
  virtual xpu_result_t setDevice(int device) = 0;
  virtual xpu_result_t getDeviceProperties(xpuDeviceProp* prop, int device) = 0;
  virtual xpu_result_t memGetInfo(size_t* free, size_t* total) = 0;
  virtual xpu_result_t getDeviceCount(int* count) = 0;

  // Stream management
  virtual xpu_result_t streamCreateWithPriority(
      xpuStream_t& stream,
      unsigned int flags,
      int priority) = 0;
  virtual xpu_result_t streamDestroy(const xpuStream_t& stream) = 0;
  virtual xpu_result_t streamWaitEvent(
      const xpuStream_t& stream,
      xpuEvent_t& event,
      unsigned int flags) = 0;
  virtual xpuStream_t getCurrentXPUStream(int device_index) = 0;
  virtual xpu_result_t streamSynchronize(const xpuStream_t& stream) = 0;
  virtual xpu_result_t streamIsCapturing(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* pCaptureStatus) = 0;
  virtual xpu_result_t streamGetCaptureInfo(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) = 0;

  // Memory management
  virtual xpu_result_t malloc(void** devPtr, size_t size) = 0;
  virtual xpu_result_t free(void* devPtr) = 0;
  virtual xpu_result_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      const xpuStream_t& stream) = 0;

  // Event management
  virtual xpu_result_t eventCreate(xpuEvent_t& event) = 0;
  virtual xpu_result_t eventCreateWithFlags(
      xpuEvent_t& event,
      unsigned int flags) = 0;
  virtual xpu_result_t eventDestroy(const xpuEvent_t& event) = 0;
  virtual xpu_result_t eventRecord(xpuEvent_t& event, const xpuStream_t& stream) = 0;
  virtual xpu_result_t eventQuery(const xpuEvent_t& event) = 0;

  // Graph operations (unsupported, kept for API compatibility)
  virtual xpu_result_t userObjectCreate(
      xpuUserObject_t* object_out,
      void* ptr,
      xpuHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) = 0;
  virtual xpu_result_t graphRetainUserObject(
      xpuGraph_t graph,
      xpuUserObject_t object,
      unsigned int count,
      unsigned int flags) = 0;
  virtual xpu_result_t streamGetCaptureInfo_v2(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      xpuGraph_t* graph_out,
      const xpuGraphNode_t** dependencies_out,
      size_t* numDependencies_out) = 0;

  // Error handling
  virtual const char* getErrorString(xpu_result_t error) = 0;
};

class DefaultXpuApi : public XpuApi {
 public:
  ~DefaultXpuApi() override = default;

  // Device management
  xpu_result_t setDevice(int device) override;
  xpu_result_t getDeviceProperties(xpuDeviceProp* prop, int device) override;
  xpu_result_t memGetInfo(size_t* free, size_t* total) override;
  xpu_result_t getDeviceCount(int* count) override;

  // Stream management
  xpu_result_t streamCreateWithPriority(
      xpuStream_t& stream,
      unsigned int flags,
      int priority) override;
  xpu_result_t streamDestroy(const xpuStream_t& stream) override;
  xpu_result_t streamWaitEvent(
      const xpuStream_t& stream,
      xpuEvent_t& event,
      unsigned int flags) override;
  xpuStream_t getCurrentXPUStream(int device_index) override;
  xpu_result_t streamSynchronize(const xpuStream_t& stream) override;
  xpu_result_t streamIsCapturing(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* pCaptureStatus) override;
  xpu_result_t streamGetCaptureInfo(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) override;

  // Memory management
  xpu_result_t malloc(void** devPtr, size_t size) override;
  xpu_result_t free(void* devPtr) override;
  xpu_result_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      const xpuStream_t& stream) override;

  // Event management
  xpu_result_t eventCreate(xpuEvent_t& event) override;
  xpu_result_t eventCreateWithFlags(xpuEvent_t& event, unsigned int flags) override;
  xpu_result_t eventDestroy(const xpuEvent_t& event) override;
  xpu_result_t eventRecord(xpuEvent_t& event, const xpuStream_t& stream) override;
  xpu_result_t eventQuery(const xpuEvent_t& event) override;

  // Graph operations (unsupported)
  xpu_result_t userObjectCreate(
      xpuUserObject_t* object_out,
      void* ptr,
      xpuHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) override;
  xpu_result_t graphRetainUserObject(
      xpuGraph_t graph,
      xpuUserObject_t object,
      unsigned int count,
      unsigned int flags) override;
  xpu_result_t streamGetCaptureInfo_v2(
      const xpuStream_t& stream,
      xpuStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      xpuGraph_t* graph_out,
      const xpuGraphNode_t** dependencies_out,
      size_t* numDependencies_out) override;

  // Error handling
  const char* getErrorString(xpu_result_t error) override;
};

} // namespace comms
} // namespace torch
