#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <sstream>

namespace torch::comms {

using npuStream_t = ::c10_npu::NPUStream;
using npuEvent_t = ::c10_npu::NPUEvent;

struct npuDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  int64_t cubeCoreNum;
};

// Graph-related types (placeholder - may not be supported in NPU)
using npuGraph_t = void*;
using npuGraphNode_t = void*;
using npuUserObject_t = void*;
using npuHostFn_t = void (*)(void*);

// Stream capture status (may not be supported in NPU)
enum npuStreamCaptureStatus {
  npuStreamCaptureStatusNone = 0,
};

// Error code type
using npu_result_t = aclError;
constexpr npu_result_t NPU_SUCCESS = ACL_SUCCESS;
constexpr npu_result_t NPU_ERROR_INVALID_VALUE = ACL_ERROR_INVALID_PARAM;
constexpr npu_result_t NPU_ERROR_NOT_READY = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
constexpr npu_result_t NPU_ERROR_INVALID_HANDLE =
    ACL_ERROR_INVALID_RESOURCE_HANDLE;
constexpr npu_result_t NPU_ERROR_OUT_OF_MEMORY = ACL_ERROR_RT_MEMORY_ALLOCATION;
constexpr npu_result_t NPU_ERROR_UNSUPPORTED = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;

#define NPU_CHECK(npu_api, call, err_str)                                \
  do {                                                                   \
    npu_result_t status = call;                                          \
    if (status != NPU_SUCCESS) {                                         \
      std::stringstream ss;                                              \
      ss << err_str << ": " << npu_api->getErrorString(status) << " at " \
         << __FILE__ << ":" << __LINE__;                                 \
      throw std::runtime_error(ss.str());                                \
    }                                                                    \
  } while (0)

/**
 * Abstract interface for NPU API operations.
 * This allows for dependency injection and testing by providing
 * a way to override NPU API calls.
 */
class NpuApi {
 public:
  virtual ~NpuApi() = default;

  // Device management
  virtual npu_result_t setDevice(int device) = 0;
  virtual npu_result_t getDeviceProperties(npuDeviceProp* prop, int device) = 0;
  virtual npu_result_t memGetInfo(size_t* free, size_t* total) = 0;
  virtual npu_result_t getDeviceCount(int* count) = 0;

  // Stream management
  virtual npu_result_t streamCreateWithPriority(
      npuStream_t& stream,
      unsigned int flags,
      int priority) = 0;
  virtual npu_result_t streamDestroy(const npuStream_t& stream) = 0;
  virtual npu_result_t streamWaitEvent(
      const npuStream_t& stream,
      npuEvent_t& event,
      unsigned int flags) = 0;
  virtual npuStream_t getCurrentNPUStream(int device_index) = 0;
  virtual npu_result_t streamSynchronize(const npuStream_t& stream) = 0;
  virtual npu_result_t streamIsCapturing(
      npuStream_t stream,
      npuStreamCaptureStatus* pCaptureStatus) = 0;
  virtual npu_result_t streamGetCaptureInfo(
      npuStream_t stream,
      npuStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) = 0;

  // Memory management
  virtual npu_result_t malloc(void** devPtr, size_t size) = 0;
  virtual npu_result_t free(void* devPtr) = 0;
  virtual npu_result_t
  memcpyAsync(void* dst, const void* src, size_t count, npuStream_t stream) = 0;

  // Event management
  virtual npu_result_t eventCreate(npuEvent_t& event) = 0;
  virtual npu_result_t eventCreateWithFlags(
      npuEvent_t& event,
      unsigned int flags) = 0;
  virtual npu_result_t eventDestroy(npuEvent_t& event) = 0;
  virtual npu_result_t eventRecord(
      npuEvent_t& event,
      const npuStream_t& stream) = 0;
  virtual npu_result_t eventQuery(const npuEvent_t& event) = 0;

  // Graph operations (unsupported, kept for API compatibility)
  virtual npu_result_t userObjectCreate(
      npuUserObject_t* object_out,
      void* ptr,
      npuHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) = 0;
  virtual npu_result_t graphRetainUserObject(
      npuGraph_t graph,
      npuUserObject_t object,
      unsigned int count,
      unsigned int flags) = 0;
  virtual npu_result_t streamGetCaptureInfo_v2(
      npuStream_t stream,
      npuStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      npuGraph_t* graph_out,
      const npuGraphNode_t** dependencies_out,
      size_t* numDependencies_out) = 0;

  // Error handling
  virtual const char* getErrorString(npu_result_t error) = 0;
};

class DefaultNpuApi : public NpuApi {
 public:
  ~DefaultNpuApi() override = default;

  // Device management
  npu_result_t setDevice(int device) override;
  npu_result_t getDeviceProperties(npuDeviceProp* prop, int device) override;
  npu_result_t memGetInfo(size_t* free, size_t* total) override;
  npu_result_t getDeviceCount(int* count) override;

  // Stream management
  npu_result_t streamCreateWithPriority(
      npuStream_t& stream,
      unsigned int flags,
      int priority) override;
  npu_result_t streamDestroy(const npuStream_t& stream) override;
  npu_result_t streamWaitEvent(
      const npuStream_t& stream,
      npuEvent_t& event,
      unsigned int flags) override;
  npuStream_t getCurrentNPUStream(int device_index) override;
  npu_result_t streamSynchronize(const npuStream_t& stream) override;
  npu_result_t streamIsCapturing(
      npuStream_t stream,
      npuStreamCaptureStatus* pCaptureStatus) override;
  npu_result_t streamGetCaptureInfo(
      npuStream_t stream,
      npuStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) override;

  // Memory management
  npu_result_t malloc(void** devPtr, size_t size) override;
  npu_result_t free(void* devPtr) override;
  npu_result_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      npuStream_t stream) override;

  // Event management
  npu_result_t eventCreate(npuEvent_t& event) override;
  npu_result_t eventCreateWithFlags(npuEvent_t& event, unsigned int flags)
      override;
  npu_result_t eventDestroy(npuEvent_t& event) override;
  npu_result_t eventRecord(npuEvent_t& event, const npuStream_t& stream)
      override;
  npu_result_t eventQuery(const npuEvent_t& event) override;

  // Graph operations (unsupported)
  npu_result_t userObjectCreate(
      npuUserObject_t* object_out,
      void* ptr,
      npuHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) override;
  npu_result_t graphRetainUserObject(
      npuGraph_t graph,
      npuUserObject_t object,
      unsigned int count,
      unsigned int flags) override;
  npu_result_t streamGetCaptureInfo_v2(
      npuStream_t stream,
      npuStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      npuGraph_t* graph_out,
      const npuGraphNode_t** dependencies_out,
      size_t* numDependencies_out) override;

  // Error handling
  const char* getErrorString(npu_result_t error) override;
};

} // namespace torch::comms
