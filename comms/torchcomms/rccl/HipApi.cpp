// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/HipApi.hpp"
#include <ATen/hip/HIPContext.h> // @manual

namespace torch::comms {

// DefaultHipApi implementation

hipError_t DefaultHipApi::setDevice(int device) {
  return hipSetDevice(device);
}

hipError_t DefaultHipApi::getDeviceProperties(
    hipDeviceProp_t* prop,
    int device) {
  return hipGetDeviceProperties(prop, device);
}

hipError_t DefaultHipApi::memGetInfo(size_t* free, size_t* total) {
  return hipMemGetInfo(free, total);
}

hipError_t DefaultHipApi::getDeviceCount(int* count) {
  return hipGetDeviceCount(count);
}

hipError_t DefaultHipApi::getStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  return hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
}

hipError_t DefaultHipApi::streamCreateWithPriority(
    hipStream_t* pStream,
    unsigned int flags,
    int priority) {
  return hipStreamCreateWithPriority(pStream, flags, priority);
}

hipError_t DefaultHipApi::streamDestroy(hipStream_t stream) {
  return hipStreamDestroy(stream);
}

hipError_t DefaultHipApi::streamWaitEvent(
    hipStream_t stream,
    hipEvent_t event,
    unsigned int flags) {
  return hipStreamWaitEvent(stream, event, flags);
}

hipStream_t DefaultHipApi::getCurrentHIPStreamMasqueradingAsCUDA(
    int device_index) {
#ifdef HIPIFY_V2
  return at::cuda::getCurrentCUDAStream(device_index).stream();
#else
  return at::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index).stream();
#endif
}

hipError_t DefaultHipApi::streamSynchronize(hipStream_t stream) {
  return hipStreamSynchronize(stream);
}

hipError_t DefaultHipApi::threadExchangeStreamCaptureMode(
    enum hipStreamCaptureMode* mode) {
  return hipThreadExchangeStreamCaptureMode(mode);
}

hipError_t DefaultHipApi::malloc(void** devPtr, size_t size) {
  return hipMalloc(devPtr, size);
}

hipError_t DefaultHipApi::free(void* devPtr) {
  return hipFree(devPtr);
}

hipError_t DefaultHipApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    hipMemcpyKind kind,
    hipStream_t stream) {
  return hipMemcpyAsync(dst, src, count, kind, stream);
}

hipError_t DefaultHipApi::eventCreate(hipEvent_t* event) {
  return hipEventCreate(event);
}

hipError_t DefaultHipApi::eventDestroy(hipEvent_t event) {
  return hipEventDestroy(event);
}

hipError_t DefaultHipApi::eventRecord(hipEvent_t event, hipStream_t stream) {
  return hipEventRecord(event, stream);
}

hipError_t DefaultHipApi::eventQuery(hipEvent_t event) {
  return hipEventQuery(event);
}

const char* DefaultHipApi::getErrorString(hipError_t error) {
  return hipGetErrorString(error);
}

} // namespace torch::comms
