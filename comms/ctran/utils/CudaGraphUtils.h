// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "comms/ctran/utils/Checks.h"

namespace ctran::utils::cudagraph {

struct StreamCaptureInfo {
  cudaStreamCaptureStatus status;
  unsigned long long id;
  cudaGraph_t g;
};

inline cudaError_t getStreamCaptureInfo(
    cudaStream_t stream,
    StreamCaptureInfo& info) {
#if CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(stream, &info.status, &info.id, &info.g);
#elif CUDART_VERSION >= 12030
  return cudaStreamGetCaptureInfo_v3(stream, &info.status, &info.id, &info.g);
#else
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return hipStreamGetCaptureInfo_v2(stream, &info.status, &info.id, &info.g);
#else
  return cudaStreamGetCaptureInfo_v2(stream, &info.status, &info.id, &info.g);
#endif // defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#endif // CUDART_VERSION >= 13000
}

inline commResult_t addHostNode(
    void* objectPtr,
    void* data,
    cudaHostFn_t execCallback,
    cudaHostFn_t destroyCallback,
    StreamCaptureInfo& info) {
  cudaHostNodeParams hostParams;
  hostParams.fn = execCallback;
  hostParams.userData = reinterpret_cast<void*>(data);
  cudaUserObject_t object;
  cudaGraphNode_t hostNode;

  FB_CUDACHECK(
      cudaGraphAddHostNode(&hostNode, info.g, nullptr, 0, &hostParams));

  FB_CUDACHECK(cudaUserObjectCreate(
      &object, objectPtr, destroyCallback, 1, cudaUserObjectNoDestructorSync));

  // Handover ownership to CUDA graph
  FB_CUDACHECK(
      cudaGraphRetainUserObject(info.g, object, 1, cudaGraphUserObjectMove));
  return commSuccess;
}
} // namespace ctran::utils::cudagraph
