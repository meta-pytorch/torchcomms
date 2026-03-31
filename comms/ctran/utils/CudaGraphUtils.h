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
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return hipStreamGetCaptureInfo_v2(stream, &info.status, &info.id, &info.g);
#elif CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(stream, &info.status, &info.id, &info.g);
#else
  return cudaStreamGetCaptureInfo_v3(stream, &info.status, &info.id, &info.g);
#endif
}

// Retain a user object on the graph so its destroy callback runs when the
// graph is destroyed. Use this to tie resource lifetime to graph lifetime
// (e.g., pinned memory that must outlive graph replays).
inline commResult_t retainUserObject(
    void* obj,
    cudaHostFn_t destroyCallback,
    StreamCaptureInfo& info) {
  cudaUserObject_t object;
  FB_CUDACHECK(cudaUserObjectCreate(
      &object, obj, destroyCallback, 1, cudaUserObjectNoDestructorSync));
  FB_CUDACHECK(
      cudaGraphRetainUserObject(info.g, object, 1, cudaGraphUserObjectMove));
  return commSuccess;
}

// Add a host node to the captured graph and retain the user object so its
// destroy callback runs on graph destruction.
inline commResult_t addHostNode(
    void* data,
    cudaHostFn_t execCallback,
    cudaHostFn_t destroyCallback,
    cudaStream_t stream,
    StreamCaptureInfo& info) {
  FB_CUDACHECK(cudaLaunchHostFunc(stream, execCallback, data));
  return retainUserObject(data, destroyCallback, info);
}
// Add an event record node to a graph being captured on `capturedStream`.
//
// During graph capture, cudaEventRecord on a captured stream taints the
// event's internal state, making subsequent live cudaStreamWaitEvent calls
// fail with cudaErrorIllegalState. cudaGraphAddEventRecordNode avoids this
// by creating a graph RECORD node that fires at replay time (updating the
// event's live state) without modifying the event during capture.
inline commResult_t addEventRecordNodeToCapture(
    cudaStream_t capturedStream,
    cudaGraph_t graph,
    cudaEvent_t event,
    cudaGraphNode_t* outRecordNode = nullptr) {
  cudaGraphNode_t recordNode;
  FB_CUDACHECK(
      cudaGraphAddEventRecordNode(&recordNode, graph, nullptr, 0, event));

  cudaStreamCaptureStatus status;
  const cudaGraphNode_t* deps = nullptr;
  size_t numDeps = 0;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  FB_CUDACHECK(hipStreamGetCaptureInfo_v2(
      capturedStream, &status, nullptr, nullptr, &deps, &numDeps));
  for (size_t i = 0; i < numDeps; i++) {
    FB_CUDACHECK(cudaGraphAddDependencies(graph, &deps[i], &recordNode, 1));
  }
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      capturedStream, &recordNode, 1, cudaStreamSetCaptureDependencies));
#else
  const cudaGraphEdgeData* edges = nullptr;
#if CUDART_VERSION >= 13000
  FB_CUDACHECK(cudaStreamGetCaptureInfo(
      capturedStream, &status, nullptr, nullptr, &deps, &edges, &numDeps));
#else
  FB_CUDACHECK(cudaStreamGetCaptureInfo_v3(
      capturedStream, &status, nullptr, nullptr, &deps, &edges, &numDeps));
#endif
  for (size_t i = 0; i < numDeps; i++) {
#if CUDART_VERSION >= 13000
    FB_CUDACHECK(cudaGraphAddDependencies(
        graph, &deps[i], &recordNode, edges ? &edges[i] : nullptr, 1));
#else
    if (edges) {
      FB_CUDACHECK(cudaGraphAddDependencies_v2(
          graph, &deps[i], &recordNode, &edges[i], 1));
    } else {
      FB_CUDACHECK(cudaGraphAddDependencies(graph, &deps[i], &recordNode, 1));
    }
#endif
  }

#if CUDART_VERSION >= 13000
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      capturedStream,
      &recordNode,
      nullptr,
      1,
      cudaStreamSetCaptureDependencies));
#else
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      capturedStream, &recordNode, 1, cudaStreamSetCaptureDependencies));
#endif
#endif // defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  if (outRecordNode) {
    *outRecordNode = recordNode;
  }
  return commSuccess;
}

} // namespace ctran::utils::cudagraph
