// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GraphCaptureSideStream.h"

#include <vector>

namespace meta::comms {

GraphSideStream::GraphSideStream(int priority) {
  // Non-blocking so the side stream can run concurrently with the main
  // stream when both are on the GPU.
  (void)cudaStreamCreateWithPriority(
      &side_stream_, cudaStreamNonBlocking, priority);
  (void)cudaEventCreateWithFlags(&dep_event_, cudaEventDisableTiming);
}

GraphSideStream::~GraphSideStream() {
  if (dep_event_ != nullptr) {
    (void)cudaEventDestroy(dep_event_);
    dep_event_ = nullptr;
  }
  if (side_stream_ != nullptr) {
    (void)cudaStreamDestroy(side_stream_);
    side_stream_ = nullptr;
  }
}

cudaError_t GraphSideStream::fork_from(
    cudaStream_t stream,
    std::function<void(cudaStream_t)> fn) {
  if (side_stream_ == nullptr || dep_event_ == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }

  // 1. Fork main → side.
  cudaError_t err = cudaEventRecord(dep_event_, stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaStreamWaitEvent(side_stream_, dep_event_);
  if (err != cudaSuccess) {
    return err;
  }

  // 2. Capture main's current dependency set — used in step 5 to rewind
  // main past the rejoin so subsequent main-stream ops aren't bound to
  // the side stream.
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaGraphNode_t* deps = nullptr;
  size_t num_deps = 0;
#if CUDART_VERSION >= 13000
  const cudaGraphEdgeData* edge_data = nullptr;
  err = cudaStreamGetCaptureInfo(
      stream, &status, nullptr, nullptr, &deps, &edge_data, &num_deps);
#else
  err = cudaStreamGetCaptureInfo_v2(
      stream, &status, nullptr, nullptr, &deps, &num_deps);
#endif
  if (err != cudaSuccess) {
    return err;
  }

  if (status != cudaStreamCaptureStatusActive) {
    // Not capturing — run the user fn directly on main. The upstream
    // eventRecord/streamWaitEvent calls above were no-ops for capture but
    // are still valid stream ops; they just added a side-stream sync point.
    fn(stream);
    return cudaSuccess;
  }

  // Snapshot the captured deps; cudaStreamUpdateCaptureDependencies
  // consumes them via a caller-owned buffer, so we need our own copy.
  std::vector<cudaGraphNode_t> saved_deps(deps, deps + num_deps);
#if CUDART_VERSION >= 13000
  std::vector<cudaGraphEdgeData> saved_edge_data;
  if (edge_data != nullptr) {
    saved_edge_data.assign(edge_data, edge_data + num_deps);
  }
#endif

  // 3. Invoke user work on the side stream.
  fn(side_stream_);

  // 4. Rejoin side → main. Required so cudaStreamEndCapture accepts the
  // capture (every forked stream must have a rejoin path).
  (void)cudaEventRecord(dep_event_, side_stream_);
  (void)cudaStreamWaitEvent(stream, dep_event_);

  // 5. Restore main's pre-rejoin deps so the rejoin node, while present
  // in the graph DAG, is NOT a predecessor of any subsequent main-stream
  // op. Implicit "clean up" of the just-added join: we never need to
  // track a "prev event" manually because the dep is removed from main's
  // active set here.
  return cudaStreamUpdateCaptureDependencies(
      stream,
      saved_deps.data(),
#if CUDART_VERSION >= 13000
      saved_edge_data.empty() ? nullptr : saved_edge_data.data(),
#endif
      saved_deps.size(),
      cudaStreamSetCaptureDependencies);
}

} // namespace meta::comms
