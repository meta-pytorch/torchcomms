// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GraphCaptureSideStream.h"

#include <vector>

namespace meta::comms {

GraphSideStream::GraphSideStream(int priority) : priority_(priority) {
  // Non-blocking so the side stream can run concurrently with the main
  // stream when both are on the GPU.
  (void)cudaStreamCreateWithPriority(
      &side_stream_, cudaStreamNonBlocking, priority);
  (void)cudaEventCreateWithFlags(&dep_event_, cudaEventDisableTiming);
}

GraphSideStream::~GraphSideStream() {
  for (auto& [_, side] : green_side_streams_) {
    (void)cudaEventDestroy(side.dep_event);
    (void)cudaStreamDestroy(side.stream);
  }
  if (dep_event_ != nullptr) {
    (void)cudaEventDestroy(dep_event_);
    dep_event_ = nullptr;
  }
  if (side_stream_ != nullptr) {
    (void)cudaStreamDestroy(side_stream_);
    side_stream_ = nullptr;
  }
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__) && \
    CUDART_VERSION >= 12050
namespace {

// Resolved through the runtime so this library keeps linking only cudart.
template <typename Fn>
// NOLINTNEXTLINE(facebook-hte-NullableReturn)
Fn driverEntryPoint(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult result{};
  if (cudaGetDriverEntryPointByVersion(
          name, &fn, 12050, cudaEnableDefault, &result) != cudaSuccess ||
      result != cudaDriverEntryPointSuccess) {
    return nullptr;
  }
  return reinterpret_cast<Fn>(fn);
}

// NOLINTNEXTLINE(facebook-hte-NullableReturn)
CUgreenCtx greenContextOf(cudaStream_t stream) {
  using StreamGetCtx = CUresult (*)(CUstream, CUcontext*, CUgreenCtx*);
  static const auto get_ctx = driverEntryPoint<StreamGetCtx>("cuStreamGetCtx");
  CUcontext ctx = nullptr;
  CUgreenCtx green_ctx = nullptr;
  if (get_ctx == nullptr || get_ctx(stream, &ctx, &green_ctx) != CUDA_SUCCESS) {
    return nullptr;
  }
  return green_ctx;
}

// A process-unique id for the green context; CUDA reuses CUgreenCtx addresses
// but never context ids. 0 when it cannot be read.
unsigned long long greenContextId(CUgreenCtx green_ctx) {
  using CtxFromGreenCtx = CUresult (*)(CUcontext*, CUgreenCtx);
  using CtxGetId = CUresult (*)(CUcontext, unsigned long long*);
  static const auto from_green =
      driverEntryPoint<CtxFromGreenCtx>("cuCtxFromGreenCtx");
  static const auto get_id = driverEntryPoint<CtxGetId>("cuCtxGetId");
  CUcontext ctx = nullptr;
  unsigned long long id = 0;
  if (from_green == nullptr || get_id == nullptr ||
      from_green(&ctx, green_ctx) != CUDA_SUCCESS ||
      get_id(ctx, &id) != CUDA_SUCCESS) {
    return 0;
  }
  return id;
}

// The event belongs to the green context's device, not whichever device is
// current when the first fork happens.
// NOLINTNEXTLINE(facebook-hte-NullableReturn)
cudaEvent_t createGreenEvent(CUgreenCtx green_ctx) {
  using CtxFromGreenCtx = CUresult (*)(CUcontext*, CUgreenCtx);
  using CtxPushCurrent = CUresult (*)(CUcontext);
  using CtxPopCurrent = CUresult (*)(CUcontext*);
  static const auto from_green =
      driverEntryPoint<CtxFromGreenCtx>("cuCtxFromGreenCtx");
  static const auto push = driverEntryPoint<CtxPushCurrent>("cuCtxPushCurrent");
  static const auto pop = driverEntryPoint<CtxPopCurrent>("cuCtxPopCurrent");
  CUcontext ctx = nullptr;
  if (from_green == nullptr || push == nullptr || pop == nullptr ||
      from_green(&ctx, green_ctx) != CUDA_SUCCESS ||
      push(ctx) != CUDA_SUCCESS) {
    return nullptr;
  }
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    event = nullptr;
  }
  CUcontext popped = nullptr;
  (void)pop(&popped);
  return event;
}

// NOLINTNEXTLINE(facebook-hte-NullableReturn)
cudaStream_t createGreenStream(CUgreenCtx green_ctx, int priority) {
  using GreenCtxStreamCreate =
      CUresult (*)(CUstream*, CUgreenCtx, unsigned int, int);
  static const auto create =
      driverEntryPoint<GreenCtxStreamCreate>("cuGreenCtxStreamCreate");
  CUstream stream = nullptr;
  if (create == nullptr ||
      create(&stream, green_ctx, CU_STREAM_NON_BLOCKING, priority) !=
          CUDA_SUCCESS) {
    return nullptr;
  }
  return stream;
}

} // namespace
#endif

GraphSideStream::SideStream GraphSideStream::side_stream_for(
    cudaStream_t stream) {
  const SideStream fallback{side_stream_, dep_event_};
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || \
    CUDART_VERSION < 12050
  (void)stream;
  return fallback;
#else
  CUgreenCtx green_ctx = greenContextOf(stream);
  const unsigned long long id =
      green_ctx == nullptr ? 0 : greenContextId(green_ctx);
  if (id == 0) {
    return fallback;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = green_side_streams_.find(id);
  if (it != green_side_streams_.end()) {
    return it->second;
  }
  // First fork from this green context usually happens mid-capture; relaxed
  // mode keeps the stream and event creation from invalidating the capture.
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  (void)cudaThreadExchangeStreamCaptureMode(&mode);
  SideStream side;
  side.stream = createGreenStream(green_ctx, priority_);
  if (side.stream != nullptr) {
    side.dep_event = createGreenEvent(green_ctx);
  }
  (void)cudaThreadExchangeStreamCaptureMode(&mode);
  if (side.dep_event == nullptr) {
    if (side.stream != nullptr) {
      (void)cudaStreamDestroy(side.stream);
    }
    return fallback;
  }
  return green_side_streams_.emplace(id, side).first->second;
#endif
}

cudaError_t GraphSideStream::fork_from(
    cudaStream_t stream,
    std::function<void(cudaStream_t)> fn) {
  if (side_stream_ == nullptr || dep_event_ == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }

  // 1. Check whether the stream is being captured. If not, run the user
  // fn directly on main — no fork/rejoin overhead needed.
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaGraphNode_t* deps = nullptr;
  size_t num_deps = 0;
  unsigned long long capture_id = 0;
  // hipify-perl (ROCm 7.0) does not map cudaStreamGetCaptureInfo_v2, so call
  // the HIP symbol explicitly on AMD (mirrors GraphCudaWaitEvent.cc).
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  cudaError_t err = hipStreamGetCaptureInfo_v2(
      stream, &status, &capture_id, nullptr, &deps, &num_deps);
#elif CUDART_VERSION >= 13000
  const cudaGraphEdgeData* edge_data = nullptr;
  cudaError_t err = cudaStreamGetCaptureInfo(
      stream, &status, &capture_id, nullptr, &deps, &edge_data, &num_deps);
#else
  cudaError_t err = cudaStreamGetCaptureInfo_v2(
      stream, &status, &capture_id, nullptr, &deps, &num_deps);
#endif
  if (err != cudaSuccess) {
    return err;
  }

  if (status != cudaStreamCaptureStatusActive) {
    fn(stream);
    return cudaSuccess;
  }

  // 2. Fork main → side.
  const SideStream side = side_stream_for(stream);
  err = cudaEventRecord(side.dep_event, stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaStreamWaitEvent(side.stream, side.dep_event);
  if (err != cudaSuccess) {
    return err;
  }
  // Keep this capture's forked work in fork order across side streams: the
  // previous side stream's dep_event was last recorded by its rejoin, after its
  // work.
  SideStream previous;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capture_id != 0 && last_capture_id_ == capture_id &&
        last_side_.stream != side.stream) {
      previous = last_side_;
    }
  }
  if (previous.stream != nullptr) {
    err = cudaStreamWaitEvent(side.stream, previous.dep_event);
    if (err != cudaSuccess) {
      return err;
    }
  }

  // Re-query capture info after the fork. The pointers returned by
  // cudaStreamGetCaptureInfo are only valid until the next API call on
  // the stream, so the earlier query's pointers were invalidated by
  // cudaEventRecord above.
  deps = nullptr;
  num_deps = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  err = hipStreamGetCaptureInfo_v2(
      stream, &status, nullptr, nullptr, &deps, &num_deps);
#elif CUDART_VERSION >= 13000
  edge_data = nullptr;
  err = cudaStreamGetCaptureInfo(
      stream, &status, nullptr, nullptr, &deps, &edge_data, &num_deps);
#else
  err = cudaStreamGetCaptureInfo_v2(
      stream, &status, nullptr, nullptr, &deps, &num_deps);
#endif
  if (err != cudaSuccess) {
    return err;
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
  fn(side.stream);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    last_capture_id_ = capture_id;
    last_side_ = side;
  }

  // 4. Rejoin side → main. Required so cudaStreamEndCapture accepts the
  // capture (every forked stream must have a rejoin path).
  (void)cudaEventRecord(side.dep_event, side.stream);
  (void)cudaStreamWaitEvent(stream, side.dep_event);

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
