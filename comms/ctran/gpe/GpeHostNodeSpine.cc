// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/gpe/GpeHostNodeSpine.h"

#include "comms/ctran/utils/Checks.h"

namespace ctran::gpe {

HostNodeSpine::~HostNodeSpine() {
  for (auto& [_, spine] : spines_) {
    (void)cudaEventDestroy(spine.syncEvent);
    (void)cudaStreamDestroy(spine.stream);
  }
}

commResult_t HostNodeSpine::submit(
    void* data,
    cudaHostFn_t execCallback,
    cudaHostFn_t destroyCallback,
    cudaStream_t userStream,
    utils::cudagraph::StreamCaptureInfo& info) {
  if (!sideStreamEnabled_) {
    return utils::cudagraph::addHostNode(
        data, execCallback, destroyCallback, userStream, info);
  }

  // The HOST node goes on the graph's spine so its only predecessor is the
  // previous collective's HOST node, then we join the spine tip back into the
  // user stream. The join must land BEFORE the algo issues the collective's
  // first node (copyToSelf / PipeStart), so that node inherits HOST[i] as a
  // predecessor -- that edge is what orders each collective behind the host
  // chain, matching how NCCL records its own captured host callbacks. Issuing
  // it after the first node makes HOST[i] parent nothing and the whole effect
  // disappears.
  Spine* spine = nullptr;
  FB_COMMCHECK(getOrCreate(info.id, userStream, &spine));
  FB_COMMCHECK(
      utils::cudagraph::addHostNode(
          data, execCallback, destroyCallback, spine->stream, info));
  FB_CUDACHECK(cudaEventRecord(spine->syncEvent, spine->stream));
  FB_CUDACHECK(cudaStreamWaitEvent(userStream, spine->syncEvent, 0));
  return commSuccess;
}

commResult_t HostNodeSpine::getOrCreate(
    unsigned long long captureId,
    cudaStream_t userStream,
    Spine** out) {
  // Reclaim spines from captures that have ended. A finished graph's stream is
  // no longer capturing, and its nodes already live in the instantiated graph,
  // so the stream/event are free to destroy. Without this sweep we would leak
  // one stream + one event per captured graph for the process lifetime.
  for (auto it = spines_.begin(); it != spines_.end();) {
    if (it->first == captureId) {
      ++it;
      continue;
    }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(it->second.stream, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusActive) {
      (void)cudaEventDestroy(it->second.syncEvent);
      (void)cudaStreamDestroy(it->second.stream);
      it = spines_.erase(it);
    } else {
      ++it;
    }
  }

  auto it = spines_.find(captureId);
  if (it == spines_.end()) {
    // Build the spine in a local so a failed creation step never leaves a
    // half-initialized (null stream/event) entry cached under `captureId`: a
    // later submit must not reuse it, nor the destructor operate on its null
    // handles. Only cache once every step has succeeded.
    Spine spine{};
    const commResult_t res = createSpine(spine, userStream);
    if (res != commSuccess) {
      if (spine.syncEvent != nullptr) {
        (void)cudaEventDestroy(spine.syncEvent);
      }
      if (spine.stream != nullptr) {
        (void)cudaStreamDestroy(spine.stream);
      }
      return res;
    }
    it = spines_.emplace(captureId, spine).first;
  }
  *out = &it->second;
  return commSuccess;
}

commResult_t HostNodeSpine::createSpine(Spine& spine, cudaStream_t userStream) {
  FB_CUDACHECK(cudaStreamCreateWithFlags(&spine.stream, cudaStreamNonBlocking));
  FB_CUDACHECK(
      cudaEventCreateWithFlags(&spine.syncEvent, cudaEventDisableTiming));

  // Bring the spine into this graph, then CLEAR the dependency set it just
  // inherited. Without the clear, the first HOST node would depend on
  // everything already captured on the user stream, which is the inline shape
  // we are trying to avoid. This is what
  // ncclStrongStreamAcquire does via cudaStreamSetCaptureDependencies with a
  // zero-length list.
  FB_CUDACHECK(cudaEventRecord(spine.syncEvent, userStream));
  FB_CUDACHECK(cudaStreamWaitEvent(spine.stream, spine.syncEvent, 0));
#if defined(__HIP_PLATFORM_AMD__)
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      spine.stream, nullptr, 0, hipStreamSetCaptureDependencies));
#elif CUDART_VERSION >= 13000
  // CUDA 13 takes an extra edge-data pointer; there is no _v2 name (see the
  // same three-way split in OrderedWorkStreamGuard.cc).
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      spine.stream, nullptr, nullptr, 0, cudaStreamSetCaptureDependencies));
#else
  FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
      spine.stream, nullptr, 0, cudaStreamSetCaptureDependencies));
#endif
  return commSuccess;
}

} // namespace ctran::gpe
