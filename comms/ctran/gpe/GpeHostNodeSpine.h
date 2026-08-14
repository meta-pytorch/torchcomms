// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_HOST_NODE_SPINE_H_
#define CTRAN_GPE_HOST_NODE_SPINE_H_

#include <unordered_map>

#include <cuda_runtime.h>

#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/utils/commSpecs.h"

namespace ctran::gpe {

// Places a captured GPE HOST node on the graph, either inline on the user's
// stream (default) or on a side capture stream -- the spine -- that serializes
// the graph's HOST nodes (NCCL_CTRAN_GPE_HOST_NODE_SIDE_STREAM, mixing=0 only).
//
// One spine per captured graph, keyed by capture id: two graphs captured
// concurrently must not share a spine, or their host chains would cross-link.
// This mirrors ncclStrongStream::captureHead, which keys its capture streams by
// graphId for the same reason.
//
// NOT thread-safe: touched only on the submit (capture) thread.
class HostNodeSpine {
 public:
  explicit HostNodeSpine(bool sideStreamEnabled)
      : sideStreamEnabled_{sideStreamEnabled} {}
  ~HostNodeSpine();

  // Add `data`'s HOST node to the graph being captured on `userStream`,
  // routing it through this graph's spine when the side-stream path is on.
  commResult_t submit(
      void* data,
      cudaHostFn_t execCallback,
      cudaHostFn_t destroyCallback,
      cudaStream_t userStream,
      utils::cudagraph::StreamCaptureInfo& info);

 private:
  // `stream` collects a graph's HOST nodes so each inherits only the previous
  // one (a serial chain); `syncEvent` publishes the chain's tail so the user
  // stream can join it before issuing the collective's first node.
  struct Spine {
    cudaStream_t stream{nullptr};
    cudaEvent_t syncEvent{nullptr};
  };

  // Reclaim spines whose capture is no longer active and return the entry for
  // `captureId`, creating it (and its stream/event) on first use.
  commResult_t getOrCreate(
      unsigned long long captureId,
      cudaStream_t userStream,
      Spine** out);

  // Create a fresh spine's stream/event and clear the dependencies it inherits
  // from `userStream`. On failure `spine` may hold partially created handles;
  // the caller is responsible for destroying them.
  commResult_t createSpine(Spine& spine, cudaStream_t userStream);

  std::unordered_map<unsigned long long, Spine> spines_;
  const bool sideStreamEnabled_;
};

} // namespace ctran::gpe

#endif // CTRAN_GPE_HOST_NODE_SPINE_H_
