// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/utils/colltrace/CollWaitEvent.h"
#include "comms/utils/colltrace/GraphCollTraceEvent.h"
#include "comms/utils/colltrace/GraphCollTraceState.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

// Default shared ring buffer size — must be a power of 2.
// At 24 bytes per entry, 65536 entries = 1.5MB per communicator.
inline constexpr uint32_t kDefaultRingSize =
    65536; // NOLINT(misc-unused-using-decls)

// Reads the NCCLX_COLLTRACE_DEVICE_WRITE cvar (off by default until the
// in-kernel cutover diff lands): whether collective kernels should publish
// their own timestamps into the graph ring (in-kernel "device write"). Defined
// out-of-line so this header (and the exported GraphCollTraceHandle.h that
// gates on it) doesn't pull in nccl_cvars.h.
bool colltraceDeviceWriteEnabled() noexcept;

// Graph-aware wait event that uses device-side globaltimer reads and a
// shared ring buffer for precise per-replay timing. All collectives across
// ALL CUDA graphs share a single ring buffer (owned by CollTrace) and
// atomically claim slots via a shared write index.
//
// The collective kernel publishes its own start/end timestamps into the ring
// from inside the kernel (see ColltraceEventScope + the GPE arming path); this
// class holds the shared ring handle and per-collective identity used for that.
//
// No back-pressure — if the poll thread falls behind by more than ringSize
// replays across all collectives, data loss is detected and logged.
class GraphCudaWaitEvent : public ICollWaitEvent {
 public:
  explicit GraphCudaWaitEvent(cudaStream_t stream, uint32_t collId = 0);

  ~GraphCudaWaitEvent() override;

  GraphCudaWaitEvent(const GraphCudaWaitEvent&) = delete;
  GraphCudaWaitEvent& operator=(const GraphCudaWaitEvent&) = delete;
  GraphCudaWaitEvent(GraphCudaWaitEvent&&) = delete;
  GraphCudaWaitEvent& operator=(GraphCudaWaitEvent&&) = delete;

  CommsMaybeVoid beforeCollKernelScheduled() noexcept override;
  CommsMaybeVoid afterCollKernelScheduled() noexcept override;

  CommsMaybe<bool> waitCollStart(
      std::chrono::milliseconds sleepTimeMs) noexcept override;
  CommsMaybe<bool> waitCollEnd(
      std::chrono::milliseconds sleepTimeMs) noexcept override;

  CommsMaybeVoid signalCollStart() noexcept override;
  CommsMaybeVoid signalCollEnd() noexcept override;

  CommsMaybe<system_clock_time_point> getCollEnqueueTime() noexcept override;
  CommsMaybe<system_clock_time_point> getCollStartTime() noexcept override;
  CommsMaybe<system_clock_time_point> getCollEndTime() noexcept override;

  void attachRingBuffer(
      ::hrdw_ring_buffer::HRDWRingBuffer<GraphCollTraceEvent>*
          ringBuffer) noexcept;

  cudaStream_t getStream() const noexcept {
    return stream_;
  }

  uint32_t getCollId() const noexcept {
    return collId_;
  }

  void setCollId(uint32_t collId) noexcept {
    collId_ = collId;
  }

  bool hasRingBuffer() const noexcept {
    return ringBuffer_ != nullptr;
  }

  // Mark that this collective's kernel is armed to publish its own start/end
  // timestamps from inside the kernel, so the host-launched timestamp path is
  // skipped. Called by the GPE / baseline arming code during submit, before the
  // launch triggers beforeCollKernelScheduled(). TEMPORARY: removed together
  // with the host-launched path at the in-kernel cutover.
  void markInKernelEmit() noexcept {
    inKernelEmit_ = true;
  }

  // Device-side write handle for the shared ring, handed to a collective kernel
  // so it can publish its own start/end timestamps from inside the kernel.
  // Returns a default (null-ring, invalid) handle if no ring is attached, so
  // callers that skip the hasRingBuffer() gate can't null-deref.
  ::hrdw_ring_buffer::HRDWRingBufferDeviceHandle<GraphCollTraceEvent>
  deviceHandle() const noexcept {
    if (ringBuffer_ == nullptr) {
      return {};
    }
    return ringBuffer_->deviceHandle();
  }

 private:
  cudaStream_t stream_;
  uint32_t collId_;
  system_clock_time_point enqueueTime_;

  // Set by markInKernelEmit() when the collective kernel is armed to self-emit;
  // when true the host-launched timestamp path is skipped so the two never
  // double-write the ring. TEMPORARY: removed at the in-kernel cutover.
  bool inKernelEmit_{false};

  // Per-collective timestamp stream + dependency event for the host-launched
  // fork/join timestamp path, used until the kernel self-emits (pre-sm90, or a
  // not-yet-migrated collective). TEMPORARY: removed at the in-kernel cutover.
  cudaStream_t timestampStream_{nullptr};
  cudaEvent_t depEvent_{nullptr};

  // owned by CollTrace, shared across ALL graphs. set via attachRingBuffer().
  ::hrdw_ring_buffer::HRDWRingBuffer<GraphCollTraceEvent>* ringBuffer_{nullptr};
};

} // namespace meta::comms::colltrace
