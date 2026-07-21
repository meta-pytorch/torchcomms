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

// Graph-aware wait event that uses device-side globaltimer reads and a
// shared ring buffer for precise per-replay timing. All collectives across
// ALL CUDA graphs share a single ring buffer (owned by CollTrace) and
// atomically claim slots via a shared write index.
//
// Each GraphCudaWaitEvent owns its own timestamp stream and dependency event.
// Per-collective streams ensure that concurrent collectives (e.g.,
// signal/wait RMA ops) don't serialize their timestamp kernels.
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

  // Device-side write handle for the shared ring, handed to a collective kernel
  // so it can publish its own start/end timestamps from inside the kernel.
  ::hrdw_ring_buffer::HRDWRingBufferDeviceHandle<GraphCollTraceEvent>
  deviceHandle() const noexcept {
    return ringBuffer_->deviceHandle();
  }

  // Once the GPE has armed the collective kernel to write its own timestamps,
  // this suppresses the host-launched timestamp kernels so start/end aren't
  // recorded twice into the ring.
  void setIntraKernelEmitActive() noexcept {
    intraKernelEmitActive_ = true;
  }

 private:
  cudaStream_t stream_;
  uint32_t collId_;
  system_clock_time_point enqueueTime_;

  // per-collective timestamp stream — runs in parallel with the collective
  // stream. each collective gets its own stream so concurrent collectives
  // (e.g., signal/wait) don't serialize their timestamp kernels.
  cudaStream_t timestampStream_{nullptr};
  // per-collective dependency event for fork/join edges.
  cudaEvent_t depEvent_{nullptr};

  // owned by CollTrace, shared across ALL graphs. set via attachRingBuffer().
  ::hrdw_ring_buffer::HRDWRingBuffer<GraphCollTraceEvent>* ringBuffer_{nullptr};

  // When set, the collective kernel writes its own start/end timestamps into
  // the ring, so beforeCollKernelScheduled()/afterCollKernelScheduled() skip
  // the host-launched timestamp kernels. Set via setIntraKernelEmitActive().
  bool intraKernelEmitActive_{false};
};

} // namespace meta::comms::colltrace
