// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/NvlProgressSlotReleaseTest.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims::test {
namespace {

static_assert(static_cast<int>(NvlProgressStage::Idle) == kNvlStageIdle);
static_assert(static_cast<int>(NvlProgressStage::Active) == kNvlStageActive);

// Arbitrary non-zero geometry. It exists only so the probe can tell an
// abandoned slot from an untouched one.
constexpr std::size_t kBaseByte = 4096;
constexpr std::size_t kNextByte = 8192; // Half a chunk in: unambiguously
                                        // mid-transfer.
constexpr std::size_t kPayloadBytes = 16384;
constexpr std::size_t kTailPadding = 64;
constexpr std::size_t kUserBytes = 16000;
constexpr std::size_t kMaxSignalBytes = 512;
constexpr int64_t kSendCursor = 1ll << 20;

__device__ ThreadGroup make_group() {
  return ThreadGroup{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};
}

// Reproduces the state a transfer leaves behind when it unwinds part-way
// through: the byte range is reserved (send_cursor advanced) and the slot is
// Active.
__global__ void stage_slot_kernel(
    NvlChannelProgress* slot,
    NvlChannelState* channel) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  channel->send_cursor = kSendCursor;
  slot->activeBaseByte = kBaseByte;
  slot->activeNextByte = kNextByte;
  slot->activePayloadBytes = kPayloadBytes;
  slot->activeTailPadding = kTailPadding;
  slot->activeUserBytes = kUserBytes;
  slot->activeMaxSignalBytes = kMaxSignalBytes;
  slot->stage = NvlProgressStage::Active;
}

__global__ void abandon_kernel(NvlChannelProgress* slot) {
  ThreadGroup g = make_group();
  abandon_progress_state(g, *slot);
}

// Stands in for the next collective queued on this channel. Traps unless the
// preceding abort left the slot idle.
__global__ void reinit_kernel(NvlChannelProgress* slot) {
  ThreadGroup g = make_group();
  assert_progress_slot_idle(g, *slot, /*channel=*/0, "send");
}

__global__ void read_slot_kernel(
    const NvlChannelProgress* slot,
    const NvlChannelState* channel,
    NvlSlotProbe* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  out->stage = static_cast<int>(slot->stage);
  out->baseByte = static_cast<unsigned long long>(slot->activeBaseByte);
  out->nextByte = static_cast<unsigned long long>(slot->activeNextByte);
  out->payloadBytes = static_cast<unsigned long long>(slot->activePayloadBytes);
  out->tailPadding = static_cast<unsigned long long>(slot->activeTailPadding);
  out->userBytes = static_cast<unsigned long long>(slot->activeUserBytes);
  out->maxSignalBytes =
      static_cast<unsigned long long>(slot->activeMaxSignalBytes);
  out->sendCursor = static_cast<long long>(channel->send_cursor);
}

// The slot lives in device global memory, not shared or local, because the
// property under test is that it survives the aborted kernel in a reusable
// state for the next one.
class NvlSlotFixture {
 public:
  NvlSlotFixture() {
    PIPES_CUDA_CHECK(cudaMalloc(&slot_, sizeof(NvlChannelProgress)));
    PIPES_CUDA_CHECK(cudaMalloc(&channel_, sizeof(NvlChannelState)));
    PIPES_CUDA_CHECK(cudaMemset(slot_, 0, sizeof(NvlChannelProgress)));
    PIPES_CUDA_CHECK(cudaMemset(channel_, 0, sizeof(NvlChannelState)));
    PIPES_CUDA_CHECK(cudaMalloc(&probe_, sizeof(NvlSlotProbe)));
    stage_slot_kernel<<<1, 1>>>(slot_, channel_);
    PIPES_KERNEL_LAUNCH_CHECK();
  }

  ~NvlSlotFixture() {
    (void)cudaFree(slot_);
    (void)cudaFree(channel_);
    (void)cudaFree(probe_);
  }

  NvlSlotFixture(const NvlSlotFixture&) = delete;
  NvlSlotFixture& operator=(const NvlSlotFixture&) = delete;

  NvlChannelProgress* slot() {
    return slot_;
  }

  NvlSlotProbe read() {
    read_slot_kernel<<<1, 1>>>(slot_, channel_, probe_);
    PIPES_KERNEL_LAUNCH_CHECK();
    PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    NvlSlotProbe host{};
    PIPES_CUDA_CHECK(
        cudaMemcpy(&host, probe_, sizeof(host), cudaMemcpyDeviceToHost));
    return host;
  }

 private:
  NvlChannelProgress* slot_{nullptr};
  NvlChannelState* channel_{nullptr};
  NvlSlotProbe* probe_{nullptr};
};

} // namespace

NvlSlotProbe stageNvlSlot() {
  NvlSlotFixture fixture;
  return fixture.read();
}

NvlSlotProbe abandonNvlSlotThenReinit() {
  NvlSlotFixture fixture;
  abandon_kernel<<<1, 32>>>(fixture.slot());
  PIPES_KERNEL_LAUNCH_CHECK();
  reinit_kernel<<<1, 32>>>(fixture.slot());
  PIPES_KERNEL_LAUNCH_CHECK();
  return fixture.read();
}

} // namespace comms::prims::test
