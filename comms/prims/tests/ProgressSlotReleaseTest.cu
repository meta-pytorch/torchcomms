// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/ProgressSlotReleaseTest.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"

namespace comms::prims::test {
namespace {

using detail::IbSendRecvProgressStage;

static_assert(static_cast<int>(IbSendRecvProgressStage::Done) == kStageDone);
static_assert(
    static_cast<int>(IbSendRecvProgressStage::WaitLocalCompletion) ==
    kStageWaitLocalCompletion);
static_assert(
    static_cast<int>(IbSendRecvProgressStage::WaitSlotFree) ==
    kStageWaitSlotFree);
static_assert(
    static_cast<int>(IbSendRecvProgressStage::WaitDataReady) ==
    kStageWaitDataReady);

// Arbitrary non-zero cursors. They exist only so the probe can tell an
// abandoned slot from an untouched one.
constexpr std::size_t kTailPadding = 64;
constexpr int64_t kBaseStep = 4096;
constexpr int64_t kNextStep = 1ll << 20;

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
// through: the byte range is reserved (nextStep advanced) and the state machine
// sits in a non-terminal stage.
__global__ void stage_slot_kernel(
    int startStage,
    std::size_t startNextByte,
    IbChannelProgress* slot) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  slot->nextStep = kNextStep;
  slot->activeNextByte = startNextByte;
  slot->activeTailPadding = kTailPadding;
  slot->activeBaseStep = kBaseStep;
  slot->activeStage = static_cast<IbSendRecvProgressStage>(startStage);
  slot->activeVariableSize = false;
}

__global__ void abandon_kernel(IbChannelProgress* slot) {
  ThreadGroup g = make_group();
  IbChannelProgress state = *slot;
  detail::abandon_progress_state(g, *slot, state);
}

// Stands in for the next collective queued on this channel. Traps unless the
// preceding abort left the slot idle.
__global__ void reinit_kernel(IbChannelProgress* slot) {
  ThreadGroup g = make_group();
  detail::assert_progress_slot_idle(g, *slot, "send");
}

__global__ void read_slot_kernel(
    const IbChannelProgress* slot,
    SlotProbe* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  out->stage = static_cast<int>(slot->activeStage);
  out->nextByte = static_cast<unsigned long long>(slot->activeNextByte);
  out->tailPadding = static_cast<unsigned long long>(slot->activeTailPadding);
  out->baseStep = static_cast<long long>(slot->activeBaseStep);
  out->nextStep = static_cast<long long>(slot->nextStep);
}

// The slot lives in device global memory, not shared or local, because the
// property under test is that it survives the aborted kernel in a reusable
// state for the next one.
class SlotFixture {
 public:
  SlotFixture(int startStage, std::size_t startNextByte) {
    PIPES_CUDA_CHECK(cudaMalloc(&slot_, sizeof(IbChannelProgress)));
    PIPES_CUDA_CHECK(cudaMalloc(&probe_, sizeof(SlotProbe)));
    stage_slot_kernel<<<1, 1>>>(startStage, startNextByte, slot_);
    PIPES_KERNEL_LAUNCH_CHECK();
  }

  ~SlotFixture() {
    (void)cudaFree(slot_);
    (void)cudaFree(probe_);
  }

  SlotFixture(const SlotFixture&) = delete;
  SlotFixture& operator=(const SlotFixture&) = delete;

  IbChannelProgress* slot() {
    return slot_;
  }

  SlotProbe read() {
    read_slot_kernel<<<1, 1>>>(slot_, probe_);
    PIPES_KERNEL_LAUNCH_CHECK();
    PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    SlotProbe host{};
    PIPES_CUDA_CHECK(
        cudaMemcpy(&host, probe_, sizeof(host), cudaMemcpyDeviceToHost));
    return host;
  }

 private:
  IbChannelProgress* slot_{nullptr};
  SlotProbe* probe_{nullptr};
};

} // namespace

SlotProbe stage_slot(int startStage, std::size_t startNextByte) {
  SlotFixture fixture(startStage, startNextByte);
  return fixture.read();
}

SlotProbe abandon_then_reinit(int startStage, std::size_t startNextByte) {
  SlotFixture fixture(startStage, startNextByte);
  abandon_kernel<<<1, 32>>>(fixture.slot());
  PIPES_KERNEL_LAUNCH_CHECK();
  reinit_kernel<<<1, 32>>>(fixture.slot());
  PIPES_KERNEL_LAUNCH_CHECK();
  return fixture.read();
}

} // namespace comms::prims::test
