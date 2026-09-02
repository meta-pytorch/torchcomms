// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/P2pNvlProgressTrapTest.cuh"

#include <chrono>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/nvl/NvlChannelProgress.cuh"
#include "comms/prims/transport/nvl/NvlChannelState.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims::test {
namespace {

constexpr int kMaxChannels = 1;
constexpr std::size_t kPipelineDepth = 2;
constexpr std::size_t kPerChannelBuffer = 128 * 1024;
constexpr std::size_t kPerChannelSlot = kPerChannelBuffer / kPipelineDepth;
// Larger than the pipeline window, so the send necessarily reaches the
// backpressure branch where the abort check lives.
constexpr std::size_t kPayloadBytes = 4 * kPerChannelBuffer;
// Bounded so a non-trapping implementation fails the test instead of hanging.
constexpr int kMaxIterations = 256;

__global__ void nvlProgressTrapKernel(
    P2pNvlTransportDevice p2p,
    NvlProgressTrapCase testCase,
    char* src,
    AbortDevice abort) {
  abort.start();
  auto group = make_block_group();

  switch (testCase) {
    case NvlProgressTrapCase::NullProgressStorage:
      p2p.init_send_progress(group, kPayloadBytes, 0);
      break;
    case NvlProgressTrapCase::ReinitWhileActive:
      p2p.init_send_progress(group, kPayloadBytes, 0);
      p2p.init_send_progress(group, kPayloadBytes, 0);
      break;
    case NvlProgressTrapCase::AbortTrapBehavior:
      p2p.init_send_progress(group, kPayloadBytes, 0);
      for (int i = 0; i < kMaxIterations; ++i) {
        const auto status =
            p2p.progress_send_once(group, src, kPayloadBytes, 0, abort);
        if (status == NvlSendRecvProgressStatus::Done ||
            status == NvlSendRecvProgressStatus::Aborted) {
          break;
        }
      }
      break;
  }
}

} // namespace

cudaError_t launchNvlProgressTrap(NvlProgressTrapCase testCase) {
  const std::size_t stagingBytes = kMaxChannels * kPerChannelBuffer;

  char* staging = nullptr;
  char* src = nullptr;
  NvlChannelState* channels = nullptr;
  NvlChannelProgress* sendProgress = nullptr;
  NvlChannelProgress* recvProgress = nullptr;
  SignalState* signals = nullptr;

  // NOLINTBEGIN(facebook-cuda-safe-api-call-check)
  cudaMalloc(reinterpret_cast<void**>(&staging), stagingBytes);
  cudaMalloc(reinterpret_cast<void**>(&src), kPayloadBytes);
  cudaMalloc(
      reinterpret_cast<void**>(&channels),
      kMaxChannels * sizeof(NvlChannelState));
  cudaMalloc(reinterpret_cast<void**>(&signals), sizeof(SignalState));
  cudaMemset(staging, 0, stagingBytes);
  cudaMemset(src, 0xA5, kPayloadBytes);
  cudaMemset(channels, 0, kMaxChannels * sizeof(NvlChannelState));
  cudaMemset(signals, 0, sizeof(SignalState));

  // The null-storage case is the one that must NOT have progress arrays.
  if (testCase != NvlProgressTrapCase::NullProgressStorage) {
    cudaMalloc(
        reinterpret_cast<void**>(&sendProgress),
        kMaxChannels * sizeof(NvlChannelProgress));
    cudaMalloc(
        reinterpret_cast<void**>(&recvProgress),
        kMaxChannels * sizeof(NvlChannelProgress));
    cudaMemset(sendProgress, 0, kMaxChannels * sizeof(NvlChannelProgress));
    cudaMemset(recvProgress, 0, kMaxChannels * sizeof(NvlChannelProgress));
  }
  // NOLINTEND(facebook-cuda-safe-api-call-check)

  const P2pNvlTransportOptions options{
      .dataBufferSize = stagingBytes,
      .pipelineDepth = kPipelineDepth,
      .per_channel_buffer = kPerChannelBuffer,
      .per_channel_slot = kPerChannelSlot,
      .max_num_channels = kMaxChannels,
  };

  // Self-loopback: local and remote staging are the same allocation. Nothing
  // returns SLOT_FREE credit, which is exactly the stall these cases need.
  const LocalState localState{
      .dataBuffer = staging,
      .signalBuffer = DeviceSpan<SignalState>(signals, 1),
      .barrierBuffer = DeviceSpan<BarrierState>(nullptr, 0),
  };
  const RemoteState remoteState{
      .dataBuffer = staging,
      .signalBuffer = DeviceSpan<SignalState>(signals, 1),
      .barrierBuffer = DeviceSpan<BarrierState>(nullptr, 0),
  };

  P2pNvlTransportDevice p2p(
      /*myRank=*/0,
      /*peerRank=*/1,
      options,
      localState,
      remoteState,
      channels,
      channels,
      sendProgress,
      recvProgress);

  // TRAP behavior with the abort already recorded on the host, so the first
  // device-side check observes it and the test does not race a deadline.
  comms::fault_tolerance::Abort abort{
      /*enabled=*/true, comms::fault_tolerance::AbortBehavior::TRAP};
  abort.startTimeout(std::chrono::milliseconds{0});
  static_cast<void>(abort.isAborted());

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  nvlProgressTrapKernel<<<1, 256>>>(
      p2p, testCase, src, abort.getDeviceHandle());

  // Synchronize here, not in the caller. `abort` is destroyed on return and its
  // destructor frees the CUDA-mapped state the kernel reads through the
  // non-owning device handle. The device buffers above are deliberately left
  // allocated: a trap leaves the context unusable, and the caller resets it.
  return cudaDeviceSynchronize();
}

} // namespace comms::prims::test
