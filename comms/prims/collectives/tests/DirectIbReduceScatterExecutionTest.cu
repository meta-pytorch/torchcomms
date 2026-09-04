// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/tests/DirectIbReduceScatterExecutionTest.cuh"

#include <cuda_runtime.h>

#include "comms/prims/collectives/ReduceScatterDirectIbExecution.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims::test {
namespace {

constexpr int kBlockThreads = 128;
using ReduceOp = TileReduceStaged<float, SumOp, 1024, kBlockThreads>;

constexpr int kTraceIbRank = 1;
constexpr int kTraceIbSize = 4;
constexpr std::size_t kTraceStrideElements = 7;
constexpr std::size_t kTraceOffsetElements = 2;
constexpr std::size_t kTraceRangeElements = 3;

struct UnusedTransport {
  template <typename Op>
  __device__ void recv(
      ThreadGroup&,
      char*,
      std::size_t,
      std::size_t,
      const AbortDevice&,
      const char*) {}

  __device__ void send(
      ThreadGroup&,
      const char*,
      std::size_t,
      std::size_t,
      const AbortDevice&) {}
};

__global__ void directIbSingleRankRangeKernel(
    const float* input,
    std::size_t strideElements,
    float* output,
    std::size_t rangeOffsetElements,
    std::size_t rangeElements,
    ReduceScatterOutputInitialization initialization) {
  auto group = make_block_group();
  direct_ib_reduce_scatter_role_range<
      float,
      DirectIbReceiveReduction<ReduceOp>>(
      /*ibRank=*/0,
      /*ibSize=*/1,
      DirectIbStridedInput<float>{
          .data = input,
          .chunkStrideBytes = strideElements * sizeof(float),
      },
      rangeOffsetElements,
      DirectIbOutput<float>{
          .data = output,
          .initialization = initialization,
      },
      rangeElements,
      DirectIbReceiveReduction<ReduceOp>{},
      /*signalingBytes=*/0,
      /*peers=*/static_cast<const UnusedTransport*>(nullptr),
      group,
      DirectIbReduceScatterRole::RECEIVE,
      AbortDevice{});
}

struct RecordingTransport {
  int peer{0};
  const char* ownSource{nullptr};
  const char* output{nullptr};
  DirectIbExecutionTrace* trace{nullptr};

  template <typename Op>
  __device__ void recv(
      ThreadGroup& group,
      char*,
      std::size_t,
      std::size_t,
      const AbortDevice&,
      const char* localInput) {
    if (group.thread_id_in_group != 0) {
      return;
    }
    const auto channel = static_cast<int>(group.group_id);
    const int step = trace->recvCount[channel]++;
    trace->recvPeers[channel][step] = peer;
    trace->recvInputKind[channel][step] = localInput == ownSource
        ? kDirectIbTraceOwnInput
        : (localInput == output ? kDirectIbTraceOutput : 0);
  }

  __device__ void send(
      ThreadGroup& group,
      const char* data,
      std::size_t,
      std::size_t,
      const AbortDevice&) {
    if (group.thread_id_in_group != 0) {
      return;
    }
    const auto channel = static_cast<int>(group.group_id);
    const int step = trace->sendCount[channel]++;
    trace->sendPeers[channel][step] = peer;
    trace->sendFirstValue[channel][step] =
        *reinterpret_cast<const float*>(data);
  }
};

struct RecordingPeerAccessor {
  const char* ownSource{nullptr};
  const char* output{nullptr};
  DirectIbExecutionTrace* trace{nullptr};

  __device__ RecordingTransport operator[](int peer) const {
    return RecordingTransport{
        .peer = peer,
        .ownSource = ownSource,
        .output = output,
        .trace = trace,
    };
  }
};

template <
    bool kStaggerChannels,
    ReduceScatterOutputInitialization kInitialization>
__global__ void directIbPeerWalkTraceKernel(
    const float* input,
    float* output,
    DirectIbExecutionTrace* trace) {
  auto group = make_block_group();
  const auto channel = static_cast<std::size_t>(group.group_id);
  const float* ownSource =
      input + kTraceIbRank * kTraceStrideElements + kTraceOffsetElements;
  float* channelOutput = output + channel * kTraceRangeElements;
  const RecordingPeerAccessor peers{
      .ownSource = reinterpret_cast<const char*>(ownSource),
      .output = reinterpret_cast<const char*>(channelOutput),
      .trace = trace,
  };

  direct_ib_reduce_scatter_role_range<
      float,
      DirectIbReceiveReduction<ReduceOp>,
      kStaggerChannels>(
      kTraceIbRank,
      kTraceIbSize,
      DirectIbStridedInput<float>{
          .data = input,
          .chunkStrideBytes = kTraceStrideElements * sizeof(float),
      },
      kTraceOffsetElements,
      DirectIbOutput<float>{
          .data = channelOutput,
          .initialization = kInitialization,
      },
      kTraceRangeElements,
      DirectIbReceiveReduction<ReduceOp>{},
      /*signalingBytes=*/17,
      peers,
      group,
      DirectIbReduceScatterRole::RECEIVE,
      AbortDevice{});
  group.sync();
  direct_ib_reduce_scatter_role_range<
      float,
      DirectIbReceiveReduction<ReduceOp>,
      kStaggerChannels>(
      kTraceIbRank,
      kTraceIbSize,
      DirectIbStridedInput<float>{
          .data = input,
          .chunkStrideBytes = kTraceStrideElements * sizeof(float),
      },
      kTraceOffsetElements,
      DirectIbOutput<float>{
          .data = channelOutput,
          .initialization = kInitialization,
      },
      kTraceRangeElements,
      DirectIbReceiveReduction<ReduceOp>{},
      /*signalingBytes=*/17,
      peers,
      group,
      DirectIbReduceScatterRole::SEND,
      AbortDevice{});
}

} // namespace

void launchDirectIbSingleRankRange(
    const float* input,
    std::size_t strideElements,
    float* output,
    std::size_t rangeOffsetElements,
    std::size_t rangeElements,
    bool outputAlreadyInitialized) {
  directIbSingleRankRangeKernel<<<1, kBlockThreads>>>(
      input,
      strideElements,
      output,
      rangeOffsetElements,
      rangeElements,
      outputAlreadyInitialized
          ? ReduceScatterOutputInitialization::ALREADY_INITIALIZED
          : ReduceScatterOutputInitialization::COPY_OWN_INPUT);
}

void launchDirectIbPeerWalkTrace(
    const float* input,
    float* output,
    DirectIbExecutionTrace* traces) {
  directIbPeerWalkTraceKernel<
      false,
      ReduceScatterOutputInitialization::COPY_OWN_INPUT>
      <<<kDirectIbTraceChannels, kBlockThreads>>>(input, output, traces);
  directIbPeerWalkTraceKernel<
      true,
      ReduceScatterOutputInitialization::COPY_OWN_INPUT>
      <<<kDirectIbTraceChannels, kBlockThreads>>>(input, output, traces + 1);
  directIbPeerWalkTraceKernel<
      true,
      ReduceScatterOutputInitialization::ALREADY_INITIALIZED>
      <<<kDirectIbTraceChannels, kBlockThreads>>>(input, output, traces + 2);
}

} // namespace comms::prims::test
