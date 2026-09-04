// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectIbExecution.cuh"
#include "comms/prims/collectives/tests/DirectIbReduceScatterExecutionTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::test {
namespace {

TEST(DirectIbReduceScatterExecutionTest, StridedInputMapsPhaseRanks) {
  float values[16]{};
  const DirectIbStridedInput<float> input{
      .data = values,
      .chunkStrideBytes = 5 * sizeof(float),
  };
  EXPECT_EQ(input.chunkData(0), values);
  EXPECT_EQ(input.chunkData(1), values + 5);
  EXPECT_EQ(input.chunkData(2), values + 10);
}

TEST(DirectIbReduceScatterExecutionTest, CopiesOffsetRangeFromPaddedChunk) {
  constexpr std::size_t kChunkElements = 19;
  constexpr std::size_t kStrideElements = 24;
  constexpr std::size_t kOffsetElements = 3;
  constexpr std::size_t kRangeElements = 13;
  std::vector<float> input(kStrideElements, -1.0F);
  for (std::size_t index = 0; index < kChunkElements; ++index) {
    input[index] = static_cast<float>(index) + 0.25F;
  }
  std::vector<float> output(kRangeElements, -2.0F);
  meta::comms::DeviceBuffer inputDevice(input.size() * sizeof(float));
  meta::comms::DeviceBuffer outputDevice(output.size() * sizeof(float));
  CUDACHECK_TEST(cudaMemcpy(
      inputDevice.get(),
      input.data(),
      input.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  launchDirectIbSingleRankRange(
      static_cast<const float*>(inputDevice.get()),
      kStrideElements,
      static_cast<float*>(outputDevice.get()),
      kOffsetElements,
      kRangeElements,
      /*outputAlreadyInitialized=*/false);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaMemcpy(
      output.data(),
      outputDevice.get(),
      output.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (std::size_t index = 0; index < kRangeElements; ++index) {
    EXPECT_EQ(output[index], input[kOffsetElements + index]);
  }
}

TEST(DirectIbReduceScatterExecutionTest, PreservesInitializedAndEmptyRanges) {
  constexpr std::size_t kElements = 17;
  const std::vector<float> input(kElements, 3.0F);
  const std::vector<float> expected(kElements, 7.0F);
  std::vector<float> output(kElements);
  meta::comms::DeviceBuffer inputDevice(kElements * sizeof(float));
  meta::comms::DeviceBuffer outputDevice(kElements * sizeof(float));
  CUDACHECK_TEST(cudaMemcpy(
      inputDevice.get(),
      input.data(),
      kElements * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      outputDevice.get(),
      expected.data(),
      kElements * sizeof(float),
      cudaMemcpyHostToDevice));

  launchDirectIbSingleRankRange(
      static_cast<const float*>(inputDevice.get()),
      kElements,
      static_cast<float*>(outputDevice.get()),
      /*rangeOffsetElements=*/0,
      kElements,
      /*outputAlreadyInitialized=*/true);
  launchDirectIbSingleRankRange(
      /*input=*/nullptr,
      /*strideElements=*/0,
      /*output=*/nullptr,
      /*rangeOffsetElements=*/0,
      /*rangeElements=*/0,
      /*outputAlreadyInitialized=*/false);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaMemcpy(
      output.data(),
      outputDevice.get(),
      output.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(output, expected);
}

TEST(DirectIbReduceScatterExecutionTest, ExecutesPeerWalksAndInitialization) {
  constexpr std::size_t kStrideElements = 7;
  constexpr std::size_t kOffsetElements = 2;
  constexpr std::size_t kRanks = 4;
  std::vector<float> input(kRanks * kStrideElements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(index) + 0.5F;
  }

  meta::comms::DeviceBuffer inputDevice(input.size() * sizeof(float));
  meta::comms::DeviceBuffer outputDevice(
      kDirectIbTraceChannels * 3 * sizeof(float));
  meta::comms::DeviceBuffer traceDevice(3 * sizeof(DirectIbExecutionTrace));
  CUDACHECK_TEST(cudaMemcpy(
      inputDevice.get(),
      input.data(),
      input.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemset(traceDevice.get(), 0, 3 * sizeof(DirectIbExecutionTrace)));

  launchDirectIbPeerWalkTrace(
      static_cast<const float*>(inputDevice.get()),
      static_cast<float*>(outputDevice.get()),
      static_cast<DirectIbExecutionTrace*>(traceDevice.get()));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DirectIbExecutionTrace traces[3]{};
  CUDACHECK_TEST(cudaMemcpy(
      traces, traceDevice.get(), sizeof(traces), cudaMemcpyDeviceToHost));
  const int expectedRecvPeers[3][kDirectIbTraceChannels][kDirectIbTracePeers] =
      {
          {{2, 3, 0}, {2, 3, 0}},
          {{2, 3, 0}, {3, 0, 2}},
          {{2, 3, 0}, {3, 0, 2}},
      };
  const int expectedSendPeers[3][kDirectIbTraceChannels][kDirectIbTracePeers] =
      {
          {{0, 3, 2}, {0, 3, 2}},
          {{0, 3, 2}, {3, 2, 0}},
          {{0, 3, 2}, {3, 2, 0}},
      };
  for (int mode = 0; mode < 3; ++mode) {
    for (int channel = 0; channel < kDirectIbTraceChannels; ++channel) {
      EXPECT_EQ(traces[mode].recvCount[channel], kDirectIbTracePeers);
      EXPECT_EQ(traces[mode].sendCount[channel], kDirectIbTracePeers);
      for (int step = 0; step < kDirectIbTracePeers; ++step) {
        EXPECT_EQ(
            traces[mode].recvPeers[channel][step],
            expectedRecvPeers[mode][channel][step]);
        const int sendPeer = expectedSendPeers[mode][channel][step];
        const std::size_t sendOffset =
            static_cast<std::size_t>(sendPeer) * kStrideElements +
            kOffsetElements;
        EXPECT_EQ(traces[mode].sendPeers[channel][step], sendPeer);
        EXPECT_EQ(
            traces[mode].sendFirstValue[channel][step], input[sendOffset]);
        EXPECT_EQ(
            traces[mode].recvInputKind[channel][step],
            mode < 2 && step == 0 ? kDirectIbTraceOwnInput
                                  : kDirectIbTraceOutput);
      }
    }
  }
}

} // namespace
} // namespace comms::prims::test
