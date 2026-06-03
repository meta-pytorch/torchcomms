// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(ENABLE_PIPES)

#include <cstddef>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {
class P2pIbgdaTransportDevice;
}

namespace ctran::allreduce::pipesflatring {

static constexpr int kBlockSize = 512;
static constexpr int kTileElems = 16384;
static constexpr int kMaxRings = 4;

struct Ring {
  int prevRank{0};
  int nextRank{0};
  comms::pipes::P2pIbgdaTransportDevice* prev{nullptr};
  comms::pipes::P2pIbgdaTransportDevice* next{nullptr};
};

struct KernArgs {
  const float* sendbuff{nullptr};
  float* recvbuff{nullptr};
  size_t count{0};
  size_t chunkElements{0};
  int rank{0};
  int nRanks{0};
  int numRings{0};
  int numBlocks{0};
  size_t signalingDataSize{0};
  comms::pipes::Timeout timeout{};
  Ring rings[kMaxRings]{};
};

} // namespace ctran::allreduce::pipesflatring

__global__ void ctranKernelAllReducePipesFlatRingReduceScatter(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::pipesflatring::KernArgs args);

__global__ void ctranKernelAllReducePipesFlatRingAllGather(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::pipesflatring::KernArgs args);

#endif // ENABLE_PIPES
