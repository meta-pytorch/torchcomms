// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllReduce/AllReducePipesFlatRing.cuh"

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/collectives/RingAllgather.cuh"
#include "comms/pipes/collectives/RingReduceScatter.cuh"

namespace {

template <int NumRings>
__device__ comms::pipes::RingReduceScatterArgs<NumRings, float>
makeReduceScatterArgs(const ctran::allreduce::pipesflatring::KernArgs& args) {
  comms::pipes::RingReduceScatterArgs<NumRings, float> phaseArgs{};
  phaseArgs.my_rank = args.rank;
  phaseArgs.num_ranks = args.nRanks;
  phaseArgs.chunk_elements = args.chunkElements;
  phaseArgs.signaling_data_size = args.signalingDataSize;
  phaseArgs.input = args.sendbuff;
  phaseArgs.output = args.recvbuff + args.rank * args.chunkElements;

  for (int r = 0; r < NumRings; r++) {
    phaseArgs.rings[r] = comms::pipes::RingTopology{
        .prev_rank = args.rings[r].prevRank,
        .next_rank = args.rings[r].nextRank,
        .prev = args.rings[r].prev,
        .next = args.rings[r].next,
    };
  }
  return phaseArgs;
}

template <int NumRings>
__device__ comms::pipes::RingAllgatherArgs<NumRings> makeAllGatherArgs(
    const ctran::allreduce::pipesflatring::KernArgs& args) {
  comms::pipes::RingAllgatherArgs<NumRings> phaseArgs{};
  phaseArgs.my_rank = args.rank;
  phaseArgs.num_ranks = args.nRanks;
  phaseArgs.sendcount = args.chunkElements * sizeof(float);
  phaseArgs.signaling_data_size = args.signalingDataSize;
  phaseArgs.sendbuf = reinterpret_cast<const char*>(
      args.recvbuff + args.rank * args.chunkElements);
  phaseArgs.recvbuf = reinterpret_cast<char*>(args.recvbuff);

  for (int r = 0; r < NumRings; r++) {
    phaseArgs.rings[r] = comms::pipes::RingTopology{
        .prev_rank = args.rings[r].prevRank,
        .next_rank = args.rings[r].nextRank,
        .prev = args.rings[r].prev,
        .next = args.rings[r].next,
    };
  }
  return phaseArgs;
}

template <int NumRings>
__device__ void runReduceScatter(
    const ctran::allreduce::pipesflatring::KernArgs& args) {
  auto phaseArgs = makeReduceScatterArgs<NumRings>(args);
  comms::pipes::ring_reduce_scatter_device<
      NumRings,
      float,
      comms::pipes::SumOp,
      ctran::allreduce::pipesflatring::kTileElems,
      ctran::allreduce::pipesflatring::kBlockSize>(phaseArgs, args.timeout);
}

template <int NumRings>
__device__ void runAllGather(
    const ctran::allreduce::pipesflatring::KernArgs& args) {
  auto phaseArgs = makeAllGatherArgs<NumRings>(args);
  comms::pipes::ring_allgather_device<
      NumRings,
      ctran::allreduce::pipesflatring::kBlockSize>(phaseArgs, args.timeout);
}

} // namespace

__global__
__launch_bounds__(ctran::allreduce::pipesflatring::kBlockSize, 1) void ctranKernelAllReducePipesFlatRingReduceScatter(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::pipesflatring::KernArgs args) {
  if (blockIdx.x >= args.numBlocks) {
    return;
  }

  switch (args.numRings) {
    case 1:
      runReduceScatter<1>(args);
      break;
    case 2:
      runReduceScatter<2>(args);
      break;
    case 4:
      runReduceScatter<4>(args);
      break;
    default:
      break;
  }
}

__global__
__launch_bounds__(ctran::allreduce::pipesflatring::kBlockSize, 1) void ctranKernelAllReducePipesFlatRingAllGather(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::pipesflatring::KernArgs args) {
  if (blockIdx.x >= args.numBlocks) {
    return;
  }

  switch (args.numRings) {
    case 1:
      runAllGather<1>(args);
      break;
    case 2:
      runAllGather<2>(args);
      break;
    case 4:
      runAllGather<4>(args);
      break;
    default:
      break;
  }
}

#endif // ENABLE_PIPES
