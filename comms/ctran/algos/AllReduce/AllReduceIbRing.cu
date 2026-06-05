// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PIPES)

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceIbRing.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceNvlDirect.cuh"
#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"

using namespace ctran::allreduce::nvl;

namespace {

template <typename T, typename RecvCopyOp>
__device__ __forceinline__ void ring_step(
    comms::pipes::P2pIbgdaTransportDevice* nextTransport,
    comms::pipes::P2pIbgdaTransportDevice* prevTransport,
    char* sendSrc,
    size_t sendBytes,
    char* recvDst,
    size_t recvBytes,
    comms::pipes::ThreadGroup& ringGroup,
    int activeGroups,
    size_t sendPipelineWindow,
    size_t recvPipelineWindow,
    comms::pipes::Timeout& timeout) {
  using Status = comms::pipes::IbgdaSendRecvProgressStatus;

  size_t sDone = 0, rDone = 0;
  bool sInitialized = false, rInitialized = false;
  comms::pipes::IbgdaSendRecvProgressState sOp{}, rOp{};
  size_t sWindow = 0, rWindow = 0;

  while (sDone < sendBytes || rDone < recvBytes) {
    if (sDone < sendBytes) {
      if (!sInitialized) {
        sWindow = (sendBytes - sDone < sendPipelineWindow) ? (sendBytes - sDone)
                                                           : sendPipelineWindow;
        sOp = nextTransport->init_send_progress(
            ringGroup, sWindow, activeGroups, 0);
        sInitialized = true;
      }
      auto s = nextTransport->progress_send_once(
          ringGroup, sOp, sendSrc + sDone, timeout);
      if (s == Status::Done) {
        sDone += sWindow;
        sInitialized = false;
      }
    }

    if (rDone < recvBytes) {
      if (!rInitialized) {
        rWindow = (recvBytes - rDone < recvPipelineWindow) ? (recvBytes - rDone)
                                                           : recvPipelineWindow;
        rOp = prevTransport->init_recv_progress(
            ringGroup, rWindow, activeGroups, 0);
        rInitialized = true;
      }
      auto r = prevTransport->template progress_recv_once<RecvCopyOp>(
          ringGroup, rOp, recvDst + rDone, timeout);
      if (r == Status::Done) {
        rDone += rWindow;
        rInitialized = false;
      }
    }
  }
}

template <typename T>
__device__ __noinline__ void phase2IbRing(
    const ctran::allreduce::hierring::KernArgs& args,
    comms::pipes::ThreadGroup& blockGroup) {
  if (args.common.localRank >= args.common.pMin) {
    return;
  }
  if (args.common.nNodes <= 1) {
    return;
  }

  const int W = args.ring.nNodes;
  const int myNode = args.ring.myNodeIdx;
  const size_t actualElems = actualSegElems(
      args.common.count, args.common.segmentElems, args.common.localRank);
  const auto tile = segmentTile(actualElems * sizeof(T), blockGroup);
  const size_t tileElems = tile.bytes / sizeof(T);

  auto* prevTransport = args.common.transports[args.ring.prevRank].p2p_ibgda;
  auto* nextTransport = args.common.transports[args.ring.nextRank].p2p_ibgda;
  const int activeGroups =
      args.common.numBlocks * ctran::allreduce::hierring::kRingLanes;

  auto ringGroup = blockGroup;
  ringGroup.group_id = blockGroup.group_id;
  ringGroup.total_groups = static_cast<uint32_t>(activeGroups);

  const size_t sendPipelineWindow =
      nextTransport->pipeline_window(activeGroups);
  const size_t recvPipelineWindow =
      prevTransport->pipeline_window(activeGroups);

  T* tileBuf =
      static_cast<T*>(args.common.phase2Buf) + (tile.offsetBytes / sizeof(T));
  const size_t baseSubChunkElems = tileElems / W;
  comms::pipes::Timeout timeout{};

  auto subChunkOffset = [&](int idx) -> size_t {
    return idx * baseSubChunkElems;
  };
  auto subChunkElems = [&](int idx) -> size_t {
    return (idx == W - 1) ? (tileElems - baseSubChunkElems * (W - 1))
                          : baseSubChunkElems;
  };
  char* base = reinterpret_cast<char*>(tileBuf);

  for (int step = 0; step < W - 1; ++step) {
    int sendIdx = (myNode - 1 - step + W) % W;
    int recvIdx = (myNode - 2 - step + W) % W;
    ring_step<T, IbReduceCopy<T>>(
        nextTransport,
        prevTransport,
        base + subChunkOffset(sendIdx) * sizeof(T),
        subChunkElems(sendIdx) * sizeof(T),
        base + subChunkOffset(recvIdx) * sizeof(T),
        subChunkElems(recvIdx) * sizeof(T),
        ringGroup,
        activeGroups,
        sendPipelineWindow,
        recvPipelineWindow,
        timeout);
  }

  for (int step = 0; step < W - 1; ++step) {
    int sendIdx = (myNode - step + W) % W;
    int recvIdx = (myNode - 1 - step + W) % W;
    ring_step<T, comms::pipes::Memcpy>(
        nextTransport,
        prevTransport,
        base + subChunkOffset(sendIdx) * sizeof(T),
        subChunkElems(sendIdx) * sizeof(T),
        base + subChunkOffset(recvIdx) * sizeof(T),
        subChunkElems(recvIdx) * sizeof(T),
        ringGroup,
        activeGroups,
        sendPipelineWindow,
        recvPipelineWindow,
        timeout);
  }
}

template <typename T>
__device__ __forceinline__ void run_allreduce_hierarchical_ring(
    const ctran::allreduce::hierring::KernArgs& args,
    comms::pipes::ThreadGroup& group) {
  runAllReduceFused<T>(
      args.common, group, [&](comms::pipes::ThreadGroup& phaseGroup) {
        phase2IbRing<T>(args, phaseGroup);
      });
}

} // namespace

__global__
__launch_bounds__(ctran::allreduce::hierring::kBlockSize, 1) void ctranKernelAllReduceHierarchicalRing(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::hierring::KernArgs args) {
  auto blockGroup = comms::pipes::make_block_group();
  const int blockId = static_cast<int>(blockIdx.x);
  if (blockId >= args.common.numBlocks) {
    return;
  }
  auto group = logicalDataGroup(blockGroup, blockId, args.common.numBlocks);

  if (args.common.datatype == commFloat32) {
    run_allreduce_hierarchical_ring<float>(args, group);
  }
}

#endif // ENABLE_PIPES
