// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/AllReduce/AllReduceRingCommon.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/utils/commSpecs.h"

namespace ctran::allreduce::ring {

namespace {

using ctran::algos::GpeKernelSyncDev::checkPost;
using ctran::algos::GpeKernelSyncDev::complete;
using ctran::algos::GpeKernelSyncDev::waitPost;

} // namespace

#define ROUND_LOG_PREFIX_FMT "partition %d round %d/%d step %d/%d "
#define ROUND_LOG_PREFIX_VAL(algoCtx, phase, op, round, opStep)            \
  algoCtx.partition, round, algoCtx.opRounds[op].totalRounds, opStep.step, \
      algoCtx.numSteps

template <typename T>
__device__ __forceinline__ T* getBufAtByteOffset(void* buf, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(buf) + offset);
}
template <typename T>
__device__ __forceinline__ const T* getBufAtByteOffset(
    const void* buf,
    size_t offset) {
  return reinterpret_cast<const T*>(
      reinterpret_cast<const char*>(buf) + offset);
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void _progressRecv(
    ctran::allreduce::ring::KernArgs& args,
    AlgoContext& algoCtx) {
  OpRound& opRound = algoCtx.opRounds[Op::kRecvRedCopy];
  int round = opRound.done;
  if (round >= opRound.totalRounds) {
    // Already finished all rounds
    return;
  }
  // Wait for host side to post the request
  if (!checkPost(args.recvRedCopySync, blockIdx.x, round)) {
    // TODO: FT CHECK_ABORT
    return;
  }

  // Use only done counters for tracking kernels side since redCopy is blocking
  const OpStep& opStep = opRound.doneStep;
  const int tmpChunkId = getTmpChunkId(algoCtx, round);
  const RoundArgs roundArgs =
      getRoundArgs<Op::kRecvRedCopy>(algoCtx, round, opStep);
  const int shardId = roundArgs.shardId;
  const Phase phase = getPhase(algoCtx, opStep.step);

  // Forward to tmpSendBuf ater RecvFwdReady becomes true and before last step
  // in AllGather phase
  const bool isRecvFwd_ = isRecvFwd(algoCtx, opStep.step);
  const int fwdRound = isRecvFwd_ ? getRecvFwdSendRound(algoCtx, round) : -1;
  const int tmpFwdChunkId = isRecvFwd_ ? getTmpChunkId(algoCtx, fwdRound) : -1;

  // Update data from last step in ReduceScatter
  const int rsSteps = algoCtx.nRanks - 1;
  const bool updateData = !isRecvFwd_ || opStep.step >= rsSteps - 1;

  T* recv_data = getBufAtByteOffset<T>(args.recvbuff, roundArgs.dataOffset);
  const T* send_data =
      getBufAtByteOffset<T>(args.sendbuff, roundArgs.dataOffset);
  const T* tmpRecvBuf =
      getBufAtByteOffset<T>(args.tmpRecvBuf, tmpChunkId * args.chunkSize);
  T* tmpSendBuf =
      getBufAtByteOffset<T>(args.tmpSendBuf, tmpFwdChunkId * args.chunkSize);

  CTRAN_DEV_TRACE(
      ROUND_LOG_PREFIX_FMT
      "posted tmpChunkId %d to recvDataOffset %ld shardId %d dataChunk %d tmpFwdChunkId %d fwdRound %d, isRecvFwd %d updateData %d\n",
      ROUND_LOG_PREFIX_VAL(algoCtx, phase, Op::kRecvRedCopy, round, opStep),
      tmpChunkId,
      roundArgs.dataOffsetElem,
      shardId,
      roundArgs.shardDataChunkId,
      tmpFwdChunkId,
      fwdRound,
      isRecvFwd_,
      updateData);

  CTRAN_DEV_TRACE(
      ROUND_LOG_PREFIX_FMT
      "    [%s] tmpRecvBuf %p data %p -> data %p, tmpSendBuf %p, recvNumel %ld\n",
      ROUND_LOG_PREFIX_VAL(algoCtx, phase, Op::kRecvRedCopy, round, opStep),
      phase == Phase::kReduceScatter ? "Reduce" : "Copy",
      tmpRecvBuf,
      send_data,
      updateData ? recv_data : nullptr,
      isRecvFwd_ ? tmpSendBuf : nullptr,
      roundArgs.numel);

  if (phase == Phase::kReduceScatter) {
    const T* srcs[2] = {send_data, tmpRecvBuf};
    if (isRecvFwd_ && !updateData) { // steps [0, n-1)
      // update only next step's sendBuf
      if constexpr (RedOp == commAvg) {
        localReduce<T, commSum>(
            2, srcs, tmpSendBuf, roundArgs.numel, algoCtx.nRanks);
      } else {
        localReduce<T, RedOp>(
            2, srcs, tmpSendBuf, roundArgs.numel, algoCtx.nRanks);
      }
    } else if (isRecvFwd_ && updateData) { // step n-1
      // update both next step's sendBuf and data
      T* dsts[2] = {recv_data, tmpSendBuf};
      localReduce<T, RedOp>(2, srcs, 2, dsts, roundArgs.numel, algoCtx.nRanks);
    }
  } else {
    if (isRecvFwd_ && updateData) { // steps [n, 2n-2)
      // all gather pipelining
      // copy recvBuf to both sendBuf and data
      ctranKernCopyMultiDestRaw<T>(
          tmpRecvBuf,
          recv_data,
          tmpSendBuf,
          roundArgs.numel,
          blockIdx.x,
          gridDim.x);
    } else if (!isRecvFwd_ && updateData) { // step 2n-2
      // copy recvBuf to only data
      ctranKernCopyRaw<T>(
          tmpRecvBuf, recv_data, roundArgs.numel, blockIdx.x, gridDim.x);
    }
  }

  // Notify host side its completion
  complete(args.recvRedCopySync, blockIdx.x, round);
  opUpdateDone<Op::kRecvRedCopy>(algoCtx);

  CTRAN_DEV_TRACE("completed\n");
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void _progressSend(
    ctran::allreduce::ring::KernArgs& args,
    AlgoContext& algoCtx) {
  OpRound& opRound = algoCtx.opRounds[Op::kSendCopy];
  int round = opRound.done;
  if (round >= opRound.totalRounds) {
    // Already finished all rounds
    return;
  }
  // Wait for host side to post the request
  if (!checkPost(args.sendCopySync, blockIdx.x, round)) {
    // TODO: FT CHECK_ABORT
    return;
  }

  // Use only done counters for tracking kernels side since copy is blocking
  const OpStep& opStep = opRound.doneStep;
  const int tmpChunkId = getTmpChunkId(algoCtx, round);
  const RoundArgs roundArgs =
      getRoundArgs<Op::kSendCopy>(algoCtx, round, opStep);
  const int shardId = roundArgs.shardId;
  // TODO: used in CTRAN_DEV_TRACE, which is not yet relevant for amd
  [[maybe_unused]] const Phase phase = getPhase(algoCtx, opStep.step);

  const T* send_data =
      getBufAtByteOffset<T>(args.sendbuff, roundArgs.dataOffset);
  T* tmpSendBuf =
      getBufAtByteOffset<T>(args.tmpSendBuf, tmpChunkId * args.chunkSize);

  CTRAN_DEV_TRACE(
      ROUND_LOG_PREFIX_FMT
      "posted dataOffsetElem %ld shardId %d dataChunk %d to tmpChunkId %d, data %p -> tmpSendBuf %p, sendNumel %ld\n",
      ROUND_LOG_PREFIX_VAL(algoCtx, phase, Op::kSendCopy, round, opStep),
      roundArgs.dataOffsetElem,
      shardId,
      roundArgs.shardDataChunkId,
      tmpChunkId,
      send_data,
      tmpSendBuf,
      roundArgs.numel);

  ctranKernCopy<T>(
      send_data, tmpSendBuf, roundArgs.numel, blockIdx.x, gridDim.x);

  // Notify host side its completion
  complete(args.sendCopySync, blockIdx.x, round);
  opUpdateDone<Op::kSendCopy>(algoCtx);
}

template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__device__ __forceinline__ void _progressRevSend(
    [[maybe_unused]] ctran::allreduce::ring::KernArgs& args,
    [[maybe_unused]] AlgoContext& algoCtx) {
  if constexpr (!EnableBidirAg) {
    return;
  }
  OpRound& opRound = algoCtx.opRounds[Op::kRevSendCopy];
  int round = opRound.done;
  if (round >= opRound.totalRounds) {
    return;
  }
  if (!checkPost(args.revSendCopySync, blockIdx.x, round)) {
    return;
  }

  const OpStep& opStep = opRound.doneStep;
  const int tmpChunkId = getTmpChunkId(algoCtx, round);
  const RoundArgs roundArgs =
      getRevRoundArgs<Op::kRevSendCopy>(algoCtx, round, opStep);

  // Copy rank's reduced shard from recvbuff to tmpSendBufRev
  const T* recv_data =
      getBufAtByteOffset<T>(args.recvbuff, roundArgs.dataOffset);
  T* tmpSendBufRev =
      getBufAtByteOffset<T>(args.tmpSendBufRev, tmpChunkId * args.chunkSize);

  ctranKernCopy<T>(
      recv_data, tmpSendBufRev, roundArgs.numel, blockIdx.x, gridDim.x);

  complete(args.revSendCopySync, blockIdx.x, round);
  opUpdateDone<Op::kRevSendCopy>(algoCtx);
}

template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__device__ __forceinline__ void _progressRevRecv(
    [[maybe_unused]] ctran::allreduce::ring::KernArgs& args,
    [[maybe_unused]] AlgoContext& algoCtx) {
  if constexpr (!EnableBidirAg) {
    return;
  }
  OpRound& opRound = algoCtx.opRounds[Op::kRevRecvCopy];
  int round = opRound.done;
  if (round >= opRound.totalRounds) {
    return;
  }
  if (!checkPost(args.revRecvCopySync, blockIdx.x, round)) {
    return;
  }

  const OpStep& opStep = opRound.doneStep;
  const int tmpChunkId = getTmpChunkId(algoCtx, round);
  const RoundArgs roundArgs =
      getRevRoundArgs<Op::kRevRecvCopy>(algoCtx, round, opStep);

  const bool isRevFwd = isRevRecvFwd(algoCtx, opStep.step);
  // Forwarded data goes to tmpSendBufRev at offset after initial rev send
  // rounds (analogous to forward direction's getRecvFwdSendRound)
  const int firstRevSendRounds = algoCtx.opRounds[Op::kRevSendCopy].totalRounds;
  const int tmpFwdChunkId =
      isRevFwd ? getTmpChunkId(algoCtx, firstRevSendRounds + round) : -1;

  T* recv_data = getBufAtByteOffset<T>(args.recvbuff, roundArgs.dataOffset);
  const T* tmpRecvBufRev =
      getBufAtByteOffset<T>(args.tmpRecvBufRev, tmpChunkId * args.chunkSize);
  T* tmpSendBufRev =
      getBufAtByteOffset<T>(args.tmpSendBufRev, tmpFwdChunkId * args.chunkSize);

  if (isRevFwd) {
    // Copy to both forward send buffer and output
    ctranKernCopy<T>(
        tmpRecvBufRev, tmpSendBufRev, roundArgs.numel, blockIdx.x, gridDim.x);
    ctranKernCopy<T>(
        tmpRecvBufRev, recv_data, roundArgs.numel, blockIdx.x, gridDim.x);
  } else {
    // Last reverse step: copy to output only
    ctranKernCopy<T>(
        tmpRecvBufRev, recv_data, roundArgs.numel, blockIdx.x, gridDim.x);
  }

  complete(args.revRecvCopySync, blockIdx.x, round);
  opUpdateDone<Op::kRevRecvCopy>(algoCtx);
}

#define KERNEL_ABORT()                                         \
  do {                                                         \
    if (ctran::device::KernelTestHostAbortBlock(kernelFlag)) { \
      return;                                                  \
    }                                                          \
  } while (0);

template <bool EnableBidirAg>
__device__ __forceinline__ void updatePartitionCtxDevice(
    const ctran::allreduce::ring::KernArgs& args,
    AlgoContext& algoCtx) {
  // update local context
  updatePartitionCtx<EnableBidirAg>(algoCtx);

  if (algoCtx.partition > 0) {
    // wait host to reach start of the next partition.
    // It ensures all posted round values for reduce and copy (always starts
    // from 0 in a partition) are for the next partition.
    waitPost(args.partitionSync, blockIdx.x, algoCtx.partition);
  }
}

template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__device__ void algoFn(ctran::allreduce::ring::KernArgs& args) {
  // Setup algorithm context
  AlgoContext algoCtx = {
      .numElements = args.count,
      .rank = statex->rank(),
      .nRanks = statex->nRanks(),
      .chunkSize = args.chunkSize,
      .numChunks = args.numChunks,
      .minShardSize = args.minShardSize,
      .typeSize = sizeof(T)};
  setupAlgoCtxImpl(algoCtx);

  while (algoCtx.partitionOffset < algoCtx.numElements) {
    updatePartitionCtxDevice<EnableBidirAg>(args, algoCtx);
    KERNEL_ABORT();
    CTRAN_DEV_TRACE(
        ALGO_CXT_LOG_FMT_DEVICE, ALGO_CXT_LOG_FIELDS(algoCtx, gridDim.x));

    // Algorithm main loop
    // When EnableBidirAg is false, reverse direction rounds are always 0
    auto notDone = [&]() {
      bool fwdNotDone = algoCtx.opRounds[Op::kSendCopy].done <
              algoCtx.opRounds[Op::kSendCopy].totalRounds ||
          algoCtx.opRounds[Op::kRecvRedCopy].done <
              algoCtx.opRounds[Op::kRecvRedCopy].totalRounds;
      if constexpr (EnableBidirAg) {
        return fwdNotDone ||
            algoCtx.opRounds[Op::kRevSendCopy].done <
            algoCtx.opRounds[Op::kRevSendCopy].totalRounds ||
            algoCtx.opRounds[Op::kRevRecvCopy].done <
            algoCtx.opRounds[Op::kRevRecvCopy].totalRounds;
      }
      return fwdNotDone;
    };
    while (notDone()) {
      _progressSend<T, RedOp>(args, algoCtx);
      _progressRecv<T, RedOp>(args, algoCtx);
      _progressRevSend<T, RedOp, EnableBidirAg>(args, algoCtx);
      _progressRevRecv<T, RedOp, EnableBidirAg>(args, algoCtx);
      KERNEL_ABORT();
    }

    updatePartitionDone(algoCtx);
  }
}

} // namespace ctran::allreduce::ring

// EnableBidirAg template parameter:
// - true: bi-directional AllGather with reverse direction (default)
// - false: standard single-direction AG (lower register usage)
template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__global__ void ncclKernelAllReduceCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::ring::KernArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;

  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  devStateLoadToShm(&flag[bId], devState);

  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  // Run algorithm main body
  ctran::allreduce::ring::algoFn<T, RedOp, EnableBidirAg>(args);

  // This sync threads ensure that every thread in the block has completed using
  // the flag status before resetting it by thread 0 below.
  __syncthreads();

  /* Complete kernel */
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

// Instantiation macro for bi-directional AG kernel
#define DECL_CTRAN_ALLREDUCERING_KERN_BIDIR(T, RedOp)                    \
  template __global__ void ncclKernelAllReduceCtranRing<T, RedOp, true>( \
      int* flag,                                                         \
      CtranAlgoDeviceState* devState,                                    \
      ctran::allreduce::ring::KernArgs args);

// Instantiation macro for simple kernel (no bidir AG, lower register usage)
#define DECL_CTRAN_ALLREDUCERING_KERN_SIMPLE(T, RedOp)                    \
  template __global__ void ncclKernelAllReduceCtranRing<T, RedOp, false>( \
      int* flag,                                                          \
      CtranAlgoDeviceState* devState,                                     \
      ctran::allreduce::ring::KernArgs args);

// Combined macro to instantiate both kernel variants
#define DECL_CTRAN_ALLREDUCERING_KERN(T, RedOp) \
  DECL_CTRAN_ALLREDUCERING_KERN_BIDIR(T, RedOp) \
  DECL_CTRAN_ALLREDUCERING_KERN_SIMPLE(T, RedOp)
