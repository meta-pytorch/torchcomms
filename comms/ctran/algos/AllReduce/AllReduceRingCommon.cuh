// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef COMMS_CTRAN_ALLREDUCE_RING_COMMON_CUH
#define COMMS_CTRAN_ALLREDUCE_RING_COMMON_CUH

#include <chrono>

#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/utils/DevAttribute.h"
#include "comms/utils/commSpecs.h"

namespace ctran::allreduce::ring {

struct KernArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  commRedOp_t redOp;
  size_t count;

  size_t chunkSize;
  size_t numChunks;
  size_t minShardSize;

  ctran::algos::GpeKernelSync* sendCopySync;
  ctran::algos::GpeKernelSync* recvRedCopySync;
  ctran::algos::GpeKernelSync* partitionSync;

  void* tmpSendBuf;
  void* tmpRecvBuf;

  // Reverse direction (bi-directional AllGather)
  ctran::algos::GpeKernelSync* revSendCopySync;
  ctran::algos::GpeKernelSync* revRecvCopySync;
  void* tmpSendBufRev;
  void* tmpRecvBufRev;
};

// used by e.g. NanChecker and Trace
enum Phase {
  kReduceScatter,
  kAllGather,
};

#ifdef __CUDA_ARCH__
// Omit host-side ops to save register on device side
enum Op {
  kSendCopy,
  kRecvRedCopy,
  kRevSendCopy,
  kRevRecvCopy,
  kMaxNumOp,
};
#else
enum Op {
  kSendCopy,
  kSendTrans,
  kRecvTrans,
  kRecvFlush,
  kRecvRedCopy,
  kRevSendCopy,
  kRevSendTrans,
  kRevRecvTrans,
  kRevRecvFlush,
  kRevRecvCopy,
  kMaxNumOp,
};
#endif

#define MAX_NUM_BLOCK_SIZE (512)

struct OpStep {
  int step = 0;
  int startRound = 0;
};

struct OpRound {
  int totalRounds = 0;
  // Track current ready to post round if the op depends on a previous op
  // (e.g., sendCopy depends on kRecvRedCopy) and the dependent op is done
  int ready = 0;
  int post = 0;
  int done = 0;

  // Track the step and the starting round of the step separately for post and
  // done, since they may be in different step.
  OpStep postStep;
  OpStep doneStep;
};

// Common context argument structures shared by both host and device side, and
// used to coordinate within the algorithm.
struct AlgoContext {
  // Const arguments set per allreduce
  const size_t numElements;
  const int rank;
  const int nRanks;
  const size_t chunkSize;
  const size_t numChunks;
  const size_t minShardSize;
  const size_t typeSize;

  // Algorithm internal parameters computed based on above const arguments.
  // See setupAlgoCtxImpl.

  // Top-level partition:
  // Split allreduce data to multiple partition if cannot fit into chunkSize *
  // numChunks. A partition is transferred via multi-shard ring, and multiple
  // partitions are transferred one by one.
  size_t partitionNumel;
  size_t partitionOffset;
  int partition;

  // Within partition (a ring):
  // Split a partition as nRanks number of shards, each rank handles one
  // shard at each step
  int numShards; // number of shards of each data partition. It is always
                 // nRanks.
  int numSteps;
  size_t shardNumel;
  int rightRank;
  int leftRank;

  // Define the starting shardId in data buffers. Step-i's shardId is
  // shifted to right by i (see getStepShardIdx).
  int sendDataShardIdx;
  int recvDataShardIdx;

  // Step/rounds within a ring within a partition:
  // Step tracks per-shard transfer to finish two-phase ring, it is always
  // nRanks * 2 - 2. Round tracks per-chunk transfer, a shard can contain
  // multiple chunks depending on the data size.
  OpRound opRounds[kMaxNumOp];
  int firstStepNumRounds;
  int recvFwdStartRound;

  int totalSendRounds;
  int totalRecvRounds;

  // Bi-directional AllGather: reverse direction distributes shards left
  int numFwdAgSteps; // ceil((nRanks-1)/2) forward AG steps
  int numRevAgSteps; // floor((nRanks-1)/2) reverse AG steps
  int revSendDataShardIdx; // rank (same base as forward; shift handles +1)
  int revRecvDataShardIdx; // rightRank (receives from right in reverse)

  // Track starting round per partition for tracing purpose
  int partitionStartSendRounds;
  int partitionStartRecvRounds;
};

#define ALGO_CXT_LOG_FMT_DEVICE \
  "Partition %d start: partitionOffset %ld partitionNumel %ld numShards %d shardNumel %ld numSteps %d totalSendRounds %d totalRecvRounds %d rank %d rightRank %d leftRank %d nRanks %d chunkSize %ld numChunks %ld numThreadBlocks %d\n"
#define ALGO_CXT_LOG_FMT_HOST \
  "Partition {} start: partitionOffset {} partitionNumel {} numShards {} shardNumel {} numSteps {} totalSendRounds {} totalRecvRounds {} rank {} rightRank {} leftRank {} nRanks {} chunkSize {} numChunks {} numThreadBlocks {}\n"
#define ALGO_CXT_LOG_FIELDS(algoCtx, numThreadBlocks)                         \
  algoCtx.partition, algoCtx.partitionOffset, algoCtx.partitionNumel,         \
      algoCtx.numShards, algoCtx.shardNumel, algoCtx.numSteps,                \
      algoCtx.totalSendRounds, algoCtx.totalRecvRounds, algoCtx.rank,         \
      algoCtx.rightRank, algoCtx.leftRank, algoCtx.nRanks, algoCtx.chunkSize, \
      algoCtx.numChunks, numThreadBlocks

DEVICE_ATTRIBUTE size_t
getShardNumel(const int shardId, const AlgoContext& algoCtx) {
  // Last shard handles remaining elements that may be larger than
  // shardNumel. E.g., shard 33 elements with 4 nRanks as [8,8,8,9].
  if (shardId == algoCtx.numShards - 1) {
    return algoCtx.partitionNumel -
        algoCtx.shardNumel * (algoCtx.numShards - 1);
  }
  return algoCtx.shardNumel;
}

template <Op op>
DEVICE_ATTRIBUTE int getStepShardIdx(
    const AlgoContext& algoCtx,
    const int step) {
  int shift = algoCtx.numShards - step % algoCtx.numShards;
  // Shifted shard to left by one per ringStep.
  if (op == Op::kSendCopy
#ifndef __CUDA_ARCH__
      || op == Op::kSendTrans
#endif
  ) {
    return (algoCtx.sendDataShardIdx + shift) % algoCtx.numShards;
  } else {
    return (algoCtx.recvDataShardIdx + shift) % algoCtx.numShards;
  }
}

// Reverse direction shard index for bi-directional AllGather.
// Data flows counter-clockwise (left): rank sends to leftRank.
// revStep is 0-based within the reverse AG phase.
//
// Forward direction uses shift = N - step (decreasing shard index per step).
// Reverse direction uses shift = revStep + 1 (increasing shard index per step)
// because data rotates in the opposite direction. The +1 accounts for the fact
// that after RS, rank r owns shard (r+1)%N (the last RS recv step reduces
// shard (leftRank + 2) % N = (rank + 1) % N).
template <Op op>
DEVICE_ATTRIBUTE int getRevStepShardIdx(
    const AlgoContext& algoCtx,
    const int revStep) {
  int shift = (revStep + 1) % algoCtx.numShards;
  if (op == Op::kRevSendCopy
#ifndef __CUDA_ARCH__
      || op == Op::kRevSendTrans
#endif
  ) {
    return (algoCtx.revSendDataShardIdx + shift) % algoCtx.numShards;
  } else {
    return (algoCtx.revRecvDataShardIdx + shift) % algoCtx.numShards;
  }
}

DEVICE_ATTRIBUTE int countChunksPerShard(
    const AlgoContext& algoCtx,
    const int shardId) {
  // Each shard is transferred per chunk with size of chunkNumel
  size_t chunkNumel = algoCtx.chunkSize / algoCtx.typeSize;
  return (getShardNumel(shardId, algoCtx) + chunkNumel - 1) / chunkNumel;
}

// Template parameter EnableBidirAg controls bi-directional AllGather:
// - true: Use bidir AG with reverse direction (reduced steps)
// - false: Use standard single-direction AG (original 2*(N-1) steps)
template <bool EnableBidirAg = true>
DEVICE_ATTRIBUTE void updatePartitionCtx(AlgoContext& algoCtx) {
  size_t numelPerChunk = algoCtx.chunkSize / algoCtx.typeSize;
  size_t minPartitionNumel =
      algoCtx.minShardSize * algoCtx.nRanks / algoCtx.typeSize;
  size_t totalTmpNumel = numelPerChunk * algoCtx.numChunks * algoCtx.nRanks;

  // Move to next partition
  size_t remainNumel = algoCtx.numElements - algoCtx.partitionOffset;
  if (remainNumel < totalTmpNumel) {
    algoCtx.partitionNumel = remainNumel;
  } else if (
      remainNumel > totalTmpNumel &&
      remainNumel < totalTmpNumel + minPartitionNumel) {
    // ensure last partition can still be handled as multi-shard ring
    algoCtx.partitionNumel = remainNumel - minPartitionNumel;
  } else {
    algoCtx.partitionNumel = totalTmpNumel;
  }

  algoCtx.numShards = algoCtx.nRanks;
  algoCtx.shardNumel = algoCtx.partitionNumel / algoCtx.numShards;

  // Bi-directional AllGather step counts
  int rsSteps = algoCtx.nRanks - 1;
  if constexpr (EnableBidirAg) {
    algoCtx.numFwdAgSteps = (rsSteps + 1) / 2; // ceil((N-1)/2)
    algoCtx.numRevAgSteps = rsSteps / 2; // floor((N-1)/2)
  } else {
    // Standard single-direction AG: all AG steps go forward
    algoCtx.numFwdAgSteps = rsSteps;
    algoCtx.numRevAgSteps = 0;
  }

  // Forward direction: RS steps + forward AG steps
  algoCtx.numSteps = rsSteps + algoCtx.numFwdAgSteps;

  // Initialize all op round counters
  for (int i = 0; i < Op::kMaxNumOp; i++) {
    algoCtx.opRounds[i] = {};
  }

  size_t totalSendRounds = 0;
  size_t totalRecvRounds = 0;

  algoCtx.sendDataShardIdx = algoCtx.rank;
  algoCtx.recvDataShardIdx = algoCtx.leftRank;
  // Reverse direction uses same base convention as forward (rank/rightRank).
  // The +1 offset to reach the actual reduced shard is handled by the shift
  // formula in getRevStepShardIdx.
  algoCtx.revSendDataShardIdx = algoCtx.rank;
  algoCtx.revRecvDataShardIdx = algoCtx.rightRank;

  // Count forward direction rounds (RS + forward AG)
  for (int step = 0; step < algoCtx.numSteps; step++) {
    totalSendRounds += countChunksPerShard(
        algoCtx, getStepShardIdx<Op::kSendCopy>(algoCtx, step));
    totalRecvRounds += countChunksPerShard(
        algoCtx, getStepShardIdx<Op::kRecvRedCopy>(algoCtx, step));
  }

  // Set first step
  algoCtx.opRounds[Op::kSendCopy].ready =
      countChunksPerShard(algoCtx, algoCtx.sendDataShardIdx);
  algoCtx.opRounds[Op::kSendCopy].totalRounds =
      algoCtx.opRounds[Op::kSendCopy].ready;
  algoCtx.firstStepNumRounds = algoCtx.opRounds[Op::kSendCopy].ready;
  algoCtx.recvFwdStartRound = 0;

  algoCtx.opRounds[Op::kRecvRedCopy].totalRounds = totalRecvRounds;
#ifndef __CUDA_ARCH__
  algoCtx.opRounds[Op::kRecvTrans].ready = -1;
  algoCtx.opRounds[Op::kSendTrans].totalRounds = totalSendRounds;
  algoCtx.opRounds[Op::kRecvTrans].totalRounds = totalRecvRounds;
  algoCtx.opRounds[Op::kRecvFlush].totalRounds = totalRecvRounds;
#endif

  algoCtx.totalSendRounds = totalSendRounds;
  algoCtx.totalRecvRounds = totalRecvRounds;

  // Reverse direction rounds for bi-directional AllGather
  // Skip entirely when bidir AG is disabled
  if constexpr (EnableBidirAg) {
    size_t totalRevSendRounds = 0;
    size_t totalRevRecvRounds = 0;
    for (int revStep = 0; revStep < algoCtx.numRevAgSteps; revStep++) {
      totalRevSendRounds += countChunksPerShard(
          algoCtx, getRevStepShardIdx<Op::kRevSendCopy>(algoCtx, revStep));
      totalRevRecvRounds += countChunksPerShard(
          algoCtx, getRevStepShardIdx<Op::kRevRecvCopy>(algoCtx, revStep));
    }

    // RevSendCopy: first step copies rank's reduced shard to tmpSendBufRev.
    // On host side, ready starts at 0 and is released when RS finishes (the
    // reduced shard is written to recvbuff). On device side, ready can be set
    // immediately since the kernel only acts when the host posts via
    // GpeKernelSync.
    if (algoCtx.numRevAgSteps > 0) {
      int firstRevShardChunks = countChunksPerShard(
          algoCtx, getRevStepShardIdx<Op::kRevSendCopy>(algoCtx, 0));
#ifdef __CUDA_ARCH__
      algoCtx.opRounds[Op::kRevSendCopy].ready = firstRevShardChunks;
#else
      algoCtx.opRounds[Op::kRevSendCopy].ready = 0;
#endif
      algoCtx.opRounds[Op::kRevSendCopy].totalRounds = firstRevShardChunks;
    }
    algoCtx.opRounds[Op::kRevRecvCopy].totalRounds = totalRevRecvRounds;
#ifndef __CUDA_ARCH__
    if (algoCtx.numRevAgSteps > 0) {
      algoCtx.opRounds[Op::kRevRecvTrans].ready = -1;
    }
    algoCtx.opRounds[Op::kRevSendTrans].totalRounds = totalRevSendRounds;
    algoCtx.opRounds[Op::kRevRecvTrans].totalRounds = totalRevRecvRounds;
    algoCtx.opRounds[Op::kRevRecvFlush].totalRounds = totalRevRecvRounds;
#endif
  }
}

DEVICE_ATTRIBUTE void updatePartitionDone(AlgoContext& algoCtx) {
  algoCtx.partitionOffset += algoCtx.partitionNumel;
  algoCtx.partition++;

#ifndef __CUDA_ARCH__
  algoCtx.partitionStartSendRounds +=
      algoCtx.opRounds[Op::kSendTrans].totalRounds;
  algoCtx.partitionStartRecvRounds +=
      algoCtx.opRounds[Op::kRecvTrans].totalRounds;
#endif
}

DEVICE_ATTRIBUTE void setupAlgoCtxImpl(AlgoContext& algoCtx) {
  algoCtx.partitionOffset = 0;
  algoCtx.rightRank = (algoCtx.rank + 1) % algoCtx.nRanks;
  algoCtx.leftRank = (algoCtx.rank - 1 + algoCtx.nRanks) % algoCtx.nRanks;

#ifndef __CUDA_ARCH__
  algoCtx.partitionStartSendRounds = 0;
  algoCtx.partitionStartRecvRounds = 0;
#endif
}

DEVICE_ATTRIBUTE Phase getPhase(const AlgoContext& algoCtx, int step) {
  int rsSteps = algoCtx.nRanks - 1;
  if (step < rsSteps) {
    return Phase::kReduceScatter;
  } else {
    return Phase::kAllGather;
  }
}

DEVICE_ATTRIBUTE int getTmpChunkId(
    const AlgoContext& algoCtx,
    const int round) {
  // Use chunk of sendBuf and recvBuf as round-robin; shift by 1 for each round.
  return round % algoCtx.numChunks;
}

struct RoundArgs {
  size_t numel;
  int shardId;
  int shardDataChunkId;
  // offset in bytes
  size_t dataOffset;
  // offset in elements
  size_t dataOffsetElem;
};

template <Op op>
DEVICE_ATTRIBUTE RoundArgs getRoundArgs(
    const AlgoContext& algoCtx,
    const int round,
    const OpStep& opStep) {
  struct RoundArgs args = {0, 0, 0, 0, 0};

  // Get current shard and its numel. Last shard may handle more than
  // algoCtx.shardNumel
  int shardId = getStepShardIdx<op>(algoCtx, opStep.step);
  size_t shardNumel = getShardNumel(shardId, algoCtx);
  size_t chunkNumel = algoCtx.chunkSize / algoCtx.typeSize;
  // Last round in step handles any remaining numel
  int roundInStep = round - opStep.startRound;
  if ((roundInStep + 1) * chunkNumel <= shardNumel) {
    args.numel = chunkNumel;
  } else {
    args.numel = shardNumel - chunkNumel * roundInStep;
  }

  args.shardId = shardId;
  args.shardDataChunkId = roundInStep;
  args.dataOffsetElem = algoCtx.partitionOffset + shardId * algoCtx.shardNumel +
      roundInStep * chunkNumel;
  args.dataOffset = args.dataOffsetElem * algoCtx.typeSize;

  return args;
}

DEVICE_ATTRIBUTE bool isRecvFwd(const AlgoContext& algoCtx, int recvStep) {
  return recvStep < algoCtx.numSteps - 1;
}

DEVICE_ATTRIBUTE int getRecvFwdSendRound(
    const AlgoContext& algoCtx,
    int recvRound) {
  return algoCtx.firstStepNumRounds + recvRound;
}

// Reverse direction: forward if not the last reverse AG step
DEVICE_ATTRIBUTE bool isRevRecvFwd(
    const AlgoContext& algoCtx,
    int revRecvStep) {
  return revRecvStep < algoCtx.numRevAgSteps - 1;
}

// Compute RoundArgs for a reverse direction op at given revStep
template <Op op>
DEVICE_ATTRIBUTE RoundArgs getRevRoundArgs(
    const AlgoContext& algoCtx,
    const int round,
    const OpStep& opStep) {
  struct RoundArgs args = {0, 0, 0, 0, 0};
  int shardId = getRevStepShardIdx<op>(algoCtx, opStep.step);
  size_t shardNumel = getShardNumel(shardId, algoCtx);
  size_t chunkNumel = algoCtx.chunkSize / algoCtx.typeSize;
  int roundInStep = round - opStep.startRound;
  if ((roundInStep + 1) * chunkNumel <= shardNumel) {
    args.numel = chunkNumel;
  } else {
    args.numel = shardNumel - chunkNumel * roundInStep;
  }
  args.shardId = shardId;
  args.shardDataChunkId = roundInStep;
  args.dataOffsetElem = algoCtx.partitionOffset + shardId * algoCtx.shardNumel +
      roundInStep * chunkNumel;
  args.dataOffset = args.dataOffsetElem * algoCtx.typeSize;
  return args;
}

template <Op op>
DEVICE_ATTRIBUTE void
opUpdateStep(AlgoContext& algoCtx, int round, OpStep& opStep) {
  // Use reverse shard index for reverse ops
  int shardId;
  if constexpr (
      op == Op::kRevSendCopy || op == Op::kRevRecvCopy
#ifndef __CUDA_ARCH__
      || op == Op::kRevSendTrans || op == Op::kRevRecvTrans ||
      op == Op::kRevRecvFlush
#endif
  ) {
    shardId = getRevStepShardIdx<op>(algoCtx, opStep.step);
  } else {
    shardId = getStepShardIdx<op>(algoCtx, opStep.step);
  }
  int numChunks = countChunksPerShard(algoCtx, shardId);
  // Update step if all rounds in current step has finished
  if (round - opStep.startRound == numChunks) {
    opStep.startRound += numChunks;
    opStep.step++;
  }
}

template <Op op>
inline bool opReadyToPost(const AlgoContext& algoCtx) {
  // On hold recvRedCopy with combined fwdCopy till all chunks in first step has
  // posted to network. It avoids out-of-order chunk.
  if (op == Op::kRecvRedCopy &&
      algoCtx.opRounds[Op::kSendTrans].postStep.step == 0) {
    return false;
  }
  // Similarly, hold revRecvCopy until first reverse send step has posted
  if (op == Op::kRevRecvCopy &&
      algoCtx.opRounds[Op::kRevSendTrans].postStep.step == 0) {
    return false;
  }
  return (algoCtx.opRounds[op].post < algoCtx.opRounds[op].ready ||
          algoCtx.opRounds[op].ready == -1) &&
      algoCtx.opRounds[op].post < algoCtx.opRounds[op].totalRounds;
}

template <Op op>
inline bool opHasPosted(const AlgoContext& algoCtx) {
  return algoCtx.opRounds[op].done < algoCtx.opRounds[op].post;
}

template <Op op>
DEVICE_ATTRIBUTE void opUpdatePost(AlgoContext& algoCtx) {
  algoCtx.opRounds[op].post++;
  opUpdateStep<op>(
      algoCtx, algoCtx.opRounds[op].post, algoCtx.opRounds[op].postStep);
}

template <Op op>
DEVICE_ATTRIBUTE void opUpdateDone(AlgoContext& algoCtx) {
  algoCtx.opRounds[op].done++;
  opUpdateStep<op>(
      algoCtx, algoCtx.opRounds[op].done, algoCtx.opRounds[op].doneStep);

// Host side only updates
#ifndef __CUDA_ARCH__
  // Update the ready to post counter in depended op
  switch (op) {
    case Op::kSendCopy:
      algoCtx.opRounds[Op::kSendTrans].ready++;
      break;
    case Op::kRecvTrans:
      algoCtx.opRounds[Op::kRecvFlush].ready++;
      break;
    case Op::kRecvFlush:
      algoCtx.opRounds[Op::kRecvRedCopy].ready++;
      break;
    case Op::kRecvRedCopy:
      algoCtx.opRounds[Op::kSendTrans].ready++;
      // Release reverse send when RS completes (rank's shard is fully
      // reduced in recvbuff). doneStep.step advances to nRanks-1 when all
      // rounds of the last RS step finish.
      if (algoCtx.numRevAgSteps > 0 &&
          algoCtx.opRounds[Op::kRecvRedCopy].doneStep.step >=
              algoCtx.nRanks - 1 &&
          algoCtx.opRounds[Op::kRevSendCopy].ready == 0) {
        algoCtx.opRounds[Op::kRevSendCopy].ready =
            algoCtx.opRounds[Op::kRevSendCopy].totalRounds;
      }
      break;
    // Reverse direction dependencies
    case Op::kRevSendCopy:
      algoCtx.opRounds[Op::kRevSendTrans].ready++;
      break;
    case Op::kRevRecvTrans:
      algoCtx.opRounds[Op::kRevRecvFlush].ready++;
      break;
    case Op::kRevRecvFlush:
      algoCtx.opRounds[Op::kRevRecvCopy].ready++;
      break;
    case Op::kRevRecvCopy:
      algoCtx.opRounds[Op::kRevSendTrans].ready++;
      break;
    default:
      break;
  }
#endif
}

} // namespace ctran::allreduce::ring

// EnableBidirAg template parameter:
// - true: bi-directional AllGather with reverse direction
// - false: standard single-direction AG (lower register usage)
template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__global__ void ncclKernelAllReduceCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::ring::KernArgs args);

#endif // COMMS_CTRAN_ALLREDUCE_RING_COMMON_CUH
