// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>

#include "network/unpack/unpack.h"

#include "meta/collectives/kernels/reduce_copy_sr_v2.cuh"

// =========================================================================
// Quantized PAT Reduce Scatter Primitives
//
// This file implements PrimitivesQuantized, a variant of the Primitives class
// (from prims_simple.h) specialized for quantized PAT reduce scatter.
//
// The class handles quantized collectives that:
// - Take input with higher precision (InputType, e.g., float)
// - Transport data with lower precision (TransportType, e.g., bf16)
// - Accumulate/reduce in higher precision (InputType)
//
// Compared to the original Primitives (prims_simple.h), changes fall into
// two categories:
//
// 1. Typing changes: T → InputType for user I/O buffers,
//    T → TransportType for transport (recv/send) buffers and step sizes.
//    These are mechanical and not individually commented.
//
// 2. Quantized-specific additions: marked with [QUANTIZED] comments.
//    These include: randomSeedPtr, baseRNGOffset parameter,
//    type configuration helpers, and reduceCopySR dispatch.
// =========================================================================

// =========================================================================
// [QUANTIZED] Type configuration for mixed-precision reduce-copy dispatch.
//
// Determines at runtime what types the sources and destination have for a
// given PAT reduce step. This modularizes the scattered type-determination
// logic that was previously inline in patReduce.
//
// The decision depends on two things:
//   1. Is the destination the user output buffer or a send buffer?
//      sendDim < 0  →  output path  →  dst = InputType
//      sendDim >= 0 →  send path    →  dst = TransportType
//
//   2. Is the local source (srcs[1]) fresh data or re-accumulated data?
//      On the output path, it's always InputType (userInput or userOutput).
//      On the send path, it depends on accSize:
//        accSize < threshold  →  new data from userInput    →  InputType
//        accSize >= threshold →  re-accumulation from dst   →  TransportType
// =========================================================================
struct QuantizedTypeConfig {
  bool dstIsOutput; // true: dst is InputType (user output)
                    // false: dst is TransportType (send buffer)
  bool src1IsAccumType; // true: local src is InputType (fresh data)
                        // false: local src is TransportType (re-accumulation)
};

template <typename TransportType>
__device__ __forceinline__ QuantizedTypeConfig determineQuantizedTypeConfig(
    const struct ncclPatStep* ps,
    const struct ncclPatShmem* shmem,
    int nelem) {
  QuantizedTypeConfig config;
  config.dstIsOutput = (ps->sendDim < 0);
  config.src1IsAccumType =
      true; // default: InputType (always true on output path)

  if (ps->sendDim >= 0) {
    const struct ncclPatPeer* sendPeer = shmem->sendDims + ps->sendDim;
    // This is the same condition used in patReduce to choose between
    // srcs[1] = userInput (new data) vs srcs[1] = dsts[0] (re-accumulation).
    bool isNewData = sendPeer->accSize < ps->sendOffset + nelem +
            (sendPeer->step + ps->stepOffset) * sendPeer->connStepSize;
    config.src1IsAccumType = isNewData;
  }
  return config;
}

// =========================================================================
// [QUANTIZED] Unified dispatch for mixed-precision reduce-copy with
// stochastic rounding. Replaces the original single reduceCopy call.
//
// Type combinations dispatched (AccType is always InputType):
//
// Case 1: recv + output   → SR(dst=Input,  recvSrc=Transport, localSrc=Input)
// Case 2: recv + send new → SR(dst=Transport, recvSrc=Transport,
// localSrc=Input) Case 3: recv + send acc → SR(dst=Transport,
// recvSrc=Transport, localSrc=Transport) Case 4: no recv + output   →
// SR(dst=Input,  localSrc=Input) Case 5: no recv + send new → SR(dst=Transport,
// localSrc=Input) Case 6: no recv + send acc → SR(dst=Transport,
// localSrc=Transport)
// =========================================================================
template <int Unroll, typename InputType, typename TransportType>
__device__ __forceinline__ void quantizedPatReduceCopy(
    int tid,
    int nthreads,
    void* dstPtr,
    void* recvSrcPtr, // srcs[0] — only dereferenced when hasRecvData
    void* localSrcPtr, // srcs[1] — always dereferenced
    int workSize,
    uint64_t seed,
    uint64_t baseRNGOffset,
    bool hasRecvData,
    const QuantizedTypeConfig& typeConfig) {
  using namespace meta::comms::ncclx::kernels::simplecopy_v2;

  if (hasRecvData) {
    auto* recvSrc = (TransportType*)recvSrcPtr;
    if (typeConfig.dstIsOutput) {
      // Output path: Transport + Input → Input (upcast, SR is identity)
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (InputType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          recvSrc,
          (InputType*)localSrcPtr);
    } else if (typeConfig.src1IsAccumType) {
      // Send path, new data: Transport + Input → Transport (SR downcast)
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (TransportType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          recvSrc,
          (InputType*)localSrcPtr);
    } else {
      // Send path, re-accumulation: Transport + Transport → Transport
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (TransportType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          recvSrc,
          (TransportType*)localSrcPtr);
    }
  } else {
    if (typeConfig.dstIsOutput) {
      // Output path, no recv: Input → Input (plain copy)
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (InputType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          (InputType*)localSrcPtr);
    } else if (typeConfig.src1IsAccumType) {
      // Send path, new data, no recv: Input → Transport (SR downcast)
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (TransportType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          (InputType*)localSrcPtr);
    } else {
      // Send path, re-accumulation, no recv: Transport → Transport
      reduceCopySR<Unroll, InputType>(
          tid,
          nthreads,
          (TransportType*)dstPtr,
          (ssize_t)workSize,
          seed,
          baseRNGOffset,
          (TransportType*)localSrcPtr);
    }
  }
}

// =========================================================================
// PrimitivesQuantized
//
// A simplified variant of Primitives (prims_simple.h) that only supports
// PAT mode. The constructor mirrors the PAT-mode branch of the original
// Primitives constructor. patReduce mirrors the original patReduce with
// quantized-specific changes annotated.
//
// Template parameters:
//   InputType     — user I/O precision (e.g., float)
//   TransportType — transport/wire precision (e.g., __nv_bfloat16)
//   RedOp         — reduction operation
//   StepPerSlice  — steps per slice (typically 1 for PAT)
//   Unroll        — unroll factor for reduce-copy kernels
// =========================================================================
template <
    typename InputType,
    typename TransportType,
    typename RedOp,
    int StepPerSlice,
    int Unroll>
class PrimitivesQuantized {
  const int tid;
  const int nthreads;
  int flags;
  int group;
  uint64_t step;
  uint64_t* randomSeedPtr; // [QUANTIZED] Seed for stochastic rounding

  static constexpr int Aborted = 0x40;
  static constexpr int RoleInput = 0x01, RoleOutput = 0x02, RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08, RolePostSend = 0x10,
                       RolePostRecv = 0x20;

  // PAT uses a single barrier across all groups
  __device__ void patBarrier() {
    barrier_sync(15, NCCL_PAT_NWORKERS);
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
    return ld_volatile_global(ptr);
  }

 public:
  // Constructor: mirrors the PAT-mode branch of the original Primitives
  // constructor (prims_simple.h, mode == primsModePatRs).
  // [Quantized] Typing: sizeof(T) → sizeof(TransportType) for connStepSize.
  // [QUANTIZED] Added randomSeedPtr parameter.
  __device__ PrimitivesQuantized(
      int tid,
      int nthreads,
      int const* recvPeers,
      int const* sendPeers,
      void const* inputBuf,
      void* outputBuf,
      uint64_t redOpArg,
      uint8_t group,
      uint64_t* randomSeedPtr = nullptr) // [QUANTIZED]
      : tid(tid),
        nthreads(nthreads),
        group(group),
        randomSeedPtr(randomSeedPtr) {
    flags = 0;
    const int roles[5] = {
        RoleWaitRecv,
        RolePostRecv,
        RoleWaitSend,
        RolePostSend,
        RoleInput | RoleOutput};
    if (tid < 5)
      flags |= roles[tid];

    int nranks = ncclShmem.comm.nRanks;
    if (tid < 32 && ((1UL << tid) < nranks)) {
      int rank = ncclShmem.comm.rank;
      uint32_t delta = 1 << tid;
      // Load recv peer
      int recvPeer = (rank - delta + nranks) % nranks;
      struct ncclPatPeer* peer = ((struct ncclPatPeer*)recvPeers) + tid;
      struct ncclConnInfo* conn = peer->conn =
          ncclShmem.channel.peers[recvPeer]->recv;
      peer->step = conn->step;
      peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
      peer->stepCache = loadStepValue(peer->tailPtr = conn->tail);
      peer->headPtr = conn->head;
      peer->accSize = 0;
      // [Quantized] Typing: sizeof(T) → sizeof(TransportType)
      peer->connStepSize = conn->stepSize / sizeof(TransportType);
      // Load send peer
      int sendPeer = (rank + delta) % nranks;
      peer = ((struct ncclPatPeer*)sendPeers) + tid;
      conn = peer->conn = ncclShmem.channel.peers[sendPeer]->send;
      peer->step = conn->step;
      peer->connFifo = conn->connFifo;
      peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
      peer->stepCache = loadStepValue(peer->headPtr = conn->head);
      peer->tailPtr = conn->tail;
      peer->accSize = 0;
      // [Quantized] Typing: sizeof(T) → sizeof(TransportType)
      peer->connStepSize = conn->stepSize / sizeof(TransportType);
    }
    if (tid == 0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.redOpArgs[0] = redOpArg; // scaler for local input
    }
    patBarrier();
  }

  // patReduce: mirrors the original patReduce from prims_simple.h.
  // Changes from original:
  //   - Typing: T → InputType for user buffers, T → TransportType for
  //     transport buffers and sizeof calculations.
  //   - [QUANTIZED] baseRNGOffset parameter added.
  //   - [QUANTIZED] reduceCopy replaced with quantizedPatReduceCopy.
  __device__ __forceinline__ void patReduce(
      struct ncclPatStep* ps,
      struct ncclPatShmem* shmem,
      uint64_t baseRNGOffset) { // [QUANTIZED] additional parameter
    if (ps->flags & PatSkipped) {
      patBarrier();
      patBarrier();
      return;
    } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    // [Quantized] Typing: T → InputType
    InputType* userInput = (InputType*)ncclShmem.groups[group].userInput;
    InputType* userOutput = (InputType*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv | RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend | RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims + ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims + ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      // [Quantized] Typing: (T*) → (TransportType*) for transport buffer
      ncclShmem.groups[group].srcs[0] = ((TransportType*)peer->buff) +
          (step % NCCL_STEPS) * peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins))
          break;
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS <
             step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins))
          break;
      }
      // [Quantized] Typing: (T*) → (TransportType*) for transport buffer
      ncclShmem.groups[group].dsts[0] = ((TransportType*)peer->buff) +
          ((step + ps->stepOffset) % NCCL_STEPS) * peer->connStepSize +
          ps->sendOffset;
      if (peer->accSize < ps->sendOffset + nelem +
              (step + ps->stepOffset) * peer->connStepSize) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 &&
        (flags & RoleOutput)) { // Destination is our own local buffer
      ncclShmem.groups[group].dsts[0] = userOutput + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    patBarrier();

    // [QUANTIZED] Determine src/dst type configuration for this step + Replace
    // original reduceCopy with type-aware dispatch.
    // Original code was:
    //   int nSrcs = 2;
    //   void** srcs = ncclShmem.groups[group].srcs;
    //   if (ps->recvDim < 0) { srcs++; nSrcs--; }
    //   reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, 0>
    //     (tid, nthreads, redOpArgs, postOp, nSrcs, srcs, 1, dsts, workSize);
    auto typeConfig =
        determineQuantizedTypeConfig<TransportType>(ps, shmem, nelem);
    int workSize = ncclShmem.aborted ? 0 : nelem;
    uint64_t seed = randomSeedPtr ? *randomSeedPtr : 0;
    quantizedPatReduceCopy<Unroll, InputType, TransportType>(
        tid,
        nthreads,
        ncclShmem.groups[group].dsts[0], // dst
        ncclShmem.groups[group]
            .srcs[0], // recv src (only used when hasRecvData)
        ncclShmem.groups[group].srcs[1], // local src (always used)
        workSize,
        seed,
        baseRNGOffset,
        /*hasRecvData=*/ps->recvDim >= 0,
        typeConfig);

    // Store conn step here inside the two barriers to make sure next reload
    // will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        // [Quantized] Typing: sizeof(T) → sizeof(TransportType)
        peer->connFifo[step % NCCL_STEPS].size =
            (ps->sendOffset + nelem) * sizeof(TransportType);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(
          &peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->sendDim < 0 && (flags & RoleOutput))
      atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & RoleWaitSend))
      atomicMax(
          &peer->accSize,
          ps->sendOffset + nelem +
              (step + ps->stepOffset) * peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo)
        fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }
};
