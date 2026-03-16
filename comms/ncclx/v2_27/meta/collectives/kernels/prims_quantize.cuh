// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>

#include "network/unpack/unpack.h"

#include "meta/collectives/kernels/common_kernel_quantize.cuh"

// The primitives class for quantized PAT reduce scatter. Mostly copied from
// prims_simple, with the exception that multiple types were changed from using
// type T (Input Type) to TransportType.
// This class handles quantized collectives that:
// - Take input with higher precision (InputType, e.g., float)
// - Transport data with lower precision (TransportType, e.g., bf16)
// - Accumulate/reduce in higher precision (InputType)
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
  uint64_t* randomSeedPtr; // Optional seed for stochastic rounding

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
  __device__ PrimitivesQuantized(
      int tid,
      int nthreads,
      int const* recvPeers,
      int const* sendPeers,
      void const* inputBuf,
      void* outputBuf,
      uint64_t redOpArg,
      uint8_t group,
      uint64_t* randomSeedPtr = nullptr)
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
      // Transport buffer uses TransportType size for step calculations
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
      // Transport buffer uses TransportType size for step calculations
      peer->connStepSize = conn->stepSize / sizeof(TransportType);
    }
    if (tid == 0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.redOpArgs[0] = redOpArg; // scaler for local input
    }
    patBarrier();
  }

  __device__ __forceinline__ void patReduce(
      struct ncclPatStep* ps,
      struct ncclPatShmem* shmem,
      uint64_t baseRNGOffset) {
    if (ps->flags & PatSkipped) {
      patBarrier();
      patBarrier();
      return;
    } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
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

    // Compute type flags for ALL threads based on shared ps data.
    // This must not be inside role-guarded blocks, because all threads in the
    // group call reduceCopyMixed and need consistent type information.
    //   src0: recv buffer is always TransportType
    //   src1: depends on whether it's new data (userInput=AccumType) or
    //         re-accumulation (send buffer=TransportType, or
    //         userOutput=AccumType)
    //   dst0: send buffer=TransportType, or userOutput=AccumType
    bool src0IsAccumType = false; // recv buffer is always TransportType
    bool src1IsAccumType = false;
    bool dst0IsAccumType = false;
    long long int localAccSize = shmem->localAccSize;

    if (ps->sendDim >= 0) {
      // Send path: dst is TransportType (send buffer)
      dst0IsAccumType = false;
      struct ncclPatPeer* sendPeer = shmem->sendDims + ps->sendDim;
      if (sendPeer->accSize < ps->sendOffset + nelem +
              (sendPeer->step + ps->stepOffset) * sendPeer->connStepSize) {
        src1IsAccumType = true; // New data from userInput (AccumType)
      } else {
        src1IsAccumType =
            false; // Re-accumulation from send buffer (TransportType)
      }
    } else {
      // Output path: dst is AccumType (userOutput), src1 is always AccumType
      // (either new userInput data or re-accumulated userOutput, both are FP32)
      dst0IsAccumType = true;
      src1IsAccumType = true;
    }

    if (recv && (flags & RoleWaitRecv)) {
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

    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;

    // When no recv peer, we shift srcs to skip srcs[0]
    // After shift, the former srcs[1] becomes srcs[0], so update
    // src0IsAccumType
    if (ps->recvDim < 0) {
      srcs++;
      nSrcs--;
      src0IsAccumType = src1IsAccumType; // Former srcs[1] is now srcs[0]
    }

    int workSize = ncclShmem.aborted ? 0 : nelem;

    // Single call with runtime type dispatch - no manual switching needed
    reduceCopyMixed<Unroll, RedOp, InputType, TransportType>(
        tid,
        nthreads,
        ncclShmem.redOpArgs[0],
        nSrcs,
        srcs,
        1,
        ncclShmem.groups[group].dsts,
        workSize,
        src0IsAccumType,
        src1IsAccumType,
        dst0IsAccumType,
        randomSeedPtr ? *randomSeedPtr : 0,
        baseRNGOffset);

    // Store conn step here inside the two barriers to make sure next reload
    // will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
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
