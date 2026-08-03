// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "meta/collectives/kernels/prims_quantize.cuh"

// NCCLX float specialization of the reduce-scatter PAT device kernel, hoisted
// out of the forked upstream device/reduce_scatter.h to keep the fork seam
// minimal. Like the other meta/collectives/kernels fragments, this header is
// meant to be included from within reduce_scatter.h after the core device
// kernel machinery (RunWorkColl, Primitives, ProtoSimple, the PAT shmem/algo
// helpers) is already in scope.
//
// When quantizeRandomSeedPtr is set, uses PrimitivesQuantized for
// mixed-precision transport (FP32 I/O, BF16 transport with stochastic
// rounding). When nullptr, falls through to regular Primitives (same as the
// generic template).
template <typename RedOp>
struct RunWorkColl<
    ncclFuncReduceScatter,
    float,
    RedOp,
    NCCL_ALGO_PAT,
    NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void
  run(int tid, int nthreads, struct ncclDevWorkColl* work) {
#if __CUDA_ARCH__ >= 600
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(
        work,
        ncclShmem.channelId,
        Proto::Id,
        sizeof(float),
        &count,
        &channelOffset,
        &channelCount,
        &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i = tid; i < NCCL_SHMEM_PAT_STEPS; i += nthreads)
      shmem->patSteps[i].flags = 0;
    if (tid == 0)
      shmem->localAccSize = 0;
    if (tid == nworkers)
      shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatRSAlgorithm<float> patAlgo(
          chunkCount * sizeof(float),
          NCCL_STEPS,
          NCCL_PAT_NWORKERS / WARP_SIZE,
          channelOffset,
          channelOffset + channelCount,
          count,
          chunkCount,
          rank,
          nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep* ps =
            shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) != 0)
          pollCount++; // Wait for workers to be done with step
                       // 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2)
          break;
      }
    } else if (tid < nworkers) { // Worker threads
      float* inputBuf = (float*)work->sendbuff;
      float* outputBuf = (float*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0)
        parallelFactor = *pfPtr;

      int groupSize = nworkers / (WARP_SIZE * parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group * groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead

      if (work->quantizeRandomSeedPtr != nullptr) {
        // Quantized path: use PrimitivesQuantized for FP32->BF16 transport
        // with stochastic rounding
        PrimitivesQuantized<float, nv_bfloat16, RedOp, 1, COLL_UNROLL>
            primsQuantized(
                tidInGroup,
                groupSize,
                (int*)shmem->recvDims,
                (int*)shmem->sendDims,
                inputBuf,
                outputBuf,
                work->redOpArg,
                group,
                work->quantizeRandomSeedPtr);

        // Large shift per channel ensures different random numbers per channel
        const uint64_t channelOffsetBase = ncclShmem.channelId * (1ULL << 40);

        int step = group;
        while (1) {
          struct ncclPatStep* ps =
              shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
          cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
          while (poll.load(cuda::memory_order_acquire) == 0)
            pollCount++; // Wait for compute thread
          int last = ps->last;
          primsQuantized.patReduce(
              ps, shmem, channelOffsetBase + step * chunkCount);
          if (tidInGroup == 0)
            poll.store(
                0,
                cuda::memory_order_release); // Return element to compute thread
          if (last)
            break;
          step += nGroups;
        }
      } else {
        // Non-quantized path: regular Primitives (same as generic template)
        Primitives<float, RedOp, FanSymmetric<1>, 0, Proto, 0> prims(
            tidInGroup,
            groupSize,
            (int*)shmem->recvDims,
            (int*)shmem->sendDims,
            inputBuf,
            outputBuf,
            work->redOpArg,
            group,
            0,
            0,
            nullptr,
            nullptr,
            0,
            primsModePatRs);

        int step = group;
        while (1) {
          struct ncclPatStep* ps =
              shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
          cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
          while (poll.load(cuda::memory_order_acquire) == 0)
            pollCount++; // Wait for compute thread
          int last = ps->last;
          prims.patReduce(ps, shmem);
          if (tidInGroup == 0)
            poll.store(
                0,
                cuda::memory_order_release); // Return element to compute thread
          if (last)
            break;
          step += nGroups;
        }
      }
    }
#endif
  }
};
