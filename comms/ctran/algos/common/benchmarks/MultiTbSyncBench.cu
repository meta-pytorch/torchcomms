// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cooperative_groups.h>
#include "comms/ctran/algos/common/MultiTbSyncDev.cuh"

namespace cg = cooperative_groups;

using namespace ctran::algos;

//------------------------------------------------------------------------------
// Sync Type Enum (must match MultiTbSyncBench.cc)
//------------------------------------------------------------------------------

enum PerfSyncType {
  kBarrier,
  kFence,
  kDispatch,
  kJoin,
  kSignal,
  kSignalWithSync,
  kBcast,
  kClusterSync
};

//------------------------------------------------------------------------------
// Reset Kernel
//------------------------------------------------------------------------------

__global__ void MultiTbSyncTestResetKernel(int* shmCnts, int numCnts) {
  auto gtId = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = gtId; i < numCnts; i += gridDim.x * blockDim.x) {
    MultiTbSyncDev::reset(shmCnts + i);
  }
}

//------------------------------------------------------------------------------
// Performance Benchmark Kernel
//------------------------------------------------------------------------------

template <PerfSyncType syncType>
__global__ void MultiTbSyncTestPerfKernel(
    const int numWorkers,
    const int numIter,
    const int runId,
    int* shmCnt) {
  int stepVal = 1;
  const auto workerId = blockIdx.x % numWorkers;

  for (int x = 0; x < numIter; x++) {
    if constexpr (syncType == PerfSyncType::kBarrier) {
      MultiTbSyncDev::barrier(
          shmCnt, workerId, numWorkers, stepVal++ * numWorkers);

    } else if constexpr (syncType == PerfSyncType::kFence) {
      __threadfence();
      MultiTbSyncDev::barrier(
          shmCnt, workerId, numWorkers, stepVal++ * numWorkers);
      __threadfence();

    } else if constexpr (syncType == PerfSyncType::kDispatch) {
      MultiTbSyncDev::dispatch(shmCnt, workerId, numWorkers, stepVal++);

    } else if constexpr (syncType == PerfSyncType::kJoin) {
      MultiTbSyncDev::join(
          shmCnt, workerId, numWorkers, stepVal++ * numWorkers);

    } else if constexpr (syncType == PerfSyncType::kSignalWithSync) {
      if (workerId == 0) {
        MultiTbSyncDev::signal<int, true>(shmCnt, stepVal);
      } else {
        while (MultiTbSyncDev::checkSignal(shmCnt, stepVal) == false)
          ;
      }
      stepVal++;

    } else if constexpr (syncType == PerfSyncType::kSignal) {
      if (workerId == 0) {
        MultiTbSyncDev::signal<int, false>(shmCnt, stepVal);
      } else {
        while (MultiTbSyncDev::checkSignal(shmCnt, stepVal) == false)
          ;
      }
      stepVal++;

    } else if constexpr (syncType == PerfSyncType::kBcast) {
      int val = 0;
      MultiTbSyncDev::bcast(
          &shmCnt[0], // dispatch sync
          &shmCnt[1], // join sync
          &shmCnt[2], // value
          workerId,
          numWorkers,
          stepVal,
          stepVal,
          val);
      stepVal++;

    } else if constexpr (syncType == PerfSyncType::kClusterSync) {
#if __CUDA_ARCH__ >= 900
      cg::cluster_group cluster = cg::this_cluster();
      cluster.sync();
#else
      printf("ClusterSync not supported on this arch\n");
#endif
    } else {
      printf("Unsupported sync type\n");
    }
  }
}

//------------------------------------------------------------------------------
// Explicit Template Instantiations
//------------------------------------------------------------------------------

#define DECL_MULTI_TB_SYNC_PERF_KERN(SYNC)                  \
  template __global__ void MultiTbSyncTestPerfKernel<SYNC>( \
      const int numWorkers, const int numIter, const int runId, int* shmCnt);

DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kBarrier);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kDispatch);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kJoin);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kFence);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kSignal);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kSignalWithSync);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kClusterSync);
DECL_MULTI_TB_SYNC_PERF_KERN(PerfSyncType::kBcast);
