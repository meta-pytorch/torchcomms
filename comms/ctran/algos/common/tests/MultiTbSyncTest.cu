// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#if __CUDA_ARCH__ >= 900
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif
#include "comms/ctran/algos/common/MultiTbSyncDev.cuh"
#include "comms/ctran/algos/common/tests/MultiTbSyncTest.cuh"

using namespace ctran::algos;

__global__ void MultiTbSyncTestResetKernel(int* shmCnts, int numCnts) {
  auto gtId = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = gtId; i < numCnts; i += gridDim.x * blockDim.x) {
    MultiTbSyncDev::reset(shmCnts + i);
  }
}

#define WORKER_ID_TO_VAL(workerId, count, bid, x) \
  (workerId * count + bid + 100000 * x)

template <TestSyncType syncType>
__global__ void MultiTbSyncTestKernel(
    const int numWorkers,
    const int numIter,
    const int count,
    int* shmData,
    int* shmCnts,
    int* outData) {
  const auto workerId = blockIdx.x % numWorkers;
  int syncVal = 1, dispatchVal = 1, joinVal = 1;
  // different type of sync should use different counters
  const int syncCntId = 0, dispCntId = 0, joinCntId = 1;

  const auto prevWorkerId = (workerId + numWorkers - 1) % numWorkers;
  const auto nextWorkerId = (workerId + 1) % numWorkers;

  for (int x = 0; x < numIter; x++) {
    // Ensure all reads have finished before next iteration writes to shmData
    if (syncType == TestSyncType::kFullBarrier) {
      MultiTbSyncDev::barrier(
          &shmCnts[syncCntId], workerId, numWorkers, syncVal++ * numWorkers);
    } else if (syncType == TestSyncType::kDispatchJoin) {
      MultiTbSyncDev::dispatch(
          &shmCnts[dispCntId], workerId, numWorkers, dispatchVal++);
    } else {
      // Every worker signals for previous iteration data read by itself, and
      // waits for the previous worker's signal on whether they have finished
      // read for shmData to be updated by this worker
      MultiTbSyncDev::signal<int, true>(&shmCnts[workerId], syncVal);
      MultiTbSyncDev::waitSignal(&shmCnts[prevWorkerId], syncVal);

      // FIXME: mode/dev seems compiling `while
      // (MultiTbSyncDev::checkSignal(&shmCnts[prevWorkerId], syncVal) ==
      // false);` differently than mode/opt, which worker overrides shmData
      // while the prevWorkerId is still reading for previous round. Need
      // followup to understand the compiled code difference.
      syncVal++;
    }

    // each worker group assigns different range of the shmData buffer
    const auto myOffset = workerId * count;
    for (auto i = threadIdx.x; i < count; i += blockDim.x) {
      shmData[myOffset + i] = WORKER_ID_TO_VAL(workerId, count, i, x);
    }

    __threadfence();
    if (syncType == TestSyncType::kFullBarrier) {
      MultiTbSyncDev::barrier(
          &shmCnts[syncCntId], workerId, numWorkers, syncVal++ * numWorkers);
    } else if (syncType == TestSyncType::kDispatchJoin) {
      MultiTbSyncDev::join(
          &shmCnts[joinCntId], workerId, numWorkers, joinVal++ * numWorkers);
    } else {
      // Every worker signals for shmData updated by itself, and waits for the
      // next worker's signal
      MultiTbSyncDev::signal<int, true>(&shmCnts[workerId], syncVal);
      MultiTbSyncDev::waitSignal(&shmCnts[nextWorkerId], syncVal);
      syncVal++;
    }
    __threadfence();

    if (syncType == TestSyncType::kFullBarrier ||
        syncType == TestSyncType::kOneSideSignal) {
      // Every worker copies the data written by the next worker. Result
      // from all iterations will be returned to host side for checking.
      const auto nextOffset = ((workerId + 1) % numWorkers) * count;
      const auto outOffset = count * numWorkers * x + myOffset;
      for (auto i = threadIdx.x; i < count; i += blockDim.x) {
        outData[outOffset + i] = shmData[nextOffset + i];
      }
    } else {
      if (workerId == 0) {
        // worker 0 copies the data written all other workers. Result
        // from all iterations will be returned to host side for checking.
        const auto iterOffset = count * numWorkers * x;
        for (auto i = threadIdx.x; i < count * numWorkers; i += blockDim.x) {
          // Follow similar pattern as above, copy data of workerId to
          // the slot of workerId+1
          const auto wId = i / count;
          const auto j = i % count;
          const auto nextOffset = ((wId + 1) % numWorkers) * count;
          outData[iterOffset + i] = shmData[nextOffset + j];
        }
      }
    }
  }
}

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

#define DECL_MULTI_TB_SYNC_TEST_KERN(SYNC)              \
  template __global__ void MultiTbSyncTestKernel<SYNC>( \
      const int numWorkers,                             \
      const int numIter,                                \
      const int count,                                  \
      int* shmData,                                     \
      int* shmCnts,                                     \
      int* outData);

DECL_MULTI_TB_SYNC_TEST_KERN(TestSyncType::kFullBarrier);
DECL_MULTI_TB_SYNC_TEST_KERN(TestSyncType::kDispatchJoin);
DECL_MULTI_TB_SYNC_TEST_KERN(TestSyncType::kOneSideSignal);
