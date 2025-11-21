// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/AllToAllvDedup/WorkerGroup.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/MultiTbSyncDev.cuh"

namespace ctran::alltoallvdedup {
using namespace ctran::algos;

/* Manage worker group used for different phase in the algorithm pipeline. Dedup
 * algorithm defines multiple roles defined in WorkerGroupType. For each role,
 * there can be 1 or multiple groups of workers, each group execute
 * independently (e.g., handle different sendNode|fwdLocalRank etc). A group
 * contains multiple workers, each worker is a thread block. All workers in the
 * same group co-work on the same role and the same data in parallel.
 *
 * The internal sync object is used to synchronize among workers in the current
 * group.
 */
struct WorkerGroup {
  int start;
  int end;
  int numGroups;
  int numWorkers;
  WorkerGroupSync* sync{nullptr};
  // step counters used for different type of worker sync
  int barrierStep;
  int dispatchStep;
  int joinStep;

  __device__ __forceinline__ void init(WorkerGroupSync* sync) {
    this->sync = sync;
    // counters should have been reset in previous prepare()/reset() kernel
    barrierStep = 1;
    dispatchStep = 1;
    joinStep = 1;
  }

  // worker 0 waits for all workers to finish previous parallel work
  __device__ __forceinline__ void syncJoin() {
    const auto goal = joinStep * numWorkers;
    MultiTbSyncDev::join(
        &sync->cnts[kJoinCntIdx], workerId(), numWorkers, goal);
    joinStep++;
  }

  // all workers wait for worker 0 to dispatch the next parallel work
  __device__ __forceinline__ void syncDispatch() {
    MultiTbSyncDev::dispatch(
        &sync->cnts[kDispatchCntIdx], workerId(), numWorkers, dispatchStep);
    dispatchStep++;
  }

  // all workers wait for worker 0 to dispatch the next parallel work
  // and get the value from worker 0. The value type must be safe to be casted
  // to int
  template <typename T>
  __device__ __forceinline__ void syncBcast(T& val) {
    static_assert(std::is_convertible_v<T, int>);
    static_assert(std::is_convertible_v<int, T>);
    static_assert(sizeof(T) <= sizeof(int));

    int valInt = (int)val;
    const auto joinGoal = joinStep * numWorkers;
    MultiTbSyncDev::bcast(
        &sync->cnts[kDispatchCntIdx],
        &sync->cnts[kJoinCntIdx],
        &sync->cnts[kBcastValIdx],
        workerId(),
        numWorkers,
        dispatchStep,
        joinGoal,
        valInt);
    dispatchStep++;
    joinStep++;

    val = (T)valInt;
  }

  // all workers wait for all other workers to arrive
  __device__ __forceinline__ void syncBarrier() {
    const auto goal = barrierStep * numWorkers;
    MultiTbSyncDev::barrier(
        &sync->cnts[kBarrierCntIdx], workerId(), numWorkers, goal);
    barrierStep++;
  }

  __device__ __forceinline__ bool contains(const int bid) const {
    return bid >= start && bid <= end;
  }

  __device__ __forceinline__ int workerId() const {
    return (blockIdx.x - start) % numWorkers;
  }

  __device__ __forceinline__ int groupId() const {
    return (blockIdx.x - start) / numWorkers;
  }
};

__device__ __forceinline__ void assignWorkerGroup(
    const int start,
    const int numGroups,
    const int numWorkers,
    WorkerGroup& group) {
  group.start = start;
  group.numGroups = numGroups;
  group.numWorkers = numWorkers;
  group.end = group.start + numGroups * numWorkers - 1;
}
} // namespace ctran::alltoallvdedup
