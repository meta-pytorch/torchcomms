// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/WorkerGroupDev.cuh"
#include "comms/ctran/algos/AllToAllvDedup/tests/WorkerGroupTest.cuh"

using namespace ctran::alltoallvdedup;

__global__ void testAssignMultiWorkerGroupKernel(
    const int numRoles,
    const int* numGroups,
    const int* numWorkers,
    WorkerGroupInfo* outputs) {
  int startBid = 0;
  WorkerGroup groups[kMaxNumRoles];
  const auto outputOffset = blockIdx.x * numRoles;
  for (int i = 0; i < numRoles; i++) {
    assignWorkerGroup(startBid, numGroups[i], numWorkers[i], groups[i]);
    startBid = groups[i].end + 1;

    if (threadIdx.x == 0) {
      auto* output = outputs + outputOffset + i;
      output->start = groups[i].start;
      output->end = groups[i].end;
      output->numGroups = groups[i].numGroups;
      output->numWorkers = groups[i].numWorkers;
      if (groups[i].contains(blockIdx.x)) {
        output->workerId = groups[i].workerId();
        output->groupId = groups[i].groupId();
      } else {
        output->workerId = -1;
        output->groupId = -1;
      }
    }
  }
}

__global__ void testWorkerGroupSyncKernel(
    const int numGroups,
    const int numWorkers,
    const int numIter,
    WorkerGroupSync* syncs,
    int* outputs) {
  WorkerGroup testGroup;
  assignWorkerGroup(0, numGroups, numWorkers, testGroup);

  // each group uses a different sync object
  testGroup.init(syncs + testGroup.groupId());

  // not really check the correctness of each sync, since they are already
  // tested in algos/comms/tests/MultiTbSyncTest.cc
  // Only checked syncBcast return to ensure the right set of workers in each
  // group are synchronized.
  for (int i = 0; i < numIter; i++) {
    int val = i * (int)gridDim.x + (int)blockIdx.x;
    testGroup.syncBcast(val);

    // will fail at static_assert(sizeof(T) <= sizeof(int)) at build time
    // size_t val1 = 0;
    // testGroup.syncBcast(val1);

    testGroup.syncBarrier();
    testGroup.syncDispatch();
    testGroup.syncJoin();

    if (threadIdx.x == 0) {
      // return to host side to check
      outputs[i * gridDim.x + blockIdx.x] = val;
    }
  }
}
