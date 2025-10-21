// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"

namespace ctran::alltoallvdedup {

/* Define a sync type used among all forwarding groups in the forwarding rank.
 * Forwarding groups within a rank share the same set of tmpRecvBuff chunks
 * allocated by receiver rank, for temporary device memory saving. Thus, each
 * group needs to sync on which remote chunkIdx to use. The synchronization
 * happens within a GPU, and the sync object is allocated in device memory.
 *
 * Atomic synchronization happens in two layers:
 * - Between worker 0 of all forwarding groups (each group may include multiple
 *   workers, and each worker is one thread block): decide the next chunkIdx to
 *   use for the given receiver rank.
 * - Worker 0 to broadcast the decided chunkIdx to rest of workers in the same
 *   group
 *
 * Code struture:
 * - struct definition in .h to be included by both host and device sides.
 * - Host side allocates the memory, and device side actually uses it. Device
 *   member functions defined in *Dev.cuh.*/
struct alignas(16) FwdGroupSync {
  enum Status {
    kUnset = -1,
  };
  int numGroups;
  int numWorkers;

  // Each forward group (numWorkers number of thread blocks) broadcasts blocks
  // from different remote nodes to all local ranks. To avoid extensive tmp
  // memory usage, we allow all forward groups to share the tmpRecvBuff on local
  // receive ranks. Thus, all forward groups need an atomic counter to sync on
  // which chunkIdx (step % numChunks) of the tmpRecvBuff to be used for a give
  // local receive rank
  int remRecvSteps[CTRAN_MAX_NVL_PEERS];
  // FWD workerId 0 in each group broadcast the decided remote step to other
  // workerId i at remRecvStepsInGroup[groupId * numWorkders + workerId]
  int remRecvStepsInGroup[CTRAN_ALGO_MAX_THREAD_BLOCKS];
};
} // namespace ctran::alltoallvdedup
