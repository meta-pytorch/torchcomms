// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"

namespace ctran::alltoallvdedup {

/* Define a sync type used among all workers within a group.
 *
 * Code struture:
 * - struct definition in .h to be included by both host and device sides.
 * - Host side allocates the memory, and device side actually uses it. Device
 *   member functions defined in *Dev.cuh.*/
struct alignas(16) WorkerSync {
  enum GroupType { kSend, kFwd, kRecv };

  int sendGroups[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  int fwdGroups[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  int recvGroups[CTRAN_ALGO_MAX_THREAD_BLOCKS];
};
} // namespace ctran::alltoallvdedup
