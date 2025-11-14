// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

enum PerfSyncType {
  kBarrier,
  kFence,
  kDispatch,
  kJoin,
  kSignal,
  kSignalWithSync,
  kClusterSync
};
enum TestSyncType { kFullBarrier, kDispatchJoin, kOneSideSignal };

#define WORKER_ID_TO_VAL(workerId, count, bid, x) \
  (workerId * count + bid + 100000 * x)
