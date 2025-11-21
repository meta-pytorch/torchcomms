// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace ctran::alltoallvdedup {
constexpr int kDispatchCntIdx = 0;
constexpr int kJoinCntIdx = 1;
constexpr int kBarrierCntIdx = 2;
constexpr int kBcastValIdx = 3;

struct WorkerGroupSync {
  static constexpr int kNumCnts = 4;
  int cnts[kNumCnts];
};

enum class WorkerGroupType {
  kSend,
  kFwd,
  kRecv,
  kIntraFwd,
  kIntraRecv,
  kNumTypes,
};
} // namespace ctran::alltoallvdedup
