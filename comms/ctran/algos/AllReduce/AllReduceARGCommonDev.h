// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>
#include <cstddef>

#include "comms/ctran/utils/DevAttribute.h"

namespace ctran::allreduce::arg {

enum KAllReduceARG {
  kIntraAllToAll = 0,
  kLocalReduce = 1,
  kIntraAllGather = 2
};

struct AllReduceARGContext {
  int localRank;
  int nLocalRanks;
  int rank;
  int nRanks;
  size_t tmpbuffSize;
  int typeSize;
  size_t count;
  int nSteps;
  // Total number of elements to process across all steps (aligned to nRanks)
  size_t totalStepCount;
  // Number of elements to process in each step based on tmpbuffSize
  size_t stepCount;
  // Per-rank offset for accessing user buffer within each step
  size_t buffOffset;
  // Per-rank offset for accessing temporary buffer within each step
  size_t tmpbuffOffset;
  // Cumulative offset into user buffer from previous steps
  size_t displOffset;
  // Number of remaining elements in the last step (count % nRanks)
  size_t remCount;
};

// total number of elements for all steps
DEVICE_ATTRIBUTE size_t
getTotalStepCount(const size_t count, const size_t nRanks) {
  auto totalStepCount = count / nRanks;
  totalStepCount = totalStepCount * nRanks / nRanks;
  return totalStepCount;
}

// number of elements for each step
DEVICE_ATTRIBUTE size_t getStepCount(
    const size_t tmpbuffSize,
    const size_t nRanks,
    const int typeSize,
    const size_t totalStepCount) {
  auto stepCount = tmpbuffSize / nRanks / typeSize;
  if (stepCount > totalStepCount) {
    stepCount = totalStepCount;
  }
  return stepCount;
}

// number of steps
DEVICE_ATTRIBUTE int getNumSteps(
    const size_t stepCount,
    const size_t totalStepCount) {
  return stepCount ? (totalStepCount + stepCount - 1) / stepCount : 0;
}

DEVICE_ATTRIBUTE void prepareContext(AllReduceARGContext& context) {
  context.totalStepCount = getTotalStepCount(context.count, context.nRanks);
  context.stepCount = getStepCount(
      context.tmpbuffSize,
      context.nRanks,
      context.typeSize,
      context.totalStepCount);
  context.nSteps = getNumSteps(context.stepCount, context.totalStepCount);
  context.remCount = context.count % context.nRanks;
  context.buffOffset = context.totalStepCount;
  context.tmpbuffOffset = context.stepCount;
}

DEVICE_ATTRIBUTE void updateContext(AllReduceARGContext& context) {
  context.totalStepCount -= context.stepCount;
  context.displOffset += context.stepCount;
  if (context.stepCount > context.totalStepCount) {
    context.stepCount = context.totalStepCount;
  }
  context.tmpbuffOffset = context.stepCount;
}

DEVICE_ATTRIBUTE void updateContextForRemainder(AllReduceARGContext& context) {
  context.stepCount = context.remCount;
  context.displOffset = context.count - context.remCount;
  context.buffOffset = 0;
  context.tmpbuffOffset = context.stepCount;
  context.totalStepCount = 0;
}

DEVICE_ATTRIBUTE size_t
getUserbuffOffset(const AllReduceARGContext& context, const int rank) {
  return context.buffOffset * rank + context.displOffset;
}

DEVICE_ATTRIBUTE size_t
getTmpbuffOffset(const AllReduceARGContext& context, const int rank) {
  return context.tmpbuffOffset * rank;
}

} // namespace ctran::allreduce::arg

template <>
struct fmt::formatter<ctran::allreduce::arg::KAllReduceARG>
    : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ctran::allreduce::arg::KAllReduceARG status, FormatContext& ctx)
      const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
