// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/trainer/TrainerContext.h"

#include <atomic>

// Reset to -1 to indicate that the iteration is not set.
std::atomic_int64_t gIteration = -1;

void ncclxSetIteration(int64_t iteration) {
  gIteration = iteration;
}

int64_t ncclxGetIteration() {
  return gIteration;
}
