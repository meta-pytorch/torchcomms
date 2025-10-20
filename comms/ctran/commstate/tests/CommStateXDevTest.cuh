// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/commstate/CommStateXDev.h"

struct CommStateXDevTestOutput {
  int rank{0};
  int nRanks{0};
  int localRank{0};
  int nLocalRanks{0};
  int node{0};
  int nNodes{0};
  int pid{0};
  uint64_t commHash{0};
  int localRankToRank{0};
};

__global__ void commStateXDevKernel(
    ctran::CommStateXDev statex,
    int queryLocalRank,
    int queryNode,
    CommStateXDevTestOutput* output);
