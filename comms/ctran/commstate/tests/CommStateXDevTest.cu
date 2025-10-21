// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/commstate/tests/CommStateXDevTest.cuh"

using namespace ctran;

__global__ void commStateXDevKernel(
    CommStateXDev statex,
    int queryLocalRank,
    int queryNode,
    CommStateXDevTestOutput* output) {
  // query from statex, and store in host pinned memory output for test body to
  // check
  output->rank = statex.rank();
  output->nRanks = statex.nRanks();
  output->localRank = statex.localRank();
  output->nLocalRanks = statex.nLocalRanks();
  output->node = statex.node();
  output->nNodes = statex.nNodes();
  output->pid = statex.pid();
  output->commHash = statex.commHash();
  output->localRankToRank = statex.localRankToRank(queryLocalRank, queryNode);
}
