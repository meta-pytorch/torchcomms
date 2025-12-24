// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "comms/ctran/algos/perftrace/Record.h"
#include "comms/ctran/algos/perftrace/Tracer.h"

namespace ctran::perftrace {

// Common helper function demonstrating PerfTrace usage.
// This function simulates a distributed algorithm that:
// 1. Creates a tracer instance for the rank
// 2. Runs multiple iterations, each with multiple pipelined steps
// 3. Records intervals for phases, RDMA operations, and instant events
//
// @param myRank The rank identifier (0 to numRanks-1)
// @param numRanks Total number of ranks in the distributed environment
// @param numIterations Number of algorithm iterations to trace
// @param numSteps Number of pipelined steps per iteration
inline void traceAlgo(
    const int myRank,
    const int numRanks,
    const int numIterations = 3,
    const int numSteps = 5) {
  // Create a tracer (typically one per lifetime of a rank)
  auto tracer = std::make_unique<Tracer>(myRank);
  EXPECT_TRUE(tracer->isTraceEnabled());

  // Calling an algorithm multiple times
  for (int iter = 0; iter < numIterations; iter++) {
    // Create a record for each algorithm iteration
    auto record = std::make_unique<Record>(
        fmt::format("myAlgorithm_iter{}", iter), myRank);

    // Add metadata for the current iteration
    record->addMetadata("iteration", std::to_string(iter));
    record->addMetadata("rank", std::to_string(myRank));
    record->addMetadata("numRanks", std::to_string(numRanks));

    // Record time intervals for an algorithm phase (e.g., reduceScatter phase
    // in Ring AllReduce). Intervals can be nested, e.g., `compute_phase` and
    // `RDMA_send`. Both interval and point can add optional metadata.
    const std::map<std::string, std::string> phaseMetadata = {
        {"numSteps", std::to_string(numSteps)}};
    record->startInterval("compute_phase", 0, std::nullopt, phaseMetadata);

    // Within each iteration, the algorithm may perform multiple steps (e.g.,
    // pipelined). Each interval {start,end} pair is labeled with seqNum, so
    // they can match even if intervals overlap (e.g., nonblocking RDMA with
    // nonblocking kernel D2D).
    for (int step = 0; step < numSteps; step++) {
      // In a ring algorithm, peer is typically (myRank + 1) % numRanks
      const int peerRank = (myRank + 1) % numRanks;

      // Simulate RDMA send to peer
      record->startInterval("RDMA_send", step, peerRank);
      std::this_thread::sleep_for(std::chrono::milliseconds(1 + myRank));
      record->endInterval("RDMA_send", step);

      // Simulate waiting for RDMA receive from peer
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      const int recvPeer = (myRank - 1 + numRanks) % numRanks;
      record->addPoint("RDMA_received", step, recvPeer);
    }
    record->endInterval("compute_phase", 0);

    // Add record to tracer; automatically record the end time of this record
    tracer->addRecord(std::move(record));
  }

  // tracer destructor will dump to JSON file in path specified by
  // NCCL_CTRAN_PERFTRACE_DIR.
}

} // namespace ctran::perftrace
