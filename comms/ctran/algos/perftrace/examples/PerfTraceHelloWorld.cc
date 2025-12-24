// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/ctran/algos/perftrace/examples/PerfTraceExampleHelper.h"
#include "comms/utils/cvars/nccl_cvars.h"

class PerfTraceHelloWorldTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // always enable INFO logging so it prints Dumped logging path
    setenv("NCCL_DEBUG", "INFO", 1);
    // also see run_hello_world.sh where we sepecify NCCL_CTRAN_PERFTRACE_DIR
    // and take care dir cleanup etc
    setenv("NCCL_CTRAN_ENABLE_PERFTRACE", "1", 1);
    ncclCvarInit();
  }

  void TearDown() override {}
};

namespace ctran::perftrace {

// This test demonstrates the basic usage of PerfTrace as documented in
// README.md. It shows how to create a tracer, record intervals and points,
// add metadata, and collect traces across multiple iterations.
TEST_F(PerfTraceHelloWorldTest, BasicUsage) {
  // Create trace on a single rank. PerfTraceDistHelloWorld.cc demonstrates how
  // multiple ranks can dump traces and be combined.
  traceAlgo(0, 1);
}

} // namespace ctran::perftrace
