// Copyright (c) Meta Platforms, Inc. and affiliates.

// This example demonstrates using PerfTrace in a distributed MPI environment.
// Each MPI rank creates its own Tracer instance and generates trace files.
// The traces can be combined using combined_trace.py to view all ranks
// together.

#include <folly/init/Init.h>

#include "comms/ctran/algos/perftrace/examples/PerfTraceExampleHelper.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Environment that sets up MPI and perftrace-specific configuration
class PerfTraceExampleEnvironment : public meta::comms::MPIEnvironmentBase {
 public:
  void SetUp() override {
    MPIEnvironmentBase::SetUp();

    // Enable INFO logging so it prints the trace file path
    setenv("NCCL_DEBUG", "INFO", 1);
    // Enable perftrace (also set in run.sh via NCCL_CTRAN_PERFTRACE_DIR)
    setenv("NCCL_CTRAN_ENABLE_PERFTRACE", "1", 1);
  }
};

// Test fixture that provides rank information from MPI
class PerfTraceDistTest : public meta::comms::MpiBaseTestFixture {
 public:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

namespace ctran::perftrace {

// This test demonstrates PerfTrace usage in a real MPI distributed environment.
// Each MPI rank runs this test independently and generates its own trace file.
// Use run.sh to launch with mpirun and combine traces for visualization.
TEST_F(PerfTraceDistTest, DistributedTracing) {
  ASSERT_GE(globalRank, 0) << "MPI rank not initialized";
  ASSERT_GT(numRanks, 0) << "MPI numRanks not initialized";

  XLOG(INFO) << "Rank " << globalRank << ": starting PerfTrace example";

  // Each rank traces its own algorithm execution
  traceAlgo(globalRank, numRanks);

  XLOG(INFO) << "Rank " << globalRank << ": PerfTrace example completed";
}

} // namespace ctran::perftrace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new PerfTraceExampleEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
