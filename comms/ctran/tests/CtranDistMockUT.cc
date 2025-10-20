// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "CtranUtUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class CtranTest : public NcclxBaseTest, public CtranBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  };
};

// FIXME: below test requires manual change code to inject failure in ctran.
// It will be automated once failure mock system is ready
TEST_F(CtranTest, DISABLED_CommAbortAtAsyncInitFailure) {
  ncclUniqueId id;
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Initializing NCCL
  auto res = ncclCommInitRankConfig(&comm, numRanks, id, globalRank, &config);
  ASSERT_TRUE(res == ncclSuccess || res == ncclInProgress);

  // Ensure init has reached the failing point
  ncclResult_t state;
  do {
    NCCLCHECK_TEST(ncclCommGetAsyncError(comm, &state));
    // Handle outside events, timeouts, progress, ...
  } while (state == ncclInProgress);

  // Expect commAbort can still finish
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);
}

// FIXME: below test requires manual change code to inject failure in ctran.
// It will be automated once failure mock system is ready
TEST_F(CtranTest, DISABLED_CommDestroyAtAsyncInitFailure) {
  ncclUniqueId id;
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Initializing NCCL
  auto res = ncclCommInitRankConfig(&comm, numRanks, id, globalRank, &config);
  ASSERT_TRUE(res == ncclSuccess || res == ncclInProgress);

  // Ensure init has reached the failing point
  ncclResult_t state;
  do {
    NCCLCHECK_TEST(ncclCommGetAsyncError(comm, &state));
    // Handle outside events, timeouts, progress, ...
  } while (state == ncclInProgress);

  // Expect commDestroy can still finish
  res = ncclCommDestroy(comm);
  ASSERT_EQ(res, ncclSuccess);
}

// FIXME: below test requires manual change code to inject failure in ctran.
// It will be automated once failure mock system is ready
TEST_F(CtranTest, DISABLED_CommAbortAtAsyncInitFailureBlocking) {
  ncclUniqueId id;
  ncclComm_t comm;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Initializing NCCL
  auto res = ncclCommInitRankConfig(&comm, numRanks, id, globalRank, nullptr);
  ASSERT_TRUE(res == ncclSuccess);

  // Expect commAbort can still finish
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);
}

// FIXME: below test requires manual change code to inject failure in ctran.
// It will be automated once failure mock system is ready
TEST_F(CtranTest, DISABLED_CommDestroyAtAsyncInitFailureBlocking) {
  ncclUniqueId id;
  ncclComm_t comm;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Initializing NCCL
  auto res = ncclCommInitRankConfig(&comm, numRanks, id, globalRank, nullptr);
  ASSERT_TRUE(res == ncclSuccess);

  // Expect commDestroy can still finish
  res = ncclCommDestroy(comm);
  ASSERT_EQ(res, ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
