// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/fake/TorchCommFake.hpp>
#include <gtest/gtest.h>
#include <cstdlib>

namespace torch::comms {

namespace {
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";
} // namespace

class TorchCommReconfigureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);
  }

  void TearDown() override {
    unsetenv(kBackendEnvKey);
  }

  std::shared_ptr<TorchComm> createComm(bool enable_reconfigure = false) {
    at::Device device(at::kCPU);
    CommOptions options;
    options.enable_reconfigure = enable_reconfigure;
    return new_comm(kBackendName, device, "test_comm", options);
  }

  ReconfigureOptions makeOpts() {
    ReconfigureOptions opts;
    opts.uuid = 42;
    opts.handles = std::vector<InitHandle>{"url0"};
    opts.timeout = std::chrono::milliseconds(1000);
    return opts;
  }
};

TEST_F(TorchCommReconfigureTest, ReconfigureSuccess) {
  auto comm = createComm(/*enable_reconfigure=*/true);
  ASSERT_NE(comm, nullptr);

  // In dynamic regime, ranks_ starts empty
  EXPECT_TRUE(comm->getRanks().empty());

  // Reconfigure with fake backend (size=1) should populate ranks
  comm->reconfigure(makeOpts());

  auto ranks = comm->getRanks();
  ASSERT_EQ(ranks.size(), 1);
  EXPECT_EQ(ranks[0], 0);
}

TEST_F(TorchCommReconfigureTest, ReconfigureWithZeroSizeClearsRanks) {
  auto comm = createComm(/*enable_reconfigure=*/true);
  ASSERT_NE(comm, nullptr);

  // First reconfigure succeeds — populates ranks
  comm->reconfigure(makeOpts());
  ASSERT_EQ(comm->getRanks().size(), 1);

  // Simulate failure by setting backend size to 0
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->setSize(0);

  // Second reconfigure with size=0 should clear ranks, not crash
  comm->reconfigure(makeOpts());
  EXPECT_TRUE(comm->getRanks().empty());
}

TEST_F(TorchCommReconfigureTest, ReconfigureFailureOnFirstCall) {
  auto comm = createComm(/*enable_reconfigure=*/true);
  ASSERT_NE(comm, nullptr);

  // Set backend size to 0 before first reconfigure
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->setSize(0);

  // First reconfigure with size=0 — this is the original SEV scenario.
  // Without the fix, this would crash with reserve(SIZE_MAX).
  comm->reconfigure(makeOpts());
  EXPECT_TRUE(comm->getRanks().empty());
}

TEST_F(TorchCommReconfigureTest, ReconfigureTrueFailureDoesNotPopulateRanks) {
  auto comm = createComm(/*enable_reconfigure=*/true);
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->setReconfigureFailure(true);

  // Reconfigure with failure -- work reports ERROR, ranks stay empty,
  // isInitialized() returns false.
  auto work = comm->reconfigure(makeOpts());
  EXPECT_FALSE(work->isCompleted());
  EXPECT_TRUE(comm->getRanks().empty());
  EXPECT_FALSE(backend->isInitialized());
}

TEST_F(TorchCommReconfigureTest, ReconfigureFailureThenSuccessRecovers) {
  auto comm = createComm(/*enable_reconfigure=*/true);
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  // Step 1: Failure
  backend->setReconfigureFailure(true);
  comm->reconfigure(makeOpts());
  EXPECT_TRUE(comm->getRanks().empty());

  // Step 2: Recovery
  backend->setReconfigureFailure(false);
  comm->reconfigure(makeOpts());

  auto ranks = comm->getRanks();
  ASSERT_EQ(ranks.size(), 1);
  EXPECT_EQ(ranks[0], 0);
  EXPECT_TRUE(backend->isInitialized());
}

TEST_F(TorchCommReconfigureTest, ReconfigureSuccessThenFailureClearsRanks) {
  auto comm = createComm(true);
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());

  // Success: populate ranks
  comm->reconfigure(makeOpts());
  ASSERT_EQ(comm->getRanks().size(), 1);

  // True failure: ranks must be cleared, not stale
  backend->setReconfigureFailure(true);
  comm->reconfigure(makeOpts());
  EXPECT_TRUE(comm->getRanks().empty());
}

} // namespace torch::comms
