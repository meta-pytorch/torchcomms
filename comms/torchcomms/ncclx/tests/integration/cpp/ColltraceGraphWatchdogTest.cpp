// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Integration tests for TorchComms colltrace-based timeout watchdog.
// Validates that when NCCL_COLLTRACE_TRACE_CUDA_GRAPH=1, TorchCommNCCLX::init()
// automatically enables the colltrace watchdog plugin via
// tryEnableColltraceTimeoutWatchdog(), and that it correctly detects timeouts.

#include <folly/Conv.h>
#include <folly/Random.h>
#include <folly/stop_watch.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual

#include "nccl.h" // @manual

#include "comms/mccl/integration_tests/CollectiveIntegrationTestMixin.h"
#include "comms/mccl/integration_tests/McclIntegrationTestUtil.h"
#include "comms/mccl/tests/CudaTestUtil.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

class ColltraceGraphWatchdogTest : public mccl::CollectiveIntegrationTestMixin,
                                   public ::testing::Test {
 public:
  void SetUp() override {
    int numRanks = 4;

    mccl::CollectiveIntegrationTestMixin::SetUp(
        mccl::CollectiveIntegrationTestMixin::Config{
            .numRanks = numRanks,
            .shouldExitOnFailure = false,
            .env =
                {
                    "NCCL_HPC_JOB_IDS=",
                    "NCCL_SOCKET_IFNAME=eth0",
                    "NCCL_CLIENT_SOCKET_IFNAME=eth0",
                    "NCCL_FASTINIT_MODE=none",
                    "NCCL_SOCKET_IPADDR_PREFIX=",
                    "NCCL_COMMSMONITOR_ENABLE=1",
                    "NCCL_COLLTRACE=trace",
                    "NCCL_COLLTRACE_TRACE_CUDA_GRAPH=1",
                    "TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING=1",
                    "NCCL_CTRAN_ENABLE=1",
                    "NCCL_CTRAN_REGISTRATION_SIZE_CHECK=1",
                    "NCCL_CTRAN_BACKENDS=ib,socket,nvl",
                    "NCCL_DEBUG=INFO",
                    fmt::format(
                        "NCCL_DEBUG_FILE={}",
                        (tmpDir_.path() / "logfile%p").string()),
                },
        });
  }

  folly::test::TemporaryDirectory tmpDir_{
      fmt::format("ColltraceGraphWatchdog{}", folly::Random::rand64())};

  std::shared_ptr<torch::comms::TorchComm>
  createTorchComm(int rank, int worldSize, std::chrono::milliseconds timeout) {
    auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);

    // Connect a new TCPStore client to the mixin's existing server.
    auto mixinStore = getTCPStore();
    auto storeOpts = c10d::TCPStoreOptions{};
    storeOpts.port = mixinStore->getPort();
    storeOpts.isServer = false;
    auto store = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", storeOpts);

    // Wrap in PrefixStore to avoid key collisions with the mixin
    auto prefixStore =
        c10::make_intrusive<c10d::PrefixStore>("torchcomm_test", store);

    torch::comms::CommOptions options;
    options.store = prefixStore;
    options.timeout = timeout;

    return torch::comms::new_comm(
        "ncclx",
        c10::Device(c10::DeviceType::CUDA, deviceId),
        "test_colltrace_watchdog",
        options);
  }

  void testDriverCheckSucceed() {
    ASSERT_TRUE(
        std::holds_alternative<
            mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_));
    auto& testDriverState =
        std::get<mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_);
    EXPECT_THAT(
        testDriverState.workerExitCodes, ::testing::Each(::testing::Eq(0)));
  }

  void testDriverCheckCrashedWithWatchdog() {
    ASSERT_TRUE(
        std::holds_alternative<
            mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_));
    auto& testDriverState =
        std::get<mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_);
    EXPECT_THAT(
        testDriverState.workerExitCodes, ::testing::Each(::testing::Ne(0)));

    // Note: The watchdog's "watchdog timeout" message is logged via
    // XLOG(FATAL) (folly logging -> stderr), not via NCCL's debug logging
    // system (NCCL_DEBUG_FILE). Checking exit codes above is sufficient to
    // validate the watchdog fired and crashed the workers.
  }
};

// Creates a TorchComm via new_comm("ncclx", ...) with
// NCCL_COLLTRACE_TRACE_CUDA_GRAPH=1, verifying that init() automatically
// enables the colltrace watchdog and it detects a timeout when a collective
// hangs.
TEST_F(ColltraceGraphWatchdogTest, TestTimeoutViaTorchCommsInit) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  constexpr auto timeout{std::chrono::seconds{5}};
  int rank = getRank();
  int worldSize = getWorldSize();

  // Create TorchComm via the full init path.
  // NCCL_COLLTRACE_TRACE_CUDA_GRAPH=1 is set in the env, so
  // initNcclxResources() will call tryEnableColltraceTimeoutWatchdog(timeout)
  // which sets crashOnAsyncError, crashOnTimeout, and timeoutMs hints.
  auto torchcomm = createTorchComm(
      rank,
      worldSize,
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));

  ASSERT_NE(torchcomm, nullptr);
  ASSERT_EQ(torchcomm->getRank(), rank);
  ASSERT_EQ(torchcomm->getSize(), worldSize);

  // Warmup AllReduce via TorchComms API
  auto tensor =
      at::ones({32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  torchcomm->all_reduce(
      tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);

  // Trigger timeout: ranks 1-3 sleep while rank 0 issues AllReduce
  if (rank != 0) {
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
    XLOG(FATAL, "COMM FATAL");
  } else {
    torchcomm->all_reduce(
        tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
  }

  FAIL() << "Watchdog did not trigger after timeout is reached";
}

// Captures an AllReduce into a CUDA graph via TorchComms, replays it, and
// verifies the colltrace watchdog detects a timeout during graph replay.
TEST_F(ColltraceGraphWatchdogTest, TestGraphReplayTimeout) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  constexpr auto timeout{std::chrono::seconds{5}};
  int rank = getRank();
  int worldSize = getWorldSize();

  auto torchcomm = createTorchComm(
      rank,
      worldSize,
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));

  ASSERT_NE(torchcomm, nullptr);

  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  auto tensor = at::ones(
      {32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, deviceId));

  // Warmup AllReduce (eager) to trigger pre-connect
  torchcomm->all_reduce(
      tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);

  // Use a non-default stream for graph capture (default stream doesn't
  // support capture). Set it as the current PyTorch stream so TorchComms
  // dispatches work onto it.
  auto captureStream = at::cuda::getStreamFromPool(false, deviceId);
  at::cuda::setCurrentCUDAStream(captureStream);

  // Capture AllReduce into a CUDA graph via TorchComms
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  ASSERT_EQ(
      cudaStreamBeginCapture(
          captureStream.stream(), cudaStreamCaptureModeGlobal),
      cudaSuccess);

  torchcomm->all_reduce(tensor, torch::comms::ReduceOp::SUM, /*async_op=*/true);

  ASSERT_EQ(cudaStreamEndCapture(captureStream.stream(), &graph), cudaSuccess);
  ASSERT_EQ(
      cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
      cudaSuccess);

  // Successful warmup replay — all ranks participate
  ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream.stream()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(captureStream.stream()), cudaSuccess);

  // Timeout replay — ranks 1-3 sleep while rank 0 replays
  if (rank != 0) {
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
    XLOG(FATAL, "COMM FATAL");
  } else {
    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream.stream()), cudaSuccess);
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
  }

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);

  FAIL() << "Watchdog did not trigger after timeout is reached";
}

// Replays a graph N times successfully, then hangs on the N+1th replay.
// Verifies the colltrace watchdog correctly resets timers between replays
// and still detects timeouts on later replays.
TEST_F(
    ColltraceGraphWatchdogTest,
    TestGraphReplayTimeoutAfterSuccessfulReplays) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  constexpr auto timeout{std::chrono::seconds{5}};
  constexpr int kSuccessfulReplays = 3;
  int rank = getRank();
  int worldSize = getWorldSize();

  auto torchcomm = createTorchComm(
      rank,
      worldSize,
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));

  ASSERT_NE(torchcomm, nullptr);

  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  auto tensor = at::ones(
      {32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, deviceId));

  // Warmup AllReduce (eager)
  torchcomm->all_reduce(
      tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);

  // Use a non-default stream for graph capture
  auto captureStream = at::cuda::getStreamFromPool(false, deviceId);
  at::cuda::setCurrentCUDAStream(captureStream);

  // Capture AllReduce into a CUDA graph
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  ASSERT_EQ(
      cudaStreamBeginCapture(
          captureStream.stream(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  torchcomm->all_reduce(tensor, torch::comms::ReduceOp::SUM, /*async_op=*/true);
  ASSERT_EQ(cudaStreamEndCapture(captureStream.stream(), &graph), cudaSuccess);
  ASSERT_EQ(
      cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
      cudaSuccess);

  // N successful replays — all ranks participate
  for (int i = 0; i < kSuccessfulReplays; ++i) {
    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(captureStream.stream()), cudaSuccess);
  }

  // N+1th replay: ranks 1-3 sleep, rank 0 replays → timeout
  if (rank != 0) {
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
    XLOG(FATAL, "COMM FATAL");
  } else {
    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream.stream()), cudaSuccess);
    std::this_thread::sleep_for(timeout + std::chrono::seconds{3});
  }

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);

  FAIL() << "Watchdog did not trigger after timeout is reached";
}

// Same init path, but all ranks complete within the timeout — verifies no
// false positive.
TEST_F(ColltraceGraphWatchdogTest, TestBelowTimeoutSucceeds) {
  if (isTestDriverProcess()) {
    testDriverCheckSucceed();
    return;
  }

  constexpr auto timeout{std::chrono::seconds{60}};
  int rank = getRank();
  int worldSize = getWorldSize();

  auto torchcomm = createTorchComm(
      rank,
      worldSize,
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));

  ASSERT_NE(torchcomm, nullptr);

  auto tensor =
      at::ones({32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  // Warmup
  torchcomm->all_reduce(
      tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);

  // Short delay, well within the 60s timeout
  if (rank != 0) {
    std::this_thread::sleep_for(std::chrono::seconds{5});
  }

  torchcomm->all_reduce(
      tensor, torch::comms::ReduceOp::SUM, /*async_op=*/false);

  torchcomm->finalize();
}
