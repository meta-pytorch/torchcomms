// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/mccl/tests/MockMcclComm.h"
#include "comms/mccl/tests/MockWorkHandle.h"
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/mccl/tests/unit/cpp/MockCudaApi.h"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace torch::comms::test {

namespace {
at::Tensor makeFakeCudaTensor(float* data, size_t numel) {
  auto tensorImpl =
      c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
          at::Storage(
              at::Storage::use_byte_size_t(),
              numel * sizeof(float),
              at::DataPtr(data, at::Device(at::DeviceType::CUDA, 0)),
              nullptr,
              false),
          c10::DispatchKey::CUDA,
          caffe2::TypeMeta::Make<float>());
  tensorImpl->set_sizes_contiguous({static_cast<int64_t>(numel)});
  return at::Tensor(std::move(tensorImpl));
}
} // namespace

class TorchCommMCCLTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_comm_ = std::make_unique<::mccl::testing::MockMcclComm>();
    mock_comm_ptr_ = mock_comm_.get();
    saveEnv("TORCHCOMM_RANK", hadTorchCommRank_, oldTorchCommRank_);
    saveEnv("TORCHCOMM_SIZE", hadTorchCommSize_, oldTorchCommSize_);
    setenv("TORCHCOMM_RANK", "0", 1);
    setenv("TORCHCOMM_SIZE", "2", 1);
  }

  void TearDown() override {
    restoreEnv("TORCHCOMM_RANK", hadTorchCommRank_, oldTorchCommRank_);
    restoreEnv("TORCHCOMM_SIZE", hadTorchCommSize_, oldTorchCommSize_);
  }

  std::unique_ptr<::mccl::testing::MockMcclComm> mock_comm_;
  ::mccl::testing::MockMcclComm* mock_comm_ptr_{};
  bool hadTorchCommRank_{false};
  bool hadTorchCommSize_{false};
  std::string oldTorchCommRank_;
  std::string oldTorchCommSize_;

  static void saveEnv(const char* name, bool& hadValue, std::string& oldValue) {
    if (const char* value = std::getenv(name)) {
      hadValue = true;
      oldValue = value;
    }
  }

  static void
  restoreEnv(const char* name, bool hadValue, const std::string& oldValue) {
    if (hadValue) {
      setenv(name, oldValue.c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  torch::comms::ReconfigureOptions makeDefaultOpts(int64_t uuid = 42) {
    torch::comms::ReconfigureOptions opts;
    opts.uuid = uuid;
    opts.handles = std::vector<torch::comms::InitHandle>{"url0", "url1"};
    opts.timeout = std::chrono::milliseconds(1000);
    return opts;
  }

  // Park a comm in a failed state without starting the watchdog thread, so
  // watchdogIteration() can be driven directly and deterministically. Each
  // consumes mock_comm_, so call at most one of these per test.
  std::shared_ptr<TorchCommMCCL> makeTimedOutComm(
      bool abortProcess,
      bool enableReconfigure) {
    return makeFailedComm(
        TorchCommMCCL::CommState::TIMEOUT, abortProcess, enableReconfigure);
  }

  std::shared_ptr<TorchCommMCCL> makeErroredComm(
      bool abortProcess,
      bool enableReconfigure) {
    return makeFailedComm(
        TorchCommMCCL::CommState::ERROR, abortProcess, enableReconfigure);
  }

  static void runWatchdogIteration(TorchCommMCCL& comm) {
    comm.watchdogIteration();
  }

  // Returns the enum as a bool so TEST_F bodies, which are not friends, can
  // assert on it without naming the private CommState.
  static bool isInFailedState(TorchCommMCCL& comm) {
    return comm.commState_ != TorchCommMCCL::CommState::NORMAL;
  }

  std::shared_ptr<TorchCommMCCL> makeHealthyComm(
      bool abortProcess,
      bool enableReconfigure) {
    return makeFailedComm(
        TorchCommMCCL::CommState::NORMAL, abortProcess, enableReconfigure);
  }

  static void markTimedOut(TorchCommMCCL& comm) {
    comm.commState_ = TorchCommMCCL::CommState::TIMEOUT;
  }

 private:
  // CommState is private to TorchCommMCCL; the fixture is a friend, the derived
  // TEST_F bodies are not, so the enum may only be named here.
  std::shared_ptr<TorchCommMCCL> makeFailedComm(
      TorchCommMCCL::CommState state,
      bool abortProcess,
      bool enableReconfigure) {
    auto mccl = std::make_shared<TorchCommMCCL>(
        std::move(mock_comm_),
        /*rank=*/0,
        /*size=*/2,
        std::make_shared<NiceMock<MockCudaApi>>());
    mccl->options_.abort_process_on_timeout_or_error = abortProcess;
    mccl->options_.enable_reconfigure = enableReconfigure;
    // INITIALIZED, not the constructor's default: production only starts the
    // watchdog once init()/reconfigure() has succeeded. Modelling the reachable
    // state keeps the zero-call assertions from going vacuous if lifecycle
    // validation is ever added, at the cost of a "was not finalized before
    // destruction" line at teardown.
    mccl->initState_ = TorchCommMCCL::InitializationState::INITIALIZED;
    mccl->commState_ = state;
    return mccl;
  }
};

// commState_ has no path back to NORMAL, which is the precondition that makes
// the missing latch harmful. Fails if a future change clears it.
TEST_F(TorchCommMCCLTest, FailedCommStateIsNeverClearedByTheWatchdog) {
  auto mccl = makeTimedOutComm(
      /*abortProcess=*/true, /*enableReconfigure=*/false);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(::testing::AnyNumber());
  ASSERT_TRUE(isInFailedState(*mccl));

  for (int i = 0; i < 25; ++i) {
    runWatchdogIteration(*mccl);
    ASSERT_TRUE(isInFailedState(*mccl)) << "state cleared on iteration " << i;
  }
}

TEST_F(TorchCommMCCLTest, WatchdogFailureBranchRunsExactlyOnce) {
  // Declared before the comm so the hook lambda never outlives its capture.
  int hookRuns = 0;
  auto mccl = makeTimedOutComm(
      /*abortProcess=*/true, /*enableReconfigure=*/false);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(1);

  mccl->registerAbortHook(/*hookId=*/1, [&hookRuns] { ++hookRuns; });

  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);

  EXPECT_EQ(1, hookRuns);
}

TEST_F(TorchCommMCCLTest, WatchdogFailureBranchRunsExactlyOnceOnError) {
  int hookRuns = 0;
  auto mccl = makeErroredComm(
      /*abortProcess=*/true, /*enableReconfigure=*/false);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(1);

  mccl->registerAbortHook(/*hookId=*/1, [&hookRuns] { ++hookRuns; });

  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);

  EXPECT_EQ(1, hookRuns);
}

// Healthy iterations must not consume the latch. Every other test here starts
// already-failed, so this is the only case that catches a hoisted exchange().
TEST_F(TorchCommMCCLTest, HealthyIterationsDoNotConsumeTheLatch) {
  int hookRuns = 0;
  auto mccl = makeHealthyComm(
      /*abortProcess=*/true, /*enableReconfigure=*/false);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(1);

  mccl->registerAbortHook(/*hookId=*/1, [&hookRuns] { ++hookRuns; });

  // NORMAL: the guard returns before the latch is touched.
  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);
  ASSERT_EQ(0, hookRuns);

  markTimedOut(*mccl);
  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);

  // The first real failure must still be handled, exactly once.
  EXPECT_EQ(1, hookRuns);
}

// Reconfigurable comms must survive a timeout so the application can abort,
// reconfigure to a new quorum, and recover in place. This is PAFT's
// inter-replica path.
TEST_F(TorchCommMCCLTest, ReconfigurableCommSkipsWatchdogFailureBranch) {
  // Declared before the comm so the hook lambda never outlives its capture.
  int hookRuns = 0;
  auto mccl = makeTimedOutComm(
      /*abortProcess=*/true, /*enableReconfigure=*/true);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(0);

  mccl->registerAbortHook(/*hookId=*/1, [&hookRuns] { ++hookRuns; });

  runWatchdogIteration(*mccl);
  runWatchdogIteration(*mccl);

  EXPECT_EQ(0, hookRuns);
}

TEST_F(TorchCommMCCLTest, OptedOutCommSkipsWatchdogFailureBranch) {
  int hookRuns = 0;
  auto mccl = makeTimedOutComm(
      /*abortProcess=*/false, /*enableReconfigure=*/false);
  EXPECT_CALL(*mock_comm_ptr_, abort(_)).Times(0);

  mccl->registerAbortHook(/*hookId=*/1, [&hookRuns] { ++hookRuns; });

  runWatchdogIteration(*mccl);

  EXPECT_EQ(0, hookRuns);
}

TEST_F(TorchCommMCCLTest, ForwardsLifecycleEventsFromMccl) {
  const auto timestamp =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{1234}};
  const std::vector<::mccl::LifecycleEvent> expectedEvents{
      {
          .replayId = 7,
          .commId = 11,
          .collId = 13,
          .executionCollId = 17,
          .eventType = ::mccl::LifecycleEventType::Start,
          .timestamp = timestamp,
      },
  };
  EXPECT_CALL(*mock_comm_ptr_, getLifecycleCommId()).WillOnce(Return(11));
  EXPECT_CALL(*mock_comm_ptr_, getLatestLifecycleCollectiveId())
      .WillOnce(Return(13));
  EXPECT_CALL(*mock_comm_ptr_, drainLifecycleEvents())
      .WillOnce(Return(expectedEvents));
  auto comm = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_),
      /*rank=*/0,
      /*size=*/2,
      std::make_shared<NiceMock<MockCudaApi>>());

  EXPECT_EQ(comm->getLifecycleCommId(), 11);
  EXPECT_EQ(comm->getLatestLifecycleCollectiveId(), 13);
  EXPECT_EQ(comm->drainLifecycleEvents(), expectedEvents);
}

TEST_F(TorchCommMCCLTest, CleanInternalStoreDestroysStoreAndBarriers) {
  // Create TorchCommMCCL with mock comm
  // Inject mock CudaApi since init() is not called in this test
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  // Set up internal state to simulate having created an internal store
  mccl->store_ = c10::make_intrusive<c10d::HashStore>();
  mccl->createdInternalStore_ = true;

  // Allocate a small buffer to act as barrierBuffer_ (normally CUDA memory,
  // but the mock allReduce won't actually use it)
  float dummy_buffer = 0.0f;
  mccl->barrierBuffer_ = &dummy_buffer;

  // Set up mock: allReduce should be called for the barrier
  auto mock_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work, waitCpu()).Times(1);
  EXPECT_CALL(*mock_comm_ptr_, allReduce(_))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work))));

  // Call cleanInternalStore
  mccl->cleanInternalStore();

  // Verify store was destroyed
  EXPECT_EQ(mccl->store_, nullptr);

  // Clean up: null out barrierBuffer_ so destructor doesn't try to free it
  mccl->barrierBuffer_ = nullptr;
}

TEST_F(TorchCommMCCLTest, CleanInternalStoreBarrierFailure) {
  // Create TorchCommMCCL with mock comm
  // Inject mock CudaApi since init() is not called in this test
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  // Set up internal state to simulate having created an internal store
  mccl->store_ = c10::make_intrusive<c10d::HashStore>();
  mccl->createdInternalStore_ = true;

  float dummy_buffer = 0.0f;
  mccl->barrierBuffer_ = &dummy_buffer;

  // Set up mock: allReduce throws to simulate failure
  EXPECT_CALL(*mock_comm_ptr_, allReduce(_))
      .WillOnce(::testing::Throw(std::runtime_error("AllReduce failed")));

  // cleanInternalStore should propagate the exception
  EXPECT_THROW(mccl->cleanInternalStore(), std::runtime_error);

  // Store should still have been destroyed (reset happens before the barrier)
  EXPECT_EQ(mccl->store_, nullptr);

  mccl->barrierBuffer_ = nullptr;
}

TEST_F(TorchCommMCCLTest, AbortDelegatesToMcclComm) {
  EXPECT_CALL(
      *mock_comm_ptr_,
      abort(::testing::Truly([](const ::mccl::AbortInfo& info) {
        return info.reason == ::mccl::AbortReason::ABORTED &&
            info.context.empty();
      })))
      .Times(1);

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  mccl->abort();
}

TEST_F(TorchCommMCCLTest, ContextualAbortDelegatesToMcclComm) {
  EXPECT_CALL(
      *mock_comm_ptr_,
      abort(::testing::Truly([](const ::mccl::AbortInfo& info) {
        return info.reason == ::mccl::AbortReason::NETWORK_ERROR &&
            info.context == "peer watchdog";
      })))
      .Times(1);

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  mccl->abort(
      AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "peer watchdog",
      });
}

TEST_F(TorchCommMCCLTest, GetAbortInfoDelegatesToMcclComm) {
  EXPECT_CALL(*mock_comm_ptr_, getAbortInfo())
      .WillOnce(Return(
          ::mccl::AbortInfo{
              .reason = ::mccl::AbortReason::BOOTSTRAP_POLL,
              .context = "origin=health_poll",
          }));

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  EXPECT_EQ(
      mccl->getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::BOOTSTRAP_POLL,
          .context = "origin=health_poll",
      }));
}

TEST_F(TorchCommMCCLTest, IsAbortSupportedDelegatesToMcclComm) {
  EXPECT_CALL(*mock_comm_ptr_, isAbortSupported()).WillOnce(Return(true));

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  EXPECT_TRUE(mccl->isAbortSupported());
}

TEST_F(TorchCommMCCLTest, IsAbortSupportedReturnsFalseWhenDisabled) {
  EXPECT_CALL(*mock_comm_ptr_, isAbortSupported()).WillOnce(Return(false));

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  EXPECT_FALSE(mccl->isAbortSupported());
}

TEST_F(TorchCommMCCLTest, SetTimeoutDelegatesToMcclComm) {
  constexpr std::chrono::milliseconds kDuration{750};
  EXPECT_CALL(*mock_comm_ptr_, setTimeout(kDuration)).Times(1);

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  mccl->setTimeout(kDuration);
}

TEST_F(TorchCommMCCLTest, SetHintsDelegatesToMcclComm) {
  const std::unordered_map<std::string, std::string> kHints{
      {"step", "42"}, {"attempt", "1"}};
  EXPECT_CALL(*mock_comm_ptr_, setHints(kHints)).Times(1);

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  mccl->setHints(kHints);
}

TEST_F(TorchCommMCCLTest, GetTimeoutDelegatesToMcclComm) {
  constexpr std::chrono::milliseconds kDuration{1234};
  EXPECT_CALL(*mock_comm_ptr_, getTimeout())
      .WillOnce(Return(std::optional<std::chrono::milliseconds>{kDuration}));

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  EXPECT_EQ(mccl->getTimeout(), kDuration);
}

TEST_F(TorchCommMCCLTest, GetTimeoutReturnsNulloptWhenUnset) {
  EXPECT_CALL(*mock_comm_ptr_, getTimeout()).WillOnce(Return(std::nullopt));

  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  EXPECT_FALSE(mccl->getTimeout().has_value());
}

TEST_F(TorchCommMCCLTest, DefaultAllReduceTimeoutLeavesMcclTimeoutUnset) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  float data[4] = {};
  auto tensor = makeFakeCudaTensor(data, 4);

  AllReduceOptions options;
  auto opts =
      mccl->getAllReduceOpts(tensor, 0, nullptr, ReduceOp::SUM, options);

  EXPECT_FALSE(opts.timeout.has_value());
}

// Restoring the strong work -> comm ref fails this test: the cycle through
// workq_ leaves nothing to destroy the comm once the watchdog is parked.
TEST_F(TorchCommMCCLTest, QueuedWorkDoesNotKeepCommAlive) {
  auto cuda_api = std::make_shared<NiceMock<MockCudaApi>>();

  auto reconfigure_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*reconfigure_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));
  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(reconfigure_work))));
  EXPECT_CALL(*mock_comm_ptr_, getInitURL()).WillRepeatedly(Return("url0"));
  EXPECT_CALL(*mock_comm_ptr_, getRankAssignment())
      .WillRepeatedly(Return(mccl::RankAssignment({"url0", "url1"})));

  std::promise<std::thread::id> destroyed_on;
  auto destroyed = destroyed_on.get_future();
  std::thread::id watchdog_id;
  // Deliberately outlives the comm.
  c10::intrusive_ptr<TorchWorkMCCL> work;

  {
    // Custom deleter, not make_shared: records which thread destroyed the comm.
    std::shared_ptr<TorchCommMCCL> mccl(
        new TorchCommMCCL(std::move(mock_comm_), 0, 2, cuda_api),
        [&destroyed_on](TorchCommMCCL* comm) {
          delete comm;
          destroyed_on.set_value(std::this_thread::get_id());
        });

    // Dynamic-regime comms have no watchdog until the first reconfigure().
    mccl->reconfigure(makeDefaultOpts(42));
    ASSERT_TRUE(mccl->isInitialized());
    ASSERT_TRUE(mccl->timeout_thread_.joinable());
    watchdog_id = mccl->timeout_thread_.get_id();
    ASSERT_NE(watchdog_id, std::this_thread::get_id());

    auto work_handle = std::make_unique<::mccl::testing::MockWorkHandle>();
    EXPECT_CALL(*work_handle, getResult())
        .WillRepeatedly(Return(
            std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));
    EXPECT_CALL(*work_handle, waitStream(::testing::_))
        .Times(::testing::AnyNumber());
    work = c10::make_intrusive<TorchWorkMCCL>(
        mccl,
        /*stream=*/nullptr,
        /*inputTensors=*/std::vector<at::Tensor>{},
        /*outputTensors=*/std::vector<at::Tensor>{},
        std::move(work_handle),
        std::chrono::milliseconds(600000));
    mccl->workq_.enqueueWork(work, /*stream=*/nullptr);

    // Park the watchdog first: it locks the comm per iteration, so leaving it
    // running makes the owner at scope exit a race.
    mccl->signalWatchdogShutdown();
    mccl->timeout_thread_.join();
  }

  ASSERT_EQ(
      std::future_status::ready, destroyed.wait_for(std::chrono::seconds(0)))
      << "queued work kept the comm alive past its last external reference";
  EXPECT_EQ(std::this_thread::get_id(), destroyed.get())
      << "comm was not destroyed on the thread that dropped the last reference";

  EXPECT_NO_THROW(work->getCurrentCUDAStream());
  EXPECT_NO_THROW(work->wait());
}

TEST_F(TorchCommMCCLTest, ExplicitAllReduceTimeoutIsPreserved) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  float data[4] = {};
  auto tensor = makeFakeCudaTensor(data, 4);
  constexpr std::chrono::milliseconds kTimeout{1234};

  AllReduceOptions options;
  options.timeout = kTimeout;
  auto opts =
      mccl->getAllReduceOpts(tensor, 0, nullptr, ReduceOp::SUM, options);

  ASSERT_TRUE(opts.timeout.has_value());
  EXPECT_EQ(*opts.timeout, kTimeout);
}

// FT enabled with no explicit setTimeout(): init must seed the comm-level
// default so omitted per-op timeouts still get a deadline.
TEST_F(TorchCommMCCLTest, InitSeedsCommDefaultTimeoutWhenUnset) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());
  constexpr std::chrono::milliseconds kTimeout{1234};
  CommOptions options;
  options.timeout = kTimeout;

  EXPECT_CALL(*mock_comm_ptr_, getTimeout()).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*mock_comm_ptr_, setTimeout(kTimeout)).Times(1);

  mccl->init(at::Device(at::kCUDA, 0), "test", options);
  mccl->finalize();
}

// A comm-level timeout already set (e.g. PAFT's set_timeout) must not be
// clobbered by init.
TEST_F(TorchCommMCCLTest, InitDoesNotClobberExistingTimeout) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());
  constexpr std::chrono::milliseconds kExistingTimeout{15000};

  EXPECT_CALL(*mock_comm_ptr_, getTimeout()).WillOnce(Return(kExistingTimeout));
  EXPECT_CALL(*mock_comm_ptr_, setTimeout(_)).Times(0);

  mccl->init(at::Device(at::kCUDA, 0), "test", CommOptions());
  mccl->finalize();
}

TEST_F(TorchCommMCCLTest, CpuBroadcastDoesNotQueryCudaGraphCaptureAndWaitsCpu) {
  // NiceMock (not StrictMock): reconfigure() starts the timeout watchdog, which
  // makes incidental CUDA calls (e.g. setDevice) on a background thread. The
  // invariant under test is that the CPU broadcast never queries graph capture.
  auto cuda_api = std::make_shared<NiceMock<MockCudaApi>>();
  EXPECT_CALL(*cuda_api, streamIsCapturing(_, _)).Times(0);
  auto mccl =
      std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2, cuda_api);

  auto reconfigure_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*reconfigure_work, waitCpu()).Times(1);
  EXPECT_CALL(*reconfigure_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(reconfigure_work))));
  EXPECT_CALL(*mock_comm_ptr_, getInitURL()).WillOnce(Return("url0"));
  EXPECT_CALL(*mock_comm_ptr_, getRankAssignment())
      .WillOnce(Return(mccl::RankAssignment({"url0", "url1"})));

  mccl->reconfigure(makeDefaultOpts(42));
  ASSERT_TRUE(mccl->isInitialized());

  auto tensor = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
  auto broadcast_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*broadcast_work, waitCpu()).Times(1);
  EXPECT_CALL(*broadcast_work, getResult())
      .WillOnce(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(*mock_comm_ptr_, broadcast(_))
      .WillOnce(
          [&broadcast_work, &tensor](const ::mccl::BroadcastOpts& opts)
              -> std::unique_ptr<::mccl::IWorkHandle> {
            EXPECT_EQ(opts.data, tensor.data_ptr());
            EXPECT_EQ(opts.numElements, static_cast<size_t>(tensor.numel()));
            EXPECT_EQ(opts.stream, nullptr);
            EXPECT_EQ(opts.root, 0);
            EXPECT_EQ(opts.deviceType, ::mccl::DeviceType::Cpu);
            return std::move(broadcast_work);
          });

  auto work = mccl->broadcast(tensor, /*root=*/0, /*async_op=*/false);
  auto mccl_work = c10::static_intrusive_pointer_cast<TorchWorkMCCL>(work);
  EXPECT_TRUE(mccl_work->isCpuWork());
  EXPECT_TRUE(work->isCompleted());

  EXPECT_NO_THROW(mccl->finalize());
}

TEST_F(TorchCommMCCLTest, ReconfigureSuccessMarksInitialized) {
  constexpr int64_t kUuid = int64_t{1} << 62;
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  auto mock_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work, waitCpu()).Times(1);
  EXPECT_CALL(*mock_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(
      *mock_comm_ptr_,
      reconfigure(::testing::Truly([](const ::mccl::InitOpts& opts) {
        return opts.uuid == std::to_string(kUuid);
      })))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work))));
  EXPECT_CALL(*mock_comm_ptr_, getInitURL()).WillOnce(Return("url0"));
  EXPECT_CALL(*mock_comm_ptr_, getRankAssignment())
      .WillOnce(Return(mccl::RankAssignment({"url0", "url1"})));

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(kUuid);

  mccl->reconfigure(opts);

  // Public API assertions
  EXPECT_TRUE(mccl->isInitialized());
  EXPECT_EQ(mccl->getRank(), 0);
  EXPECT_EQ(mccl->getSize(), 2);
  EXPECT_EQ(mccl->getUuid(), std::to_string(kUuid));
}

TEST_F(TorchCommMCCLTest, ReconfigureFailureDoesNotMarkInitialized) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  auto mock_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work, waitCpu()).Times(1);
  EXPECT_CALL(*mock_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{
              .code = commInternalError, .message = "test failure"})));

  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work))));

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(42);

  mccl->reconfigure(opts);

  // Public API assertions — uninitialized backend must throw on getRank/getSize
  EXPECT_FALSE(mccl->isInitialized());
  EXPECT_THROW(mccl->getRank(), std::runtime_error);
  EXPECT_THROW(mccl->getSize(), std::runtime_error);
  EXPECT_TRUE(mccl->getUuid().empty());
}

TEST_F(TorchCommMCCLTest, ReReconfigureFailureResetsToUninitialized) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  // Step 1: Successful reconfigure
  auto mock_work_success1 = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work_success1, waitCpu()).Times(1);
  EXPECT_CALL(*mock_work_success1, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  // Step 2: Failed reconfigure
  auto mock_work_fail = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work_fail, waitCpu()).Times(1);
  EXPECT_CALL(*mock_work_fail, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{
              .code = commInternalError, .message = "test failure"})));

  // Step 3: Recovery reconfigure
  auto mock_work_success2 = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mock_work_success2, waitCpu()).Times(1);
  EXPECT_CALL(*mock_work_success2, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work_success1))))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work_fail))))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work_success2))));
  EXPECT_CALL(*mock_comm_ptr_, getInitURL()).WillRepeatedly(Return("url0"));
  EXPECT_CALL(*mock_comm_ptr_, getRankAssignment())
      .WillRepeatedly(Return(mccl::RankAssignment({"url0", "url1"})));

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(42);

  // Step 1: First reconfigure succeeds
  mccl->reconfigure(opts);
  EXPECT_TRUE(mccl->isInitialized());
  EXPECT_EQ(mccl->getRank(), 0);
  EXPECT_EQ(mccl->getSize(), 2);
  EXPECT_EQ(mccl->getUuid(), "42");

  // Step 2: Second reconfigure fails — must reset to UNINITIALIZED
  opts.uuid = 43;
  mccl->reconfigure(opts);
  EXPECT_FALSE(mccl->isInitialized());
  EXPECT_THROW(mccl->getRank(), std::runtime_error);
  EXPECT_THROW(mccl->getSize(), std::runtime_error);
  EXPECT_TRUE(mccl->getUuid().empty());

  // Step 3: Third reconfigure succeeds — recovery works
  opts.uuid = 44;
  mccl->reconfigure(opts);
  EXPECT_TRUE(mccl->isInitialized());
  EXPECT_EQ(mccl->getRank(), 0);
  EXPECT_EQ(mccl->getSize(), 2);
  EXPECT_EQ(mccl->getUuid(), "44");

  mccl->finalize();
  EXPECT_TRUE(mccl->getUuid().empty());
}

TEST_F(TorchCommMCCLTest, ReconfigureExceptionDoesNotMarkInitialized) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(::testing::Throw(std::runtime_error("reconfigure failed")));

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(42);

  EXPECT_THROW(mccl->reconfigure(opts), std::runtime_error);

  // Public API assertions
  EXPECT_FALSE(mccl->isInitialized());
  EXPECT_THROW(mccl->getRank(), std::runtime_error);
  EXPECT_THROW(mccl->getSize(), std::runtime_error);
  EXPECT_TRUE(mccl->getUuid().empty());
}

TEST_F(TorchCommMCCLTest, ReconfigureWithNullCommThrows) {
  // Default constructor leaves mccl_comm_ as nullptr
  auto mccl = std::make_shared<TorchCommMCCL>();

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(42);

  EXPECT_THROW(mccl->reconfigure(opts), std::runtime_error);
  EXPECT_FALSE(mccl->isInitialized());
  EXPECT_TRUE(mccl->getUuid().empty());
}

TEST_F(TorchCommMCCLTest, ReconfigureFailureWorkWaitDoesNotThrow) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  // cuda_api_ is intentionally NOT set: the failure path through
  // reconfigure() -> createWork() -> waitBlocking() -> waitCPU() never
  // accesses cuda_api_ (TracingGuard, mocks, and cached values suffice).
  // This also avoids needing a friend declaration for private member access.

  auto mock_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  // waitCpu() called twice: once inside reconfigure(), once in waitBlocking()
  EXPECT_CALL(*mock_work, waitCpu()).Times(2);
  EXPECT_CALL(*mock_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{
              .code = commInternalError, .message = "test failure"})));

  EXPECT_CALL(*mock_comm_ptr_, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(mock_work))));

  torch::comms::ReconfigureOptions opts = makeDefaultOpts(42);
  auto work = mccl->reconfigure(opts);

  // The key assertion: waitBlocking() (which calls waitCPU() with
  // TracingGuard) must not throw even though initState_ is UNINITIALIZED.
  EXPECT_NO_THROW(work->waitBlocking());

  // Work should report ERROR since reconfigure failed
  EXPECT_FALSE(work->isCompleted());
}

// ---------------------------------------------------------------------------
// Persistent AllGather (all_gather_p_init / exec / free)
// ---------------------------------------------------------------------------

// The init opts builder forwards recvbuff/maxRecvCount/dataType/stream and the
// timeout + kvPairs (forwarded for API symmetry; MCCL ignores them at init).
TEST_F(TorchCommMCCLTest, GetAllGatherPInitOptsForwardsFields) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  float data[8] = {};
  auto output = makeFakeCudaTensor(data, 8);
  auto* sentinelStream = reinterpret_cast<cudaStream_t>(0x1234);
  constexpr std::chrono::milliseconds kTimeout{4321};

  AllGatherPInitOptions options;
  options.timeout = kTimeout;
  options.hints["backend"] = "ctran";

  auto opts = mccl->getAllGatherPInitOpts(output, 0, sentinelStream, options);

  EXPECT_EQ(opts.recvbuff, output.data_ptr());
  EXPECT_EQ(opts.maxRecvCount, static_cast<size_t>(output.numel()));
  EXPECT_EQ(opts.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(opts.stream, sentinelStream);
  ASSERT_TRUE(opts.timeout.has_value());
  EXPECT_EQ(*opts.timeout, kTimeout);
  EXPECT_EQ(opts.kvPairs, options.hints);
}

// A default (unset) timeout maps to std::nullopt, matching other collectives.
TEST_F(TorchCommMCCLTest, GetAllGatherPInitOptsDefaultTimeoutUnset) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  float data[8] = {};
  auto output = makeFakeCudaTensor(data, 8);

  auto opts =
      mccl->getAllGatherPInitOpts(output, 0, nullptr, AllGatherPInitOptions{});

  EXPECT_FALSE(opts.timeout.has_value());
}

// Non-CUDA and empty receive tensors are rejected before any MCCL call.
TEST_F(TorchCommMCCLTest, GetAllGatherPInitOptsRejectsInvalidTensor) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  AllGatherPInitOptions options;

  auto cpuTensor = at::ones({8}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_THROW(
      mccl->getAllGatherPInitOpts(cpuTensor, 0, nullptr, options), c10::Error);

  float dummy = 0.0f;
  auto emptyTensor = makeFakeCudaTensor(&dummy, 0);
  EXPECT_THROW(
      mccl->getAllGatherPInitOpts(emptyTensor, 0, nullptr, options),
      c10::Error);
}

// The exec opts builder forwards sendbuff/count/dataType.
TEST_F(TorchCommMCCLTest, GetAllGatherPExecOptsForwardsFields) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  float data[4] = {};
  auto input = makeFakeCudaTensor(data, 4);

  auto opts = mccl->getAllGatherPExecOpts(input, 0, AllGatherPExecOptions{});

  EXPECT_EQ(opts.sendbuff, input.data_ptr());
  EXPECT_EQ(opts.count, static_cast<size_t>(input.numel()));
  EXPECT_EQ(opts.dataType, commDataType_t::commFloat32);
}

// Non-CUDA and empty send tensors are rejected before any MCCL call.
TEST_F(TorchCommMCCLTest, GetAllGatherPExecOptsRejectsInvalidTensor) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);

  auto cpuTensor = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_THROW(
      mccl->getAllGatherPExecOpts(cpuTensor, 0, AllGatherPExecOptions{}),
      c10::Error);

  float dummy = 0.0f;
  auto emptyTensor = makeFakeCudaTensor(&dummy, 0);
  EXPECT_THROW(
      mccl->getAllGatherPExecOpts(emptyTensor, 0, AllGatherPExecOptions{}),
      c10::Error);
}

// init auto-registers the (unregistered) recv buffer, pins the op to the
// internal stream, and returns the ctran handle on success.
TEST_F(TorchCommMCCLTest, AllGatherPInitAutoRegistersAndReturnsHandle) {
  auto cuda_api = std::make_shared<NiceMock<MockCudaApi>>();
  auto mccl =
      std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2, cuda_api);
  mccl->initState_ = TorchCommMCCL::InitializationState::INITIALIZED;
  auto* sentinelStream = reinterpret_cast<cudaStream_t>(0x1000);
  mccl->internalStream_ = sentinelStream;

  float data[8] = {};
  auto output = makeFakeCudaTensor(data, 8);
  auto* fakeHandle = reinterpret_cast<void*>(0x7000);
  auto* fakeRegHandle = reinterpret_cast<void*>(0x6000);

  EXPECT_CALL(*mock_comm_ptr_, commRegister(output.data_ptr(), _, _))
      .WillOnce(
          ::testing::DoAll(
              ::testing::SetArgPointee<2>(fakeRegHandle), Return(commSuccess)));

  auto init_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*init_work, waitCpu()).Times(1);
  EXPECT_CALL(*init_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(*mock_comm_ptr_, allGatherPInit(_, _))
      .WillOnce(
          [&](::mccl::AllGatherPHandle& handle,
              const ::mccl::AllGatherPInitOpts& opts)
              -> std::unique_ptr<::mccl::IWorkHandle> {
            EXPECT_EQ(opts.recvbuff, output.data_ptr());
            EXPECT_EQ(opts.maxRecvCount, static_cast<size_t>(output.numel()));
            EXPECT_EQ(opts.dataType, commDataType_t::commFloat32);
            EXPECT_EQ(opts.stream, sentinelStream);
            handle = fakeHandle;
            return std::move(init_work);
          });

  auto handle = mccl->all_gather_p_init(output);
  EXPECT_EQ(handle, fakeHandle);

  // Avoid touching the sentinel stream at teardown and suppress the
  // destructor's "not finalized" warning (finalize() would need real CUDA
  // state).
  mccl->internalStream_ = nullptr;
  mccl->initState_ = TorchCommMCCL::InitializationState::FINALIZED;
}

// A failed MCCL init result surfaces synchronously as an exception (the generic
// all_gather_p_init has no deferred-error channel).
TEST_F(TorchCommMCCLTest, AllGatherPInitThrowsOnFailure) {
  auto cuda_api = std::make_shared<NiceMock<MockCudaApi>>();
  auto mccl =
      std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2, cuda_api);
  mccl->initState_ = TorchCommMCCL::InitializationState::INITIALIZED;

  float data[8] = {};
  auto output = makeFakeCudaTensor(data, 8);

  EXPECT_CALL(*mock_comm_ptr_, commRegister(_, _, _))
      .WillOnce(
          ::testing::DoAll(
              ::testing::SetArgPointee<2>(reinterpret_cast<void*>(0x6000)),
              Return(commSuccess)));

  auto init_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*init_work, waitCpu()).Times(1);
  EXPECT_CALL(*init_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(
              mccl::Result{.code = commInternalError, .message = "boom"})));
  EXPECT_CALL(*mock_comm_ptr_, allGatherPInit(_, _))
      .WillOnce(Return(::testing::ByMove(std::move(init_work))));

  EXPECT_THROW(mccl->all_gather_p_init(output), std::runtime_error);

  mccl->initState_ = TorchCommMCCL::InitializationState::FINALIZED;
}

// exec inserts the current -> internalStream input-readiness event on every
// call and enqueues the work on the internal stream.
TEST_F(TorchCommMCCLTest, AllGatherPExecInsertsEventBridgeAndEnqueues) {
  auto cuda_api = std::make_shared<NiceMock<MockCudaApi>>();
  auto* currentStream = reinterpret_cast<cudaStream_t>(0x3000);
  auto* internalStream = reinterpret_cast<cudaStream_t>(0x1000);
  auto* depEvent = reinterpret_cast<cudaEvent_t>(0x2000);

  ON_CALL(*cuda_api, getCurrentCUDAStream(_))
      .WillByDefault(Return(currentStream));
  // enqueueWork queries graph-capture status for GPU work; report
  // not-capturing.
  ON_CALL(*cuda_api, streamIsCapturing(_, _))
      .WillByDefault(
          ::testing::DoAll(
              ::testing::SetArgPointee<1>(cudaStreamCaptureStatusNone),
              Return(cudaSuccess)));

  auto mccl =
      std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2, cuda_api);
  mccl->initState_ = TorchCommMCCL::InitializationState::INITIALIZED;
  mccl->internalStream_ = internalStream;
  mccl->dependencyEvent_ = depEvent;

  EXPECT_CALL(*cuda_api, eventRecord(depEvent, currentStream)).Times(1);
  EXPECT_CALL(*cuda_api, streamWaitEvent(internalStream, depEvent, 0)).Times(1);

  auto* fakeHandle = reinterpret_cast<void*>(0x7000);
  float data[4] = {};
  auto input = makeFakeCudaTensor(data, 4);

  auto exec_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*exec_work, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));
  EXPECT_CALL(*mock_comm_ptr_, allGatherPExec(fakeHandle, _))
      .WillOnce(
          [&](::mccl::AllGatherPHandle /* handle */,
              const ::mccl::AllGatherPExecOpts& opts)
              -> std::unique_ptr<::mccl::IWorkHandle> {
            EXPECT_EQ(opts.sendbuff, input.data_ptr());
            EXPECT_EQ(opts.count, static_cast<size_t>(input.numel()));
            EXPECT_EQ(opts.dataType, commDataType_t::commFloat32);
            return std::move(exec_work);
          });

  auto work = mccl->all_gather_p_exec(fakeHandle, input, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  // Drain the enqueued (completed) work so its back-reference to the comm is
  // released, then mark finalized to suppress the destructor warning.
  mccl->workq_.finalize();
  mccl->internalStream_ = nullptr;
  mccl->dependencyEvent_ = nullptr;
  mccl->initState_ = TorchCommMCCL::InitializationState::FINALIZED;
}

// free releases only the persistent handle; it never deregisters the recv
// buffer, and a null handle is a no-op.
TEST_F(TorchCommMCCLTest, AllGatherPFreeCallsFreeAndDoesNotDeregister) {
  auto mccl = std::make_shared<TorchCommMCCL>(std::move(mock_comm_), 0, 2);
  auto* fakeHandle = reinterpret_cast<void*>(0x7000);

  EXPECT_CALL(*mock_comm_ptr_, commDeregister(_)).Times(0);

  auto free_work = std::make_unique<::mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*free_work, waitCpu()).Times(1);
  EXPECT_CALL(*mock_comm_ptr_, allGatherPFree(fakeHandle))
      .WillOnce(Return(::testing::ByMove(std::move(free_work))));

  mccl->all_gather_p_free(fakeHandle);

  // Null handle: no additional allGatherPFree call.
  mccl->all_gather_p_free(nullptr);
}

} // namespace torch::comms::test
