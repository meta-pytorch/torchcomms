#include <folly/Random.h>
#include <folly/stop_watch.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/mccl/integration_tests/CollectiveIntegrationTestMixin.h"
#include "comms/mccl/integration_tests/McclIntegrationTestUtil.h"
#include "comms/mccl/tests/CudaStream.h"
#include "comms/mccl/tests/CudaTestUtil.h"
#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.h"
#include "ftar/DynMemGpuBuffer.h"

#define NCCLCHECK_FATAL(cmd)                                            \
  do {                                                                  \
    ncclResult_t res = cmd;                                             \
    if (res != ncclSuccess) {                                           \
      XLOGF(FATAL, "NCCL error {} '{}'", res, ncclGetErrorString(res)); \
    }                                                                   \
  } while (0)

#define CUDACHECK_FATAL(cmd)                                            \
  do {                                                                  \
    cudaError_t err = cmd;                                              \
    if (err != cudaSuccess) {                                           \
      XLOGF(FATAL, "CUDA error {} '{}'", err, cudaGetErrorString(err)); \
    }                                                                   \
  } while (0)

// RAII for NCCL Communicator
class NcclComm {
 public:
  NcclComm(int worldSize, int globalRank) {
    const std::string uniqueIDKey{"uniqueID"};
    ncclUniqueId ncclUniqueID;
    // NCCL unique ID must be generated on one node and shared to all
    // others
    if (globalRank == 0) {
      // Rank 0 creates the unique id
      NCCLCHECK_FATAL(ncclGetUniqueId(&ncclUniqueID));
      mccl::McclIntegrationTestUtil::setKey(
          uniqueIDKey,
          std::string(ncclUniqueID.internal, NCCL_UNIQUE_ID_BYTES),
          std::nullopt);
    } else {
      // Everyone else waits for it
      auto value = mccl::McclIntegrationTestUtil::waitForKey(
          uniqueIDKey, [](const auto& versionAndValue) {
            return versionAndValue.has_value();
          });
      std::memcpy(
          ncclUniqueID.internal, value.value.data(), NCCL_UNIQUE_ID_BYTES);
    }
    NCCLCHECK_FATAL(
        ncclCommInitRank(&comm_, worldSize, ncclUniqueID, globalRank));
  }

  ~NcclComm() {
    NCCLCHECK_FATAL(ncclCommDestroy(comm_));
  }

  // Movable, not copyable
  NcclComm(const NcclComm&) = delete;
  NcclComm& operator=(const NcclComm&) = delete;
  NcclComm(NcclComm&& other) noexcept = default;
  NcclComm& operator=(NcclComm&& other) = default;

  ncclComm_t raw() const {
    return comm_;
  }

 private:
  ncclComm_t comm_{};
};

void waitStreamWithTimeout(
    cudaStream_t stream,
    std::chrono::milliseconds timeout) {
  folly::stop_watch<> timer;
  while (!timer.elapsed(timeout)) {
    auto res = cudaStreamQuery(stream);
    if (res == cudaSuccess) {
      return;
    }
    if (res != cudaErrorNotReady) {
      XLOGF(
          FATAL, "Unexpected CUDA error {} '{}'", res, cudaGetErrorString(res));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  }
  FAIL() << "Wait Stream did not complete within timeout";
}

class CollTraceWatchdogTest : public mccl::CollectiveIntegrationTestMixin,
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
                    // needed for bootstrapping
                    "NCCL_SOCKET_IFNAME=eth0",
                    "NCCL_CLIENT_SOCKET_IFNAME=eth0",
                    "NCCL_FASTINIT_MODE=none",
                    "NCCL_SOCKET_IPADDR_PREFIX=",
                    // enable commsDumpAll
                    "NCCL_COMMSMONITOR_ENABLE=1",
                    "NCCL_COLLTRACE=trace",
                    "NCCL_COLLTRACE_USE_NEW_COLLTRACE=1",
                    // enable ctran
                    "NCCL_CTRAN_ENABLE=1",
                    "NCCL_CTRAN_REGISTRATION_SIZE_CHECK=1",
                    "NCCL_CTRAN_BACKENDS=ib,socket,nvl",
                    // enable logging
                    "NCCL_DEBUG=INFO",
                    fmt::format(
                        "NCCL_DEBUG_FILE={}",
                        (tmpDir_.path() / "logfile%p").string()),
                },
        });
  }

  folly::test::TemporaryDirectory tmpDir_{
      fmt::format("CollTraceWatchdog{}", folly::Random::rand64())};

  void testDriverCheckSucceed() {
    ASSERT_TRUE(
        std::holds_alternative<
            mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_));
    auto& testDriverState =
        std::get<mccl::CollectiveIntegrationTestMixin::TestDriverState>(
            this->state_);
    // Check that all ranks exited with non-zero exit code
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
    // Check that all ranks exited with non-zero exit code
    EXPECT_THAT(
        testDriverState.workerExitCodes, ::testing::Each(::testing::Ne(0)));

    int count = 0;
    for (const auto& entry : folly::fs::directory_iterator(tmpDir_.path())) {
      std::string fileContents;
      ASSERT_TRUE(folly::readFile(entry.path().c_str(), fileContents));
      EXPECT_THAT(fileContents, ::testing::HasSubstr("COMM FATAL"));
      count++;
    }
    EXPECT_EQ(count, 4);
  }
};

class NcclAllReduce {
 public:
  NcclAllReduce(ncclComm_t comm, mccl::cuda::CudaStream& stream, int64_t size)
      : sendBuff_(size * sizeof(float)), recvBuff_(size * sizeof(float)) {
    NCCLCHECK_FATAL(ncclAllReduce(
        (const void*)sendBuff_.raw(),
        (void*)recvBuff_.raw(),
        size,
        ncclFloat,
        ncclSum,
        comm,
        stream.raw()));
  }

 private:
  facebook::ftar::DynMemGpuBuffer sendBuff_;
  facebook::ftar::DynMemGpuBuffer recvBuff_;
};

TEST_F(CollTraceWatchdogTest, TestAsyncErrorFromGPE) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnAsyncError", folly::to<std::string>(true)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;
  mccl::cuda::CudaStream stream;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  // Allocate memory on the CPU. A buffer size smaller than 4097 shall trigger
  // error from Ctran when NCCL_CTRAN_REGISTRATION_SIZE_CHECK=1. This is just a
  // way to trigger any async error from GPE. Once the behavior of GPE is
  // changed, we can simply disable this test.
  constexpr int size = 32;
  float sendBuff[size * sizeof(float)];

  ncclx::Hints hints;
  hints.set("window_buffer_location", "cpu");
  void* basePtr;
  ncclWindow_t win;
  NCCLCHECK_FATAL(ncclWinAllocate(
      size * sizeof(float) * worldSize, comm.raw(), &basePtr, &win, hints));

  auto srcRank = (rank - 1 + worldSize) % worldSize;
  auto dstRank = (rank + 1) % worldSize;

  NCCLCHECK_FATAL(
      ncclPutSignal(sendBuff, 32, ncclFloat, dstRank, 0, win, stream.raw()));
  NCCLCHECK_FATAL(ncclWaitSignal(srcRank, win, stream.raw()));
  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{80});
}

TEST_F(CollTraceWatchdogTest, TestAsyncErrorWithGenericAsyncError) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnAsyncError", folly::to<std::string>(true)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);
  mccl::cuda::CudaStream stream;

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  NcclAllReduce allReduce(comm.raw(), stream, 32);

  ncclCommSetAsyncError(comm.raw(), ncclInternalError);

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});

  // Wait for sufficiently long for watchdog to stop waiting and exit
  sleep(70);
}

TEST_F(CollTraceWatchdogTest, TestTimeoutBeforeColl) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  constexpr auto timeoutSec{std::chrono::seconds{5}};
  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnTimeout", folly::to<std::string>(true)));

  // Set 5 seconds collective timeout
  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.timeoutMs",
          folly::to<std::string>(timeoutSec.count() * 1000)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);
  mccl::cuda::CudaStream stream;

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  // Need a have an allReduce here to trigger pre-connect
  NcclAllReduce initAllReduce(comm.raw(), stream, 32);

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});

  meta::comms::colltrace::CPUControlledKernel kernel(stream.raw());
  // Insert a kernel before the collective kernel in the stream;
  kernel.launch();

  NcclAllReduce allReduce(comm.raw(), stream, 32);

  std::this_thread::sleep_for(timeoutSec + std::chrono::seconds{3});

  // Failing case, we should not reach here. Release resources so the process
  // and exit normally.
  kernel.endKernel();

  FAIL() << "Watchdog did not trigger after timeout is reached";
}

TEST_F(CollTraceWatchdogTest, TestTimeoutInColl) {
  if (isTestDriverProcess()) {
    testDriverCheckCrashedWithWatchdog();
    return;
  }

  constexpr auto timeoutSec{std::chrono::seconds{5}};
  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnTimeout", folly::to<std::string>(true)));

  // Set 5 seconds collective timeout
  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.timeoutMs",
          folly::to<std::string>(timeoutSec.count() * 1000)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);
  mccl::cuda::CudaStream stream;

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  // Need a have an allReduce here to trigger pre-connect
  NcclAllReduce initAllReduce(comm.raw(), stream, 32);

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});

  if (rank != 0) {
    // Sleep long enough to make other ranks timeout
    std::this_thread::sleep_for(timeoutSec + std::chrono::seconds{3});
    // Hacky way to make the driver process believe rank 0 is faulty as well
    XLOG(FATAL, "COMM FATAL");
  } else {
    NcclAllReduce allReduce(comm.raw(), stream, 32);
    std::this_thread::sleep_for(timeoutSec + std::chrono::seconds{3});
  }

  FAIL() << "Watchdog did not trigger after timeout is reached";
}

TEST_F(CollTraceWatchdogTest, TestBelowTimeoutInColl) {
  if (isTestDriverProcess()) {
    testDriverCheckSucceed();
    return;
  }

  constexpr auto timeoutSec{std::chrono::seconds{60}};
  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnTimeout", folly::to<std::string>(true)));

  // Set 5 seconds collective timeout
  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.timeoutMs",
          folly::to<std::string>(timeoutSec.count() * 1000)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);
  mccl::cuda::CudaStream stream;

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  // Need a have an allReduce here to trigger pre-connect
  NcclAllReduce initAllReduce(comm.raw(), stream, 32);

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});

  if (rank != 0) {
    // Sleep shortly, but longer than how long we slept for small timeout case
    std::this_thread::sleep_for(std::chrono::seconds{10});
    NcclAllReduce allReduce(comm.raw(), stream, 32);
  } else {
    NcclAllReduce allReduce(comm.raw(), stream, 32);
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});
}

TEST_F(CollTraceWatchdogTest, TestBelowTimeoutBeforeColl) {
  if (isTestDriverProcess()) {
    testDriverCheckSucceed();
    return;
  }

  constexpr auto timeoutSec{std::chrono::seconds{60}};
  int rank = getRank();
  int worldSize = getWorldSize();

  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.crashOnTimeout", folly::to<std::string>(true)));

  // Set 5 seconds collective timeout
  NCCLCHECK_FATAL(
      ncclx::setGlobalHint(
          "ncclx.colltrace.timeoutMs",
          folly::to<std::string>(timeoutSec.count() * 1000)));

  // Initialize CUDA state
  auto deviceId = mccl::CudaTestUtil::getCudaDeviceId(rank);
  XLOG(INFO) << "CUDA device id: " << deviceId;

  // Initialize NCCL communicator
  NcclComm comm(worldSize, rank);
  mccl::cuda::CudaStream stream;

  // Ensure we are using new colltrace
  ASSERT_EQ(comm.raw()->ctranComm_->collTrace_, nullptr);
  ASSERT_NE(comm.raw()->newCollTrace, nullptr);

  // Need a have an allReduce here to trigger pre-connect
  NcclAllReduce initAllReduce(comm.raw(), stream, 32);

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});

  meta::comms::colltrace::CPUControlledKernel kernel(stream.raw());
  // Insert a kernel before the collective kernel in the stream;
  kernel.launch();

  NcclAllReduce allReduce(comm.raw(), stream, 32);

  std::this_thread::sleep_for(std::chrono::seconds{10});
  kernel.endKernel();

  waitStreamWithTimeout(stream.raw(), std::chrono::seconds{20});
}
