#include <folly/Random.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/mccl/integration_tests/CollectiveIntegrationTestMixin.h"
#include "comms/mccl/integration_tests/McclIntegrationTestUtil.h"
#include "comms/mccl/tests/CudaStream.h"
#include "comms/mccl/tests/CudaTestUtil.h"
#include "comms/testinfra/IbverbMockTestUtils.h"
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

class CollTraceWatchdogTest : public mccl::CollectiveIntegrationTestMixin,
                              public ::testing::Test {
 public:
  void SetUp() override {
    int numRanks = 4;

    std::vector<std::string> envList = {
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
        "NCCL_CTRAN_BACKENDS=ib",
        "NCCL_ALLREDUCE_ALGO=ctran",
        fmt::format(
            "NCCL_DEBUG_FILE={}", (tmpDir_.path() / "logfile%p").string()),
    };

    // Set up the environment variables for IbVerbs mock.
    auto hookIbVerbs = getenv("NCCL_IBVERBS_PATH");
    if (hookIbVerbs != nullptr || strlen(hookIbVerbs) != 0) {
      envList.emplace_back(fmt::format("NCCL_IBVERBS_PATH={}", hookIbVerbs));
    }

    mccl::CollectiveIntegrationTestMixin::SetUp(
        mccl::CollectiveIntegrationTestMixin::Config{
            .numRanks = numRanks,
            .shouldExitOnFailure = false,
            .env = envList});
  }

  folly::test::TemporaryDirectory tmpDir_{
      fmt::format("CollTraceWatchdog{}", folly::Random::rand64())};

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

TEST_F(CollTraceWatchdogTest, TestAsyncErrorWithIbVerbMock) {
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

  // Allocate memory on the GPU
  constexpr int size = 32;
  facebook::ftar::DynMemGpuBuffer sendBuff(size * sizeof(float));
  facebook::ftar::DynMemGpuBuffer recvBuff(size * sizeof(float));

  // Inject ibv failure to trigger async error
  ::meta::comms::setFailureInjection("ibv_post_send", 0, rank);

  NCCLCHECK_FATAL(ncclAllReduce(
      (const void*)sendBuff.raw(),
      (void*)recvBuff.raw(),
      size,
      ncclFloat,
      ncclSum,
      comm.raw(),
      stream.raw()));

  CUDACHECK_FATAL(cudaStreamSynchronize(stream.raw()));
}
