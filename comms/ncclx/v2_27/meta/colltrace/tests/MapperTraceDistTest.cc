// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <folly/synchronization/Baton.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"

#include "comm.h"
#define CAPTURE_STDOUT_WITH_FAIL_SAFE()                                    \
  testing::internal::CaptureStdout();                                      \
  SCOPE_FAIL {                                                             \
    std::string output = testing::internal::GetCapturedStdout();           \
    std::cout << "Test failed with stdout being: " << output << std::endl; \
  };

class MapperTraceTest : public NcclxBaseTest {
 public:
  MapperTraceTest() = default;
  void SetUp() override {
    // Allow user to change via command line
    setenv("NCCL_DEBUG_SUBSYS", "INIT,COLL", 0);

    // Set up dummy values for environment variables for Scuba test
    setenv("WORLD_SIZE", "4", 0);
    setenv("HPC_JOB_NAME", "CollTraceUT", 0);
    setenv("HPC_JOB_VERSION", "1", 0);
    setenv("HPC_JOB_ATTEMPT_INDEX", "2", 0);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        0);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);

    NcclxBaseTest::SetUp();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void prepareCtranAllToAll(ncclComm* comm, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, sendBuf, count * this->numRanks * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
  cudaStream_t stream;
};

TEST_F(MapperTraceTest, CtranAllToAll) {
  auto ctranGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto ctranOnGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  auto backend = comm->ctranComm_->ctran_->mapper->getBackend(
      (comm->ctranComm_->statex_->rank() + 1) %
      comm->ctranComm_->statex_->nRanks());
  std::string backendStr = "Unknown";
  if (backend == CtranMapperBackend::IB) {
    backendStr = "IB";
  } else if (backend == CtranMapperBackend::NVL) {
    backendStr = "NVL";
  }
  printf(
      "nNodes stateX: %d nNodes baseline: %d Backend: %s\n",
      comm->ctranComm_->statex_->nNodes(),
      comm->nNodes,
      backendStr.c_str());

  if (!ctranAllToAllSupport(
          count, commInt8, comm->ctranComm_.get(), NCCL_ALLTOALL_ALGO) ||
      comm->ctranComm_->statex_->nNodes() < 2) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran All to All support or it is not a multi-node comm";
  }

  prepareCtranAllToAll(comm, count);
  auto guard = folly::makeGuard([&]() {
    if (sendHandle != nullptr) {
      ncclCommDeregister(comm, sendHandle);
    }
    if (recvHandle != nullptr) {
      ncclCommDeregister(comm, recvHandle);
    }
  });

  folly::Baton dumpBaton;
  ncclx::colltrace::MapperTrace::Dump dump;
  auto trace = ncclx::colltrace::getMapperTrace(comm->ctranComm_.get());
  trace->registerBeforeCollEndCallback([&]() {
    static int64_t counter = 0;
    if (counter == nColl - 1) {
      dump = trace->dump();
      dumpBaton.post();
    }
    counter++;
  });
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  ASSERT_EQ(dumpBaton.try_wait_for(std::chrono::seconds(1)), true);

  std::unordered_map<std::string, std::string> map;
  if (dump.currentColl != nullptr) {
    map["MT_currentColl"] = folly::toJson(dump.currentColl->toDynamic());
  } else {
    map["MT_currentColl"] = "null";
  }
  map["MT_unfinishedRequests"] = serializeObjects(dump.unfinishedRequests);
  map["MT_recvNotifiedByPeer"] = mapToJson(dump.recvNotifiedByPeer);
  map["MT_putFinishedByPeer"] = mapToJson(dump.putFinishedByPeer);
  if (comm->rank == 0) {
    for (auto& it : map) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  // EXPECT_TRUE(dump.currentColl != std::nullopt);
  EXPECT_GT(dump.putFinishedByPeer.size(), 0);
  EXPECT_GT(dump.recvNotifiedByPeer.size(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
