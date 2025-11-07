// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unistd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/testing/TestUtil.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/memory/Utils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/tests/MockScubaTable.h"
#include "meta/colltrace/CollTrace.h"

#include "LoggerUtil.h"
#include "comm.h" // @manual
#include "debug.h" // @manual
#include "nccl.h" // @manual

class MemoryLoggingTestFixture : public NcclxBaseTestFixture {
 public:
  MemoryLoggingTestFixture() = default;
  void SetUp() override {
    ctran::utils::commCudaLibraryInit();
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_MEMORY_EVENT_LOGGING", "pipe:nccl_memory_logging", 1);
    NcclxBaseTestFixture::SetUp();
    setenv("RANK", std::to_string(this->globalRank).c_str(), 1);
    LOG(INFO) << "Rank " << this->globalRank << " localRank "
              << this->localRank;
    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    // Reset filter to ensure filter setup won't affect other tests
    MemoryEvent::resetFilter();
    if (sendBuf) {
      CUDACHECK_TEST(cudaFree(sendBuf));
      sendBuf = nullptr;
    }
    if (recvBuf) {
      CUDACHECK_TEST(cudaFree(recvBuf));
      recvBuf = nullptr;
    }
    NcclxBaseTestFixture::TearDown();
  }

  std::string initLogger() {
    folly::Singleton<const DataTableAllTables, DataTableAllTablesTag>::
        make_mock([this]() {
          return new DataTableAllTables(createAllMockTables(mockPassthru));
        });
    // force singleton init
    folly::Singleton<const DataTableAllTables, DataTableAllTablesTag>::
        try_get();
    initNcclLogger();
    auto logFileName = getMemoryEventScubaFile();
    std::cout << "Rank " << this->globalRank
              << " reading from memory logging file " << logFileName
              << std::endl;
    return logFileName;
  }

  void prepBuffers(size_t sendBytes, size_t recvBytes) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, sendBytes));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, recvBytes));
    CUDACHECK_TEST(cudaMemset(sendBuf, 0, sendBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvBytes));
    // Ensure value has been set before colletive runs on nonblocking stream
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  // Generate set of unqiue buffer keys based on communicator channel/peer
  // information
  std::vector<std::string> parseCallsites(ncclComm* comm, bool p2pOnly) {
    std::vector<std::string> expectedBufKeys;
    for (int c = 0; c < comm->nChannels; c++) {
      auto& channel = comm->channels[c];
      if (channel.id != -1) {
        EXPECT_EQ(channel.id, c);
        struct ncclChannelPeer** channelPeers = comm->channels[c].peers;
        int peerRank = 0;
        while (channelPeers[peerRank] != nullptr) {
          parseChannelPeer(
              channelPeers[peerRank],
              c /* channelId*/,
              0 /* connIndex */,
              true /* isSend*/,
              peerRank,
              comm->ctranComm_->statex_->isSameNode(this->globalRank, peerRank),
              comm->commHash,
              p2pOnly,
              expectedBufKeys);
          parseChannelPeer(
              channelPeers[peerRank],
              c /* channelId*/,
              0 /* connIndex */,
              false /* isSend*/,
              peerRank,
              comm->ctranComm_->statex_->isSameNode(this->globalRank, peerRank),
              comm->commHash,
              p2pOnly,
              expectedBufKeys);
          parseChannelPeer(
              channelPeers[peerRank],
              c /* channelId*/,
              1 /* connIndex */,
              true /* isSend*/,
              peerRank,
              comm->ctranComm_->statex_->isSameNode(this->globalRank, peerRank),
              comm->commHash,
              p2pOnly,
              expectedBufKeys);
          parseChannelPeer(
              channelPeers[peerRank],
              c /* channelId*/,
              1 /* connIndex */,
              false /* isSend*/,
              peerRank,
              comm->ctranComm_->statex_->isSameNode(this->globalRank, peerRank),
              comm->commHash,
              p2pOnly,
              expectedBufKeys);
          peerRank++;
        }
      }
    }
    return expectedBufKeys;
  }

  void parseChannelPeer(
      struct ncclChannelPeer* channelPeer,
      int channelId,
      int connIndex,
      bool isSend,
      int peerRank,
      bool isSameNode,
      uint64_t commHash,
      bool p2pOnly,
      std::vector<std::string>& expectedCallsites) {
    struct ncclConnector* connector =
        isSend ? &channelPeer->send[connIndex] : &channelPeer->recv[connIndex];
    if (!connector->connected) {
      return;
    }
    bool shared = connector->conn.shared;
    int tpLocalRank = connector->proxyConn.tpLocalRank;
    std::string setupMethod = isSameNode ? "ProxySetup" : "ProxyConnect";
    bool isP2pWrite = isSend && (connector->conn.flags & NCCL_P2P_WRITE);
    // Only Net p2p buffers are shared currently
    if (shared) {
      if (!p2pOnly) {
        expectedCallsites.push_back(
            fmt::format(
                "sharedNetBuffersInit:{}/{}/{}",
                commHash,
                tpLocalRank,
                isSend ? 0 : 1));
      }
    } else if ((isSameNode || !p2pOnly) && !isP2pWrite) {
      expectedCallsites.push_back(
          ncclx::memory::genKey(
              setupMethod, isSameNode, isSend, channelId, connIndex, peerRank));
    }
  }

  void verifyEventCallsites(
      const std::string& output,
      uint64_t commHash,
      const std::vector<std::string>& expectedBufKeys) {
    std::vector<int> expEventCallsiteCount(expectedBufKeys.size(), 0);

    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
      folly::dynamic jsonLog = folly::parseJson(line);
      if (jsonLog["int"]["commHash"].asInt() != commHash) {
        continue;
      }
      EXPECT_EQ(jsonLog["int"]["globalRank"].asInt(), this->globalRank);
      auto callsite = jsonLog["normal"]["callsite"].asString();
      for (int i = 0; i < expectedBufKeys.size(); i++) {
        if (callsite == expectedBufKeys[i]) {
          expEventCallsiteCount[i]++;
        }
      }
    }
    for (int i = 0; i < expEventCallsiteCount.size(); i++) {
      EXPECT_EQ(expEventCallsiteCount[i], 1)
          << " use " << expectedBufKeys[i] << " on globalRank "
          << this->globalRank << " occurred " << expEventCallsiteCount[i]
          << " times, should be exactly once" << std::endl;
    }
  }

  void verifyEventsUse(
      const std::string& output,
      const std::vector<std::string>& expEventUse) {
    std::vector<int> expEventUseCount(expEventUse.size(), 0);

    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
      folly::dynamic jsonLog = folly::parseJson(line);
      EXPECT_EQ(jsonLog["int"]["globalRank"].asInt(), this->globalRank);
      auto use = jsonLog["normal"]["use"].asString();
      for (int i = 0; i < expEventUse.size(); i++) {
        if (use.find(expEventUse[i]) != std::string::npos) {
          expEventUseCount[i]++;
        }
      }
    }
    for (int i = 0; i < expEventUse.size(); i++) {
      EXPECT_GE(expEventUseCount[i], 1)
          << " use " << expEventUse[i] << " on globalRank " << this->globalRank
          << " not found" << std::endl;
    }
  }

 protected:
  cudaStream_t stream;
  void* sendBuf{nullptr};
  void* recvBuf{nullptr};
  bool mockPassthru{true};
};

TEST_P(MemoryLoggingTestFixture, ncclInternalBufferLogTest) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto memoryEventFilterGuard = EnvRAII(NCCL_FILTER_MEM_LOGGING_BY_RANKS, {});
  auto memCacheFilterGuard = EnvRAII(NCCL_USE_MEM_CACHE, false);
  auto logFileName = initLogger();

  // First comm creation as well as first kernel launch has some extra memory
  // usage (https://fburl.com/code/rxjvjads), use second comm
  // creation/collective for testing
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, nullptr));
  std::cout << "Rank " << this->globalRank << " finished init, run AR"
            << std::endl;
  size_t count = 1 << 10; // 1K elements
  prepBuffers(count * ncclTypeSize(ncclInt), count * ncclTypeSize(ncclInt));
  EXPECT_EQ(
      ncclAllReduce(
          sendBuf, recvBuf, 1 * numRanks, ncclInt, ncclSum, comm, stream),
      ncclSuccess);
  // run alltoall logic to trigger p2p buffer allocation
  NCCLCHECK_TEST(ncclGroupStart());
  for (int r = 0; r < this->numRanks; ++r) {
    EXPECT_EQ(ncclSend(sendBuf, count, ncclInt, r, comm, stream), ncclSuccess);
    EXPECT_EQ(ncclRecv(recvBuf, count, ncclInt, r, comm, stream), ncclSuccess);
  }
  NCCLCHECK_TEST(ncclGroupEnd());
  std::cout << "Rank " << this->globalRank
            << " finished AllReduce and AllToAll on world communicator "
            << std::endl;

  size_t before_free, before_total;
  CUDACHECK_TEST(cudaMemGetInfo(&before_free, &before_total));
  ncclComm_t testComm = nullptr;
  NCCLCHECK_TEST(ncclCommSplit(comm, 0, globalRank, &testComm, nullptr));

  // run AllReduce logic to trigger collective buffer allocation
  EXPECT_EQ(
      ncclAllReduce(
          sendBuf,
          recvBuf,
          count * numRanks,
          ncclInt,
          ncclSum,
          testComm,
          stream),
      ncclSuccess);

  // run alltoall logic to trigger p2p buffer allocation
  NCCLCHECK_TEST(ncclGroupStart());
  for (int r = 0; r < this->numRanks; ++r) {
    EXPECT_EQ(
        ncclSend(sendBuf, count, ncclInt, r, testComm, stream), ncclSuccess);
    EXPECT_EQ(
        ncclRecv(recvBuf, count, ncclInt, r, testComm, stream), ncclSuccess);
  }
  NCCLCHECK_TEST(ncclGroupEnd());
  size_t after_free, after_total;
  CUDACHECK_TEST(cudaMemGetInfo(&after_free, &after_total));
  size_t groundTruthUsage = before_free - after_free;
  // wait until memory logging completes
  NcclLogger::close();
  auto output = readFromFile(logFileName);
  EXPECT_NE(output, "");

  std::istringstream iss(output);
  folly::dynamic jsonLog;
  std::string line;
  size_t totalBytesLogged = 0;
  while (std::getline(iss, line)) {
    jsonLog = folly::parseJson(line);
    if (jsonLog["int"]["commHash"].asInt() == testComm->commHash &&
        jsonLog["normal"]["callsite"].asString() != "") {
      totalBytesLogged += jsonLog["int"]["bytes"].asInt();
    }
  }
  constexpr size_t kExpectMemDeltaBytes = 5 * 1024 * 1024;
  EXPECT_FALSE(totalBytesLogged == 0);
  EXPECT_GE(groundTruthUsage, totalBytesLogged);
  EXPECT_TRUE(
      (groundTruthUsage - totalBytesLogged) <=
      kExpectMemDeltaBytes); // expect logged usage to be with in 5MB of
                             // ground truth

  // Only run this test on GPUs newer than H100
  auto expectedCallsites = parseCallsites(comm, false /* p2pOnly*/);
  if (comm->compCap > 80) {
    // Verify staging buffer allocation logs; each callsite is expected to be
    // logged exactly once
    verifyEventCallsites(
        output,
        comm->commHash,
        std::vector<std::string>(
            expectedCallsites.begin(), expectedCallsites.end()));
  }
  NCCLCHECK_TEST(ncclCommDestroy(testComm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(MemoryLoggingTestFixture, userBufferLoggingTest) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto memoryEventFilterGuard =
      EnvRAII(NCCL_FILTER_MEM_LOGGING_BY_RANKS, {"0", "1"});
  auto memoryRegEventFilterGuard =
      EnvRAII(NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS, {"2"});
  auto ncclCommRegisterEnableGuard =
      EnvRAII(NCCL_COMM_REGISTER_LOG_ENABLE, true);

  auto logFileName = initLogger();
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, nullptr));

  /* mapper registration logic */
  void *buf = nullptr, *segHdl = nullptr;
  constexpr size_t kBufferSize = 1024 * 1024;
  CUDACHECK_TEST(cudaMalloc(&buf, kBufferSize));
  COMMCHECK_TEST(comm->ctranComm_->ctran_->mapper->regMem(
      buf, kBufferSize, &segHdl, true /* forceRegist */));

  COMMCHECK_TEST(comm->ctranComm_->ctran_->mapper->deregMem(segHdl));
  CUDACHECK_TEST(cudaFree(buf));

  void *userBuf = nullptr, *userSegHdl = nullptr;
  /* user buffer allocation/registration logic */
  NCCLCHECK_TEST(ncclMemAlloc(&userBuf, kBufferSize));
  NCCLCHECK_TEST(ncclCommRegister(comm, userBuf, kBufferSize, &userSegHdl));
  NCCLCHECK_TEST(ncclCommDeregister(comm, userSegHdl));
  NCCLCHECK_TEST(ncclMemFree(userBuf));

  // wait until memory logging completes
  NcclLogger::close();
  auto output = readFromFile(logFileName);

  // Expect ranks 0-1 to have memory allocation events logged
  if (this->globalRank <= 1) {
    EXPECT_NE(output, "");
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
      folly::dynamic jsonLog = folly::parseJson(line);
      EXPECT_EQ(jsonLog["int"]["globalRank"].asInt(), this->globalRank);
    }
  }

  // Additionally, expect ranks 0-1 to have user buffer allocation events
  // logged
  if (this->globalRank <= 1) {
    verifyEventsUse(output, {"ncclMemAlloc", "ncclCuMemFree"});
  }
  // expect rank 2 to have registration events logged
  else if (this->globalRank == 2) {
    verifyEventsUse(
        output,
        {"eagerRegMem", "deregMem", "ncclCommRegister", "ncclCommDeregister"});
  } else {
    // Expect other ranks to have no events logged
    EXPECT_EQ(output, "");
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

INSTANTIATE_TEST_SUITE_P(
    MyTestSuite,
    MemoryLoggingTestFixture,
    testing::Values(
        // Baseline
        NcclxEnvs({{"NCCL_USE_MEM_CACHE", "0"}}),
        // MemOpt + lazy setup channels
        NcclxEnvs(
            {{"NCCL_USE_MEM_CACHE", "1"}, {"NCCL_LAZY_SETUP_CHANNELS", "1"}})),
    [](const testing::TestParamInfo<MemoryLoggingTestFixture::ParamType>&
           info) {
      // generate test-name for a given NcclxEnvs
      std::string name;
      for (const auto& [key, val] : info.param) {
        if (key == "NCCL_USE_MEM_CACHE") {
          name += (val == "1") ? "memOpt" : "Baseline";
        }
      }
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
