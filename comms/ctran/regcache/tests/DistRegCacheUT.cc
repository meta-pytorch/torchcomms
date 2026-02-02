// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranNcclTestUtils.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class DistRegCacheTest : public NcclxBaseTest {
 public:
  int cudaDev{0};
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_FASTINIT_MODE", "ring_hybrid", 1);
    NcclxBaseTest::SetUp();

    commDeprecated_ = createNcclComm(globalRank, numRanks, localRank);
    comm_ = commDeprecated_->ctranComm_.get();
    cudaDev = comm_->statex_->cudaDev();

    // Turn on profiler after initialization to track only test registrations
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = 0;

    if (!ctranInitialized(comm_) || !comm_->ctran_->mapper->hasBackend()) {
      GTEST_SKIP()
          << "Ctran is not initialized or backend is not available.  Skip test.";
    }

    regCache = ctran::RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    // Turn off profiler to avoid internal in comm destroy.
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = -1;

    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated_));
    NcclxBaseTest::TearDown();
  }

  commResult_t
  ibSendCtrl(ControlMsg& msg, int peer, std::unique_ptr<CtranIb>& ctranIb) {
    CtranIbRequest req;
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), peer, req));
    while (!req.isComplete()) {
      COMMCHECK_TEST(ctranIb->progress());
    }
    return commSuccess;
  }

  commResult_t
  ibRecvCtrl(ControlMsg& msg, int peer, std::unique_ptr<CtranIb>& ctranIb) {
    CtranIbRequest req;
    COMMCHECK_TEST(ctranIb->irecvCtrlMsg(&msg, sizeof(msg), peer, req));
    while (!req.isComplete()) {
      COMMCHECK_TEST(ctranIb->progress());
    }
    return commSuccess;
  }

  commResult_t ibNotify(int peer, std::unique_ptr<CtranIb>& ctranIb) {
    CtranIbRequest req;
    COMMCHECK_TEST(ctranIb->notify(peer, &req));
    while (!req.isComplete()) {
      COMMCHECK_TEST(ctranIb->progress());
    }
    return commSuccess;
  }

 protected:
  ncclComm_t commDeprecated_{nullptr};
  CtranComm* comm_{nullptr};
};

class DistRegCacheTestSuite
    : public DistRegCacheTest,
      public ::testing::WithParamInterface<MemAllocType> {};

TEST_P(DistRegCacheTestSuite, RegMem) {
  // Expect IpcRegCache can locally register and deregister GPU buffer without
  // internal error
  const auto memType = GetParam();

  const size_t size = 1024;
  constexpr int numThreads = 10;
  std::vector<void*> bufs(numThreads, nullptr);
  for (int i = 0; i < numThreads; i++) {
    std::vector<TestMemSegment> segments;
    bufs[i] = ctran::CtranNcclTestHelpers::prepareBuf(size, memType, segments);
    ASSERT_NE(bufs[i], nullptr);
  }

  // Stress regMem by multiple threads
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          void* ipcRegElem = nullptr;
          CUDACHECK_TEST(cudaSetDevice(localRank));

          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          if (memType == kMemCudaMalloc) {
            COMMCHECK_TEST(
                ctran::IpcRegCache::regMem(
                    bufs[tid], size, localRank, &ipcRegElem, true));
          } else {
            COMMCHECK_TEST(
                ctran::IpcRegCache::regMem(
                    bufs[tid], size, localRank, &ipcRegElem));
          }

          ASSERT_NE(ipcRegElem, nullptr);
          ctran::IpcRegCache::deregMem(ipcRegElem);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < numThreads; i++) {
    ctran::CtranNcclTestHelpers::releaseBuf(bufs[i], size, memType);
  }
}

TEST_P(DistRegCacheTestSuite, ExportImportMem) {
  // Test that rank 0 can export a buffer and share with rank 1 for importing
  // via control message. After importing, rank 1 confirms remote access to the
  // buffer. Finally, rank 0 releases the buffer and notifies rank 1 for the
  // remote release. Uses CtranMapperRegMem APIs for export/import/release.
  // Require IB backend for control message exchange and notify for ACK.
  const auto memType = GetParam();

  auto& mapper = comm_->ctran_->mapper;
  ASSERT_NE(mapper, nullptr);
  auto* ipcRegCache = mapper->ipcRegCachePtr();
  ASSERT_NE(mapper, nullptr);

  std::unique_ptr<CtranIb> ctranIb;
  try {
    ctranIb = std::make_unique<CtranIb>(comm_, nullptr);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "IB backend failed to allocate. Skip test";
  }

  const size_t bufSize = 8192;
  const size_t dataCount = 100;
  const size_t dataOffset = 50;
  size_t dataRange = dataCount * sizeof(int);

  CtranIbEpochRAII epochRAII(ctranIb.get());
  if (globalRank == 0) {
    const int peer = 1;
    std::vector<TestMemSegment> segments;
    void* dataBase =
        ctran::CtranNcclTestHelpers::prepareBuf(bufSize, memType, segments);
    ASSERT_NE(dataBase, nullptr);

    void* data = reinterpret_cast<void*>(
        reinterpret_cast<uint64_t>(dataBase) + dataOffset);
    assignChunkValue((int*)data, dataCount, (int)dataCount, (int)1);

    void* segHdl;
    ctran::regcache::RegElem* regHdl = nullptr;
    if (memType == kMemCudaMalloc) {
      COMMCHECK_TEST(mapper->regMem(
          dataBase, bufSize, &segHdl, true, true, (void**)&regHdl));
    } else {
      COMMCHECK_TEST(mapper->regMem(
          data, dataRange, &segHdl, true, true, (void**)&regHdl));
    }
    ASSERT_NE(regHdl, nullptr);

    ControlMsg msg(ControlMsgType::NVL_EXPORT_MEM);
    COMMCHECK_TEST(
        ipcRegCache->exportMem(data, regHdl->ipcRegElem, msg.ipcDesc));
    ctran::regcache::IpcRegElem* ipcRegElem =
        reinterpret_cast<ctran::regcache::IpcRegElem*>(regHdl->ipcRegElem);
    auto ipcMem = ipcRegElem->ipcMem.rlock();

    EXPECT_EQ(msg.type, ControlMsgType::NVL_EXPORT_MEM);
    EXPECT_EQ(msg.ipcDesc.desc.range, ipcMem->getRange());
    EXPECT_EQ(msg.ipcDesc.desc.numSegments, 1);
    EXPECT_NE(msg.ipcDesc.desc.segments[0].sharedHandle.fd, 0);
    EXPECT_GT(msg.ipcDesc.desc.segments[0].range, 0);
    EXPECT_EQ(msg.ipcDesc.desc.pid, getpid());
    EXPECT_EQ(msg.ipcDesc.offset, dataOffset);
    COMMCHECK_TEST(ibSendCtrl(msg, peer, ctranIb));

    // send release-mem msg to peer
    ctranIb->waitNotify(peer);
    ControlMsg releaseMsg(ControlMsgType::NVL_RELEASE_MEM);
    releaseMsg.ipcRls.base = msg.ipcDesc.desc.base;
    COMMCHECK_TEST(ibSendCtrl(releaseMsg, peer, ctranIb));

    COMMCHECK_TEST(mapper->deregMem(segHdl, true));
    ctran::CtranNcclTestHelpers::releaseBuf(dataBase, bufSize, memType);

  } else if (globalRank == 1) {
    const int peer = 0;
    auto peerId = comm_->statex_->gPid(peer);
    ControlMsg msg;
    COMMCHECK_TEST(ibRecvCtrl(msg, peer, ctranIb));
    EXPECT_EQ(msg.type, ControlMsgType::NVL_EXPORT_MEM);
    EXPECT_GE(msg.ipcDesc.desc.range, dataRange);

    void* mappedData = nullptr;
    CtranMapperRemoteAccessKey remKey{};
    remKey.backend = CtranMapperBackend::NVL;
    COMMCHECK_TEST(ipcRegCache->importMem(
        peerId, msg.ipcDesc, &mappedData, &remKey.nvlKey));
    EXPECT_NE(mappedData, nullptr);
    EXPECT_EQ(remKey.nvlKey.basePtr, msg.ipcDesc.desc.base);

    COMMCHECK_TEST(ibNotify(peer, ctranIb));

    EXPECT_EQ(
        checkChunkValue((int*)mappedData, dataCount, (int)dataCount, (int)1),
        0);
    ControlMsg releaseMsg(ControlMsgType::NVL_RELEASE_MEM);
    COMMCHECK_TEST(ibRecvCtrl(releaseMsg, peer, ctranIb));
    EXPECT_EQ(releaseMsg.type, ControlMsgType::NVL_RELEASE_MEM);
    EXPECT_EQ(releaseMsg.ipcRls.base, msg.ipcDesc.desc.base);

    COMMCHECK_TEST(ipcRegCache->releaseRemReg(
        peerId, reinterpret_cast<void*>(msg.ipcDesc.desc.base)));

    EXPECT_EQ(ipcRegCache->getNumRemReg(peerId), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DistRegCacheInstance,
    DistRegCacheTestSuite,
#if !defined(USE_ROCM)
    ::testing::Values(kMemNcclMemAlloc, kMemCudaMalloc, kCuMemAllocDisjoint));
#else
    ::testing::Values(kMemCudaMalloc));
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
