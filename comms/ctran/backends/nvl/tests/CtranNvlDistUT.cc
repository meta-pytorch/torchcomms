// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <pthread.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/nvl/CtranNvlImpl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

#include "comms/testinfra/TestsCuUtils.h"
#if !defined(USE_ROCM)
// needed because we use ncclMemAlloc to test kMemNcclMemAlloc mem type.
// cuMem API is not supported on AMD so we don't test it on AMD.
#include "comms/testinfra/TestUtils.h"
#endif

class CtranNvlTest : public ctran::CtranDistTestFixture {
 public:
  CtranNvlTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();

    // Check epoch lock for the entire test
    NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK = true;

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
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
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
};

class CtranNvlTestSuite : public CtranNvlTest,
                          public ::testing::WithParamInterface<MemAllocType> {};

TEST_P(CtranNvlTestSuite, NormalInitialize) {
  // Expect CtranNvl to be initialized without internal error
  try {
    auto ctranNvl = std::make_unique<CtranNvl>(this->comm);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "NVL backend failed to allocate. Skip test";
  }
}

TEST_P(CtranNvlTestSuite, RegMem) {
  // Expect CtranNvl can locally register and deregister GPU buffer without
  // internal error
  const auto memType = GetParam();

  std::unique_ptr<CtranNvl> ctranNvl;
  try {
    ctranNvl = std::make_unique<CtranNvl>(this->comm);

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "NVL backend failed to allocate. Skip test";
  }
  const size_t size = 1024;
  constexpr int numThreads = 10;
  std::vector<void*> bufs(numThreads, nullptr);
  for (int i = 0; i < numThreads; i++) {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&bufs[i], size));
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemAlloc(&bufs[i], size));
#else
      GTEST_SKIP() << "CuMem API is not supported on AMD, skip test";
#endif
    };
    ASSERT_NE(bufs[i], nullptr);
  }

  // Stress regMem by multiple threads
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          void* nvlRegElem = nullptr;
          CUDACHECK_TEST(cudaSetDevice(this->localRank));

          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          if (memType == kMemCudaMalloc) {
            COMMCHECK_TEST(
                CtranNvl::regMem(
                    bufs[tid], size, this->localRank, &nvlRegElem, true));
          } else {
            COMMCHECK_TEST(
                CtranNvl::regMem(
                    bufs[tid], size, this->localRank, &nvlRegElem));
          }

          ASSERT_NE(nvlRegElem, nullptr);
          COMMCHECK_TEST(CtranNvl::deregMem(nvlRegElem));
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < numThreads; i++) {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(bufs[i]));
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemFree(bufs[i]));
#else
      GTEST_SKIP() << "CuMem API is not supported on AMD, skip test";
#endif
    };
  }
}

TEST_P(CtranNvlTestSuite, CudaMallocRegMem) {
  // Expect regMem return success but nvlRegElem is empty, since cudaMalloc-ed
  // buffer is not supported
  const auto memType = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  try {
    auto ctranNvl = std::make_unique<CtranNvl>(this->comm);
    const size_t size = 1024;
    void* data = nullptr;
    CUDACHECK_TEST(cudaMalloc(&data, size));

    void* nvlRegElem = nullptr;
    auto res = CtranNvl::regMem(data, size, this->localRank, &nvlRegElem);

    EXPECT_EQ(res, commSuccess);
    ASSERT_EQ(nvlRegElem, nullptr);

    CUDACHECK_TEST(cudaFree(data));

  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "NVL backend failed to allocate. Skip test";
  }
}

TEST_P(CtranNvlTestSuite, ExportImportMem) {
  // - Expect rank 0 can export the buffer and share with rank 1 for importing
  // via control message.
  // - After importing, rank 1 confirms remote access to the
  // buffer.
  // - Finally, rank 0 releases the buffer and notifies rank 1 for the
  // remote release via another control message.
  // Require IB backend for control message exchange and notify for ACK.

  const auto memType = GetParam();

  std::unique_ptr<CtranNvl> ctranNvl;
  std::unique_ptr<CtranIb> ctranIb;
  try {
    ctranNvl = std::make_unique<CtranNvl>(this->comm);
    // TODO: move this to CtranComm once CtranIB is refactored
    ctranIb = std::make_unique<CtranIb>(this->comm, nullptr);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "NVL or IB backend failed to allocate. Skip test";
  }
  const size_t bufSize = 8192, dataCount = 100, dataOffset = 50;
  size_t dataRange = dataCount * sizeof(int);
  std::vector<int> assignVals(dataCount);
  for (int i = 0; i < dataCount; i++) {
    assignVals[i] = dataCount + i + 1;
  }

  CtranIbEpochRAII epochRAII(ctranIb.get());

  // Rank 0 exports a buffer
  const auto statex = comm->statex_.get();
  if (statex->rank() == 0) {
    const int peer = 1;
    void* dataBase = nullptr;
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&dataBase, bufSize));
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemAlloc(&dataBase, bufSize));
#else
      GTEST_SKIP() << "CuMem API is not supported on AMD, skip test";
#endif
    };
    ASSERT_NE(dataBase, nullptr);

    // Register a buffer in the middle of the cumem allocation
    void* data = reinterpret_cast<void*>(
        reinterpret_cast<uint64_t>(dataBase) + dataOffset);
    CUDACHECK_TEST(cudaMemcpy(
        data, assignVals.data(), dataCount * sizeof(int), cudaMemcpyDefault));

    // Local register
    void* regElems = nullptr;
    if (memType == kMemCudaMalloc) {
      COMMCHECK_TEST(
          CtranNvl::regMem(
              dataBase,
              dataCount * sizeof(int),
              this->localRank,
              &regElems,
              true));
    } else {
      COMMCHECK_TEST(
          CtranNvl::regMem(
              data, dataCount * sizeof(int), this->localRank, &regElems));
    }
    ASSERT_NE(regElems, nullptr);

    // Export - check export control message content and send to peer
    CtranNvlRegElem* nvlRegElem = reinterpret_cast<CtranNvlRegElem*>(regElems);
    ControlMsg msg(ControlMsgType::NVL_EXPORT_MEM);

    COMMCHECK_TEST(CtranNvl::exportMem(data, regElems, msg));
    auto ipcMem = nvlRegElem->ipcMem.rlock();
    dataRange = ipcMem->getRange();

    EXPECT_EQ(msg.type, ControlMsgType::NVL_EXPORT_MEM);
    EXPECT_EQ(
        reinterpret_cast<void*>(msg.nvlExp.ipcDesc.base),
        reinterpret_cast<void*>(ipcMem->getBase()));
    EXPECT_EQ(msg.nvlExp.ipcDesc.range, ipcMem->getRange());
    EXPECT_EQ(msg.nvlExp.ipcDesc.numSegments, 1);
    EXPECT_NE(msg.nvlExp.ipcDesc.segments[0].sharedHandle.fd, 0);
    EXPECT_GT(msg.nvlExp.ipcDesc.segments[0].range, 0);
    EXPECT_EQ(msg.nvlExp.ipcDesc.pid, getpid());
    EXPECT_EQ(msg.nvlExp.offset, dataOffset);

    COMMCHECK_TEST(ibSendCtrl(msg, peer, ctranIb));

    // Ensure remote rank has imported the sharedHandle before local release
    ctranIb->waitNotify(peer);

    // Remote release - check control message content and send to peer
    ControlMsg releaseMsg;
    CtranNvl::remReleaseMem(nvlRegElem, releaseMsg);
    EXPECT_EQ(releaseMsg.type, ControlMsgType::NVL_RELEASE_MEM);
    EXPECT_EQ(releaseMsg.nvlRls.base, msg.nvlExp.ipcDesc.base);

    COMMCHECK_TEST(ibSendCtrl(releaseMsg, peer, ctranIb));

    // Local deregister
    COMMCHECK_TEST(ctranNvl->deregMem(regElems));
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(dataBase));
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemFree(dataBase));
#else
      GTEST_SKIP() << "CuMem API is not supported on AMD, skip test";
#endif
    };

  } else if (statex->rank() == 1) {
    // Rank 1 imports a buffer from rank 0
    const int peer = 0;

    // Import - receive from peer and check control message content
    ControlMsg msg(ControlMsgType::NVL_EXPORT_MEM);
    COMMCHECK_TEST(ibRecvCtrl(msg, peer, ctranIb));

    EXPECT_EQ(msg.type, ControlMsgType::NVL_EXPORT_MEM);
    EXPECT_EQ(msg.nvlExp.offset, dataOffset);
    EXPECT_GT(msg.nvlExp.ipcDesc.pid, 0);
    EXPECT_EQ(msg.nvlExp.ipcDesc.numSegments, 1);
    EXPECT_NE(msg.nvlExp.ipcDesc.segments[0].sharedHandle.fd, 0);
    EXPECT_GT(msg.nvlExp.ipcDesc.segments[0].range, 0);
    EXPECT_GE(msg.nvlExp.ipcDesc.range, dataRange);

    // Import
    void* mappedData = nullptr;
    struct CtranNvlRemoteAccessKey rkey{};
    auto res = ctranNvl->importMem(&mappedData, &rkey, peer, msg);
    EXPECT_EQ(res, commSuccess);
    EXPECT_NE(mappedData, nullptr);

    EXPECT_EQ(rkey.peerRank, peer);
    EXPECT_EQ(rkey.basePtr, msg.nvlExp.ipcDesc.base);

    // Ack to peer after import so that peer can release the sharedHandle
    COMMCHECK_TEST(ibNotify(peer, ctranIb));

    // Check access to the mapped buffer
    std::vector<int> checkVals(dataCount);
    CUDACHECK_TEST(cudaMemcpy(
        checkVals.data(),
        mappedData,
        dataCount * sizeof(int),
        cudaMemcpyDefault));
    EXPECT_THAT(checkVals, ::testing::ElementsAreArray(assignVals));

    // Check received release message
    ControlMsg releaseMsg(ControlMsgType::NVL_RELEASE_MEM);
    COMMCHECK_TEST(ibRecvCtrl(releaseMsg, peer, ctranIb));
    EXPECT_EQ(releaseMsg.type, ControlMsgType::NVL_RELEASE_MEM);
    EXPECT_EQ(releaseMsg.nvlRls.base, msg.nvlExp.ipcDesc.base);

    // Skip check for implicit release triggered via internal
    // callback. Expect ctranNvl destructor can trigger the release

    // Explicitly release imported memory
    COMMCHECK_TEST(ctranNvl->releaseMem(&rkey));

    // Expect no more remote registration
    ASSERT_EQ(ctranNvl->getNumRemMem(peer), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CtranNvlTestInstance,
    CtranNvlTestSuite,
#if !defined(USE_ROCM)
    ::testing::Values(kMemNcclMemAlloc, kMemCudaMalloc));
#else
    ::testing::Values(kMemCudaMalloc));
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
