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
