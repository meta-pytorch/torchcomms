// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <thread>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CtranBroadcastTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  CtranBroadcastTest() = default;
  std::vector<TestMemSegment> segments;

  void SetUp() override {
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    ctran::CtranDistTestFixture::SetUp();
    srand(time(NULL));
    ctranComm = makeCtranComm();
    segments.clear();
    if (!ctranBroadcastSupport(ctranComm.get(), NCCL_BROADCAST_ALGO)) {
      GTEST_SKIP() << "ctranBroadcastSupport returns false, skip test";
    }
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
  }

  template <typename T>
  ulong checkChunkValue(T* buf, ssize_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    ulong errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
};

class CtranTestBroadcastFixture
    : public CtranBroadcastTest,
      public ::testing::WithParamInterface<std::tuple<
          std::tuple<size_t, ssize_t, TestInPlaceType, MemAllocType>,
          bool>> {};

TEST_P(CtranTestBroadcastFixture, Broadcast) {
  auto res = commSuccess;
  // test various size and various num of max QP, intentionally make some sizes
  // not aligned
  const auto& [innerTuple, binomialTreeAlgo] = GetParam();
  const auto& [offset, count, inplace, memType] = innerTuple;
  const size_t pageSize = getpagesize();
  commDataType_t dt = commFloat32;

  // Check cumem after comm creation to make sure we have loaded cu symbols
  if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
      ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  if (memType == kCuMemAllocDisjoint && !NCCL_CTRAN_IB_DMABUF_ENABLE) {
    GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
  }

#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
  if (memType == kMemCudaMalloc) {
    GTEST_SKIP() << "Socket backend does not support cudaMalloc";
  }
#endif

  // always allocate buffer in page size
  size_t bufSize =
      (((offset + count) * commTypeSize(dt) + pageSize - 1) / pageSize) *
      pageSize * 2;
  size_t sendSize = count * commTypeSize(dt);
  const int sendRank = 0;
  void* base = prepareBuf(bufSize, memType, segments);

  ASSERT_TRUE(
      meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
  }

  char* sourceBuf = reinterpret_cast<char*>(base) + offset;
  char* targetBuf = sourceBuf;
  if (inplace == kTestOutOfPlace) {
    targetBuf = reinterpret_cast<char*>(base) + bufSize / 2 + offset;
  }

  if (globalRank == sendRank) {
    printf(
        "Rank %d sendRank %d send to others with offset %ld count %ld %s %s\n",
        ctranComm->statex_->rank(),
        sendRank,
        offset,
        count,
        testInPlaceTypeToStr(inplace).c_str(),
        testMemAllocTypeToStr(memType).c_str());

    CUDACHECK_TEST(cudaMemset(sourceBuf, 1, sendSize));
  } else {
    CUDACHECK_TEST(cudaMemset(base, rand(), bufSize));
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (binomialTreeAlgo) {
    res = ctranBroadcastBinomialTree(
        sourceBuf, targetBuf, count, dt, sendRank, ctranComm.get(), testStream);
  } else {
    res = ctranBroadcastDirect(
        sourceBuf, targetBuf, count, dt, sendRank, ctranComm.get(), testStream);
  }
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(testStream));

  CUDACHECK_TEST(cudaDeviceSynchronize());
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::seconds(2));

  ASSERT_NE(ctranComm->colltraceNew_, nullptr);
  auto dumpMap = ctran::dumpCollTrace(ctranComm.get());

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColls"], "[]");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), 1);

  const auto& coll = pastCollsJson[0];
  EXPECT_EQ(coll["opName"].asString(), "Broadcast");
  EXPECT_EQ(coll["count"].asInt(), count);
  if (binomialTreeAlgo) {
    EXPECT_EQ(
        coll["algoName"].asString(),
        broadcastAlgoName(NCCL_BROADCAST_ALGO::ctbtree));
  } else {
    EXPECT_EQ(
        coll["algoName"].asString(),
        broadcastAlgoName(NCCL_BROADCAST_ALGO::ctdirect));
  }

  if (globalRank == sendRank) {
    verifyBackendsUsed(
        ctranComm->ctran_.get(), ctranComm->statex_.get(), memType);
  }

  verifyGpeLeak(ctranComm->ctran_.get());

  // First deregister buffer to catch potential 'remote access error' caused
  // by incomplete ctranSend when ctranRecv has returned incorrectly.
  // Delaying it after check can lead to false positive since ctranSend may
  // eventually complete.
  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
  }

  if (globalRank != sendRank) {
    ulong errs = checkChunkValue(targetBuf, sendSize, (char)1);
    EXPECT_EQ(errs, 0);
  }

  releaseBuf(base, bufSize, memType);
}

// test various size and various num of max QP, intentionally make some sizes
// not aligned
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranTestBroadcastFixture,
    ::testing::Combine(
        ::testing::Values(
            // short buffers <4097B (1024 FP32)
            std::make_tuple(0, 1UL, kTestOutOfPlace, kMemNcclMemAlloc),
            std::make_tuple(0, 64UL, kTestInPlace, kMemNcclMemAlloc),
            std::make_tuple(0, 1024UL, kTestOutOfPlace, kMemNcclMemAlloc),
            std::make_tuple(0, 4096UL, kTestInPlace, kMemCudaMalloc),
            std::make_tuple(0, 65536UL, kTestInPlace, kMemCudaMalloc),
            // test ncclMemAlloc based memory
            std::make_tuple(0, 4096UL, kTestInPlace, kMemNcclMemAlloc),
            // unaligned addr and size
            std::make_tuple(5, 2097155UL, kTestInPlace, kMemNcclMemAlloc),
            // // unaligned size
            std::make_tuple(0, 2097155UL, kTestInPlace, kMemNcclMemAlloc),
            // // large and unaligned
            std::make_tuple(5, 1073741819UL, kTestInPlace, kMemNcclMemAlloc),

            // test out-of-place
            std::make_tuple(0, 4096UL, kTestOutOfPlace, kMemNcclMemAlloc),
            // unaligned addr and size
            std::make_tuple(5, 2097155UL, kTestOutOfPlace, kMemNcclMemAlloc),
            // unaligned size
            std::make_tuple(0, 2097155UL, kTestOutOfPlace, kMemNcclMemAlloc),
            // large and unaligned
            std::make_tuple(5, 1073741819UL, kTestOutOfPlace, kMemNcclMemAlloc)
            //  test ncclMemAllocDisjoint memory
            // std::make_tuple(
            //     0,
            //     1UL << 21,
            //     kTestOutOfPlace,
            //     kCuMemAllocDisjoint)
            ),
        ::testing::Values(false, true)),
    [&](const testing::TestParamInfo<CtranTestBroadcastFixture::ParamType>&
            info) {
      return std::to_string(std::get<0>(std::get<0>(info.param))) + "offset_" +
          std::to_string(std::get<1>(std::get<0>(info.param))) + "fp32_" +
          testInPlaceTypeToStr(std::get<2>(std::get<0>(info.param))) + "_" +
          testMemAllocTypeToStr(std::get<3>(std::get<0>(info.param))) + "_" +
          (std::get<1>(info.param) ? "ctbtree" : "ctran");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
