// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

#include <folly/json/json.h>

class CtranBroadcastTest : public CtranDistBaseTest {
 public:
  CtranBroadcastTest() = default;
  // TODO: mark it as deprectated
  // TODO: replace ncclComm with CtranComm
  ncclComm_t comm;
  std::vector<TestMemSegment> segments;
  std::vector<void*> segHandles;

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    CtranDistBaseTest::SetUp();
    srand(time(NULL));
    comm = commWorld;
    segments.clear();
    segHandles.clear();
    if (!ctranBroadcastSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "ctranBroadcastSupport returns false, skip test";
    }
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
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

  if (memType == kCuMemAllocDisjoint &&
      (!comm->dmaBufSupport || !NCCL_CTRAN_IB_DMABUF_ENABLE)) {
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

  ASSERT_TRUE(meta::comms::colltrace::testOnlyClearCollTraceRecords(
      comm->ctranComm_.get()));

  for (auto& segment : segments) {
    void* hdl = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
    segHandles.push_back(hdl);
  }

  char* sourceBuf = reinterpret_cast<char*>(base) + offset;
  char* targetBuf = sourceBuf;
  if (inplace == kTestOutOfPlace) {
    targetBuf = reinterpret_cast<char*>(base) + bufSize / 2 + offset;
  }

  if (globalRank == sendRank) {
    printf(
        "Rank %d sendRank %d send to others with offset %ld count %ld %s %s\n",
        comm->ctranComm_->statex_->rank(),
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
        sourceBuf,
        targetBuf,
        count,
        dt,
        sendRank,
        comm->ctranComm_.get(),
        stream);
  } else {
    res = ctranBroadcastDirect(
        sourceBuf,
        targetBuf,
        count,
        dt,
        sendRank,
        comm->ctranComm_.get(),
        stream);
  }
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  CUDACHECK_TEST(cudaDeviceSynchronize());
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

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
        comm->ctranComm_->ctran_.get(),
        comm->ctranComm_->statex_.get(),
        memType);
  }

  verifyGpeLeak(comm->ctranComm_->ctran_.get());

  // First deregister buffer to catch potential 'remote access error' caused
  // by incomplete ctranSend when ctranRecv has returned incorrectly.
  // Delaying it after check can lead to false positive since ctranSend may
  // eventually complete.
  for (auto& hdl : segHandles) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
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
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
