// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/utils/Checks.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

// Reduce the value range to avoid integer overflow when running large count
constexpr size_t VAL_RANGE = 1024;
// Reduce the value range for commProd to avoid accumulated precision loss or
// numerical difference between CPU and GPU for floating points
constexpr size_t VAL_RANGE_PROD = 8;

template <typename TYPE>
class CtranAllReduceTest : public CtranDistBaseTest {
 public:
  CtranAllReduceTest() = default;
  commDataType_t dt = getCommDataType<TYPE>();
  size_t bytes;
  size_t bufSize;
  void *sendbuf, *recvbuf;
  std::vector<TestMemSegment> segments;
  std::vector<void*> segHandles;
  TYPE* hostbuf;
  ncclComm_t comm;

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    segments.clear();
    segHandles.clear();
    if (!ctranAllReduceSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "ctranAllReduceSupport returns fails, skip test";
    }
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void memorySetUp(
      size_t count,
      TestInPlaceType inplace,
      commRedOp_t op,
      MemAllocType memType) {
    sendbuf = recvbuf = nullptr;
    bytes = count * commTypeSize(dt);
    if (bytes < CTRAN_MIN_REGISTRATION_SIZE) {
      bytes = CTRAN_MIN_REGISTRATION_SIZE;
    }
    bufSize = bytes;

    CUDACHECK_TEST(cudaHostAlloc(&hostbuf, bytes, 0));
    for (size_t i = 0; i < count; i++) {
      auto val = i % VAL_RANGE + globalRank;
      if (op == commProd) {
        // use smaller value range to avoid overflow or accumulated precision
        // loss for floating points
        hostbuf[i] = (TYPE)(val % VAL_RANGE_PROD);
      } else {
        hostbuf[i] = (TYPE)(val);
      }
    }

    sendbuf = prepareBuf(bytes, memType, segments);
    CUDACHECK_TEST(cudaMemcpy(sendbuf, hostbuf, bytes, cudaMemcpyDefault));

    if (inplace == kTestOutOfPlace) {
      recvbuf = prepareBuf(bytes, memType, segments);
      CUDACHECK_TEST(cudaMemcpy(recvbuf, hostbuf, bytes, cudaMemcpyDefault));
    } else {
      recvbuf = sendbuf;
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void memoryCleanUp(MemAllocType memType) {
    CUDACHECK_TEST(cudaFreeHost(hostbuf));
    if (recvbuf != sendbuf) {
      releaseBuf(recvbuf, bytes, memType);
    }
    releaseBuf(sendbuf, bytes, memType);
  }

  void verifyResult(size_t count, commRedOp_t op) {
    std::vector<TYPE> observedVals(count, 117);
    FB_CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        recvbuf,
        count * commTypeSize(dt),
        cudaMemcpyDefault));
    int error_count = 0;
    for (size_t i = 0; i < count; i++) {
      TYPE exp = (TYPE)0;
      size_t baseVal = i % VAL_RANGE;
      if (op == commSum) {
        exp = (TYPE)(baseVal * this->numRanks +
                     this->numRanks * (this->numRanks - 1) / 2);
      } else if (op == commProd) {
        exp = (TYPE)1;
        for (size_t j = 0; j < this->numRanks; j++) {
          exp *= TYPE((baseVal + j) % VAL_RANGE_PROD);
        }
      } else if (op == commMax) {
        exp = (TYPE)(baseVal + this->numRanks - 1);
      } else if (op == commMin) {
        exp = (TYPE)(baseVal);
      } else if (op == commAvg) {
        exp = (TYPE)(baseVal + TYPE(TYPE(this->numRanks - 1) / 2));
      }
      // log the first 3 errors
      if (error_count < 3) {
        EXPECT_EQ(observedVals[i], exp) << "  i=" << i << std::endl
                                        << "  count=" << count << " on rank "
                                        << this->globalRank << std::endl;
      }
      // count errors
      if (observedVals[i] != exp) {
        if (error_count < 20) {
          XLOG(WARN) << "error[" << error_count << "]: " << " data[" << i
                     << "] " << observedVals[i] << " vs exp " << exp;
        }
        error_count++;
      }
    }
    ASSERT_EQ(error_count, 0) << "  error count=" << count << " on rank "
                              << this->globalRank << std::endl;
  }

  /* test given Allreduce function */
  void beginTest(
      commResult_t allreduceFunc(
          const void* sendbuff,
          void* recvbuff,
          size_t count,
          commDataType_t datatype,
          commRedOp_t redOp,
          CtranComm* comm,
          cudaStream_t stream,
          std::optional<std::chrono::milliseconds> timeout),
      enum NCCL_ALLREDUCE_ALGO algo,
      size_t count,
      TestInPlaceType inplace,
      commRedOp_t op,
      MemAllocType memType) {
    if (memType == kCuMemAllocDisjoint &&
        (!comm->dmaBufSupport || !NCCL_CTRAN_IB_DMABUF_ENABLE)) {
      GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
    }

    memorySetUp(count, inplace, op, memType);

    for (auto& segment : segments) {
      void* hdl = nullptr;
      NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
      segHandles.push_back(hdl);
    }

    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(
            comm->ctranComm_.get()));

    if (inplace == kTestInPlace) {
      auto res = allreduceFunc(
          recvbuf,
          recvbuf,
          count,
          dt,
          op,
          comm->ctranComm_.get(),
          stream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
    } else {
      auto res = allreduceFunc(
          sendbuf,
          recvbuf,
          count,
          dt,
          op,
          comm->ctranComm_.get(),
          stream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
    }

    FB_CUDACHECKIGNORE(cudaStreamSynchronize(stream));

    verifyResult(count, op);

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

    auto lastColl = pastCollsJson[0];
    EXPECT_EQ(lastColl["opName"].asString(), "AllReduce");
    EXPECT_EQ(lastColl["count"].asInt(), count);
    EXPECT_THAT(
        lastColl["algoName"].asString(),
        testing::HasSubstr(allReduceAlgoName(algo)));

    verifyBackendsUsed(
        comm->ctranComm_->ctran_.get(),
        comm->ctranComm_->statex_.get(),
        kMemNcclMemAlloc,
        // AllReduce uses kernel reduce not NVL iput
        {CtranMapperBackend::NVL});
    verifyGpeLeak(comm->ctranComm_->ctran_.get());

    for (auto& hdl : segHandles) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
    }

    memoryCleanUp(memType);
  }
};

class CtranAllReduceTestParamUInt64
    : public CtranAllReduceTest<uint64_t>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};

TEST_P(CtranAllReduceTestParamUInt64, AllReduceDirectUInt64) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceDirect,
      NCCL_ALLREDUCE_ALGO::ctdirect,
      count,
      inplace,
      op,
      memType);
}

class CtranAllReduceTestParamFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};

TEST_P(CtranAllReduceTestParamFp32, AllReduceDirectFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceDirect,
      NCCL_ALLREDUCE_ALGO::ctdirect,
      count,
      inplace,
      op,
      memType);
}

// common parameters for all tests
auto testingValues = ::testing::Values(
    std::make_tuple(1, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(2, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(3, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(9, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(1, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(2, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(3, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(9, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024,
        kTestInPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestInPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commAvg, kMemNcclMemAlloc));

// common function to get test name from test parameter
inline std::string getTestName(
    const testing::TestParamInfo<CtranAllReduceTestParamUInt64::ParamType>&
        info) {
  return std::to_string(std::get<0>(info.param)) + "elements_" +
      testInPlaceTypeToStr(std::get<1>(info.param)) + "_" +
      commOpToString(std::get<2>(info.param)) + "_" +
      testMemAllocTypeToStr(std::get<3>(info.param));
}

// Tests for UInt64
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceTestParamUInt64,
    testingValues,
    getTestName);
// Tests for Float32
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceTestParamFp32,
    testingValues,
    getTestName);

// TODO: enable ctring test for nLocalRanks > 1 case, currently CtranIB connect
// to localRanks does not seem to work.
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL

// TODO: enable tiny message sizes and ops other than commSum. This is a
// separate class because Ring does not support some sizes & ops yet
class CtranAllReduceRingTestParamUInt64
    : public CtranAllReduceTest<uint64_t>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};
TEST_P(CtranAllReduceRingTestParamUInt64, AllReduceRingUInt64) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

// TODO: enable tiny message sizes and ops other than commSum. This is a
// separate class because Ring does not support some sizes & ops yet
class CtranAllReduceRingTestParamFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};
TEST_P(CtranAllReduceRingTestParamFp32, AllReduceRingFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

auto testingValuesRing = ::testing::Values(
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024,
        kTestInPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestInPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 15,
        kTestOutOfPlace,
        commAvg,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestOutOfPlace,
        commAvg,
        kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 15,
        kTestInPlace,
        commAvg,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 * 1024 + 17,
        kTestInPlace,
        commAvg,
        kMemNcclMemAlloc));

// Tests for UInt64
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamUInt64,
    testingValuesRing,
    getTestName);
// Tests for Float32
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamFp32,
    testingValuesRing,
    getTestName);

#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
