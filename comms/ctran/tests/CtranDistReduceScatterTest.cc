// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <iostream>
#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

#include <folly/json/json.h>

// Reduce the value range to avoid integer overflow when running large count
constexpr size_t VAL_RANGE = 1000;
// Reduce the value range for commProd to avoid accumulated precision loss or
// numerical difference between CPU and GPU for floating points
constexpr size_t VAL_RANGE_PROD = 10;

template <typename T>
class CtranReduceScatterTest : public CtranDistBaseTest {
 public:
  CtranReduceScatterTest() = default;
  commDataType_t dt = getCommDataType<T>();
  size_t sendBytes, recvBytes;
  T *sendBuf, *recvBuf;
  std::vector<TestMemSegment> segments;
  std::vector<void*> segHandles;
  T* hostbuf;
  ncclComm_t comm;

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    if (!ctranReduceScatterSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "ctranReduceScatterSupport returns fails, skip test";
    }
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void memorySetUp(
      MemAllocType memType,
      size_t count,
      commRedOp_t op,
      bool registerFlag = true) {
    // Check cumem after comm creation to make sure we have loaded cu symbols
    if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
        ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    sendBuf = recvBuf = nullptr;
    recvBytes = count * commTypeSize(dt);
    if (recvBytes < CTRAN_MIN_REGISTRATION_SIZE) {
      recvBytes = CTRAN_MIN_REGISTRATION_SIZE;
    }
    sendBytes = recvBytes * numRanks;

    CUDACHECK_TEST(cudaHostAlloc(&hostbuf, sendBytes, 0));
    for (size_t i = 0; i < count * numRanks; i++) {
      auto val = i % VAL_RANGE + globalRank;
      if (op == commProd) {
        // use smaller value range to avoid overflow or accumulated precision
        // loss for floating points
        hostbuf[i] = (T)(val % VAL_RANGE_PROD);
      } else {
        hostbuf[i] = (T)(val);
      }
    }

    sendBuf = reinterpret_cast<T*>(
        prepareBuf(pageAligned(sendBytes), memType, segments));
    CUDACHECK_TEST(cudaMemcpy(sendBuf, hostbuf, sendBytes, cudaMemcpyDefault));

    recvBuf = reinterpret_cast<T*>(
        prepareBuf(pageAligned(recvBytes), memType, segments));
    CUDACHECK_TEST(cudaMemcpy(recvBuf, hostbuf, recvBytes, cudaMemcpyDefault));

    if (registerFlag) {
      for (auto& segment : segments) {
        void* hdl = nullptr;
        NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
        segHandles.push_back(hdl);
      }
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void memoryCleanUp(MemAllocType memType) {
    CUDACHECK_TEST(cudaFreeHost(hostbuf));
    for (auto& hdl : segHandles) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
    }

    releaseBuf(sendBuf, sendBytes, memType);
    releaseBuf(recvBuf, recvBytes, memType);
  }

  void verifyResult(size_t count, commRedOp_t op, T* recvBufComm) {
    std::vector<T> observedVals(count, 117);
    CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        recvBufComm,
        count * commTypeSize(dt),
        cudaMemcpyDefault));
    for (size_t i = 0; i < count; i++) {
      T exp = 0;
      size_t baseVal = (i + globalRank * count) % VAL_RANGE;
      if (op == commSum) {
        exp = (T)(baseVal * this->numRanks +
                  this->numRanks * (this->numRanks - 1) / 2);
      } else if (op == commProd) {
        exp = 1;
        for (size_t j = 0; j < this->numRanks; j++) {
          baseVal = (i + globalRank * count) % VAL_RANGE;
          exp *= T((baseVal + j) % VAL_RANGE_PROD);
        }
      } else if (op == commMax) {
        exp = (T)(baseVal + this->numRanks - 1);
      } else if (op == commMin) {
        exp = (T)(baseVal);
      } else if (op == commAvg) {
        exp = (T)(baseVal + T(T(this->numRanks - 1) / 2));
      }
      ASSERT_EQ(observedVals[i], exp) << "  i=" << i << std::endl
                                      << "  count=" << count << " on rank "
                                      << this->globalRank << std::endl;
    }
  }

  void beginTest(
      size_t count,
      TestInPlaceType inplace,
      bool regist,
      MemAllocType memType,
      commRedOp_t redOp,
      enum NCCL_REDUCESCATTER_ALGO algo) {
    EnvRAII env(NCCL_REDUCESCATTER_ALGO, algo);

    if (algo == NCCL_REDUCESCATTER_ALGO::ctdirect) {
      const int nNodes = comm->ctranComm_->statex_->nNodes();
      if (nNodes != 1) {
        GTEST_SKIP() << "ctdirect only supports nNodes=1, but got " << nNodes
                     << ", skip test";
      }
    } else if (algo == NCCL_REDUCESCATTER_ALGO::ctring) {
      const int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
      if (nLocalRanks != 1) {
        GTEST_SKIP() << "ctring only supports nLocalRanks=1, but got "
                     << nLocalRanks << ", skip test";
      }
    } else if (algo == NCCL_REDUCESCATTER_ALGO::ctrhd) {
      const int nNodes = comm->ctranComm_->statex_->nNodes();
      const int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
      if (nLocalRanks != 1) {
        GTEST_SKIP() << "ctrhd only supports nLocalRanks=1, but got "
                     << nLocalRanks << ", skip test";
      }
      if ((nNodes & (nNodes - 1)) != 0) {
        GTEST_SKIP() << "ctrhd only supports power-of-two number "
                     << "of nodes but got " << nNodes << ", skip test";
      }
      const size_t recvBytes = count * commTypeSize(dt);
      const size_t totalBufSize = recvBytes * numRanks;
      if (NCCL_CTRAN_INTERNODE_TMPBUF_SIZE < totalBufSize) {
        GTEST_SKIP() << "data buffer of size " << totalBufSize
                     << " bytes is too large to fit in tmpBuf for "
                     << "ctrhd, skip test";
      }
    }

    if (memType == kCuMemAllocDisjoint &&
        (!comm->dmaBufSupport || !NCCL_CTRAN_IB_DMABUF_ENABLE)) {
      GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
    }

    memorySetUp(memType, count, redOp, regist);
    ASSERT_TRUE(meta::comms::colltrace::testOnlyClearCollTraceRecords(
        comm->ctranComm_.get()));

    T* recvBufComm = recvBuf;
    if (inplace == kTestInPlace) {
      recvBufComm = reinterpret_cast<T*>(sendBuf) + globalRank * count;
    }
    auto res = ctranReduceScatter(
        sendBuf, recvBufComm, count, dt, redOp, comm->ctranComm_.get(), stream);
    EXPECT_EQ(res, commSuccess);

    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    verifyResult(count, redOp, recvBufComm);

    verifyBackendsUsed(
        comm->ctranComm_->ctran_.get(),
        comm->ctranComm_->statex_.get(),
        memType,
        // ReduceScatter uses kernel reduce not NVL iput
        {CtranMapperBackend::NVL});
    verifyGpeLeak(comm->ctranComm_->ctran_.get());

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

    const auto& lastColl = pastCollsJson[0];
    EXPECT_EQ(lastColl["opName"].asString(), "ReduceScatter");
    EXPECT_EQ(lastColl["count"].asInt(), count);
    EXPECT_EQ(lastColl["algoName"].asString(), reduceScatterAlgoName(algo));

    memoryCleanUp(memType);
  }
};

class CtranReduceScatterTestParamInt
    : public CtranReduceScatterTest<int>,
      public ::testing::WithParamInterface<std::tuple<
          size_t,
          TestInPlaceType,
          bool,
          MemAllocType,
          commRedOp_t,
          enum NCCL_REDUCESCATTER_ALGO>> {};

class CtranReduceScatterTestParamFp32
    : public CtranReduceScatterTest<float>,
      public ::testing::WithParamInterface<std::tuple<
          size_t,
          TestInPlaceType,
          bool,
          MemAllocType,
          commRedOp_t,
          enum NCCL_REDUCESCATTER_ALGO>> {};

class CtranReduceScatterTestParamSpecial
    : public CtranReduceScatterTest<int>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, bool, MemAllocType>> {};

TEST_P(CtranReduceScatterTestParamFp32, Test) {
  const auto& [count, inplace, regist, memType, redOp, algo] = GetParam();
  beginTest(count, inplace, regist, memType, redOp, algo);
}

TEST_P(CtranReduceScatterTestParamInt, Test) {
  const auto& [count, inplace, regist, memType, redOp, algo] = GetParam();
  beginTest(count, inplace, regist, memType, redOp, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestRingSum) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctring;
  beginTest(count, inplace, regist, memType, commSum, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestRHDSum) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctrhd;
  beginTest(count, inplace, regist, memType, commSum, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestDirectSum) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctdirect;
  beginTest(count, inplace, regist, memType, commSum, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestRingAvg) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctring;
  beginTest(count, inplace, regist, memType, commAvg, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestRHDAvg) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctrhd;
  beginTest(count, inplace, regist, memType, commAvg, algo);
}

TEST_P(CtranReduceScatterTestParamSpecial, TestDirectAvg) {
  const auto& [count, inplace, regist, memType] = GetParam();
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctdirect;
  beginTest(count, inplace, regist, memType, commAvg, algo);
}

// common function to get test name from test parameter
inline std::string getTestName(
    const testing::TestParamInfo<CtranReduceScatterTestParamFp32::ParamType>&
        info) {
  return std::to_string(std::get<0>(info.param)) + "elements_" +
      testInPlaceTypeToStr(std::get<1>(info.param)) + "_" +
      (std::get<2>(info.param) ? "REGISTER" : "NO_REGISTER") + "_" +
      testMemAllocTypeToStr(std::get<3>(info.param)) + "_" + "_" +
      commOpToString(std::get<4>(info.param)) + "_" +
      reduceScatterAlgoName(std::get<5>(info.param));
}

inline std::string getSpecialTestName(
    const testing::TestParamInfo<CtranReduceScatterTestParamSpecial::ParamType>&
        info) {
  return std::to_string(std::get<0>(info.param)) + "elements_" +
      testInPlaceTypeToStr(std::get<1>(info.param)) + "_" +
      (std::get<2>(info.param) ? "REGISTER" : "NO_REGISTER") + "_" +
      testMemAllocTypeToStr(std::get<3>(info.param));
}

auto basicTestsValue = ::testing::Combine(
    testing::Values(8192),
    testing::Values(kTestInPlace, kTestOutOfPlace),
    testing::Values(true),
    testing::Values(kMemNcclMemAlloc),
    testing::Values(commSum, commProd, commMax, commMin, commAvg),
    testing::Values(
        NCCL_REDUCESCATTER_ALGO::ctdirect,
        NCCL_REDUCESCATTER_ALGO::ctring,
        NCCL_REDUCESCATTER_ALGO::ctrhd));

auto specialTestsValue = ::testing::Values(
    // dynamic registration
    std::make_tuple(8192, kTestInPlace, false, kMemNcclMemAlloc),
    // unaligned size
    std::make_tuple(1048567, kTestInPlace, true, kMemNcclMemAlloc),
    // larger than tmpbuf size
    std::make_tuple(33554176, kTestOutOfPlace, true, kMemNcclMemAlloc)
    // disjoint test
    // std::make_tuple(
    //     1UL << 22,
    //     kTestOutOfPlace,
    //     true,
    //     kCuMemAllocDisjoint)
);

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranReduceScatterTestParamFp32,
    basicTestsValue,
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranReduceScatterTestParamInt,
    basicTestsValue,
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranReduceScatterTestParamSpecial,
    specialTestsValue,
    getSpecialTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
