// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <cstddef>
#include <list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "checks.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/algoconf/AlgoConfig.h"
#include "meta/hints/GlobalHints.h"

#include "meta/wrapper/MetaFactory.h"

// Define a struct to hold all test parameters
struct AllReduceTestParams {
  size_t count;
  ncclDataType_t dataType;
  bool enableDequant;

  // For test name generation
  friend std::ostream& operator<<(
      std::ostream& os,
      const AllReduceTestParams& params) {
    return os << "Count_" << params.count << "_Type_" << params.dataType
              << (params.enableDequant ? "_WithDequant" : "");
  }
};

class AllReduceTest
    : public NcclxBaseTest,
      public ::testing::WithParamInterface<AllReduceTestParams> {
 public:
  AllReduceTest() = default;
  void SetUp() override {
    // registry the user buffer as NCCL_MANAGED memory. It enable the nvl
    // intra-node connection
    NcclxBaseTest::SetUp();

    comm = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    finalizeNcclComm(globalRank, server.get());
    NcclxBaseTest::TearDown();
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(
      T* buf,
      size_t count,
      std::vector<T> vals,
      ncclDataType_t dataType) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;

    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != vals[i]) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %f, expectedVal = %f\n",
              this->globalRank,
              i,
              static_cast<float>(observedVals[i]),
              static_cast<float>(vals[i]));
        }
        errs++;
      }
    }
    return errs;
  }

  template <typename T>
  T testValue(const size_t offset, const int rep, const int rank) {
    uint8_t v = (rank + rep);
    return (T)v;
  }
  template <>
  float testValue<float>(const size_t offset, const int rep, const int rank) {
    return (float)testValue<int>(offset, rep, rank);
  }
  template <>
  __device__ __host__ __nv_bfloat16
  testValue<__nv_bfloat16>(const size_t offset, const int rep, const int rank) {
    return __float2bfloat16(testValue<float>(offset, rep, rank));
  }

  template <typename T>
  void run(
      enum NCCL_ALLREDUCE_ALGO algo,
      ncclDataType_t dataType,
      bool enableDequant,
      size_t count) {
    auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, algo);
    T *sendBuf = nullptr, *recvBuf = nullptr;

    // create and register buffers
    // constexpr int count = 1 << 24;
    FB_COMMCHECKTHROW(
        ncclToMetaComm(ncclMemAlloc((void**)&sendBuf, count * sizeof(T))));
    FB_COMMCHECKTHROW(
        ncclToMetaComm(ncclMemAlloc((void**)&recvBuf, count * sizeof(T))));
    assignChunkValue<T>(sendBuf, count, this->globalRank);

    void *sendHandle = nullptr, *recvHandle = nullptr;
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, count * sizeof(T), &sendHandle));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, recvBuf, count * sizeof(T), &recvHandle));

    if (enableDequant) {
      auto envGuard =
          EnvRAII(NCCL_ALLREDUCE_TYPE, NCCL_ALLREDUCE_TYPE::ncclFloat32);
    }

    for (int i = 0; i < 1; i++) {
      ncclResult_t res;
      res = ncclAllReduce(
          sendBuf, recvBuf, count, dataType, ncclSum, comm, stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    // Check results
    std::vector<T> expectedVals(count);
    for (int i = 0; i < count; i++) {
      T expectedVal = 0;
      for (int r = 0; r < this->numRanks; r++) {
        expectedVal += testValue<T>(i, 0, r);
      }
      expectedVals[i] = expectedVal;
    }

    int errs = checkChunkValue<T>(recvBuf, count, expectedVals, dataType);
    EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " at " << recvBuf
                       << " with " << errs << " errors";

    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    NCCLCHECK_TEST(ncclMemFree(sendBuf));
    NCCLCHECK_TEST(ncclMemFree(recvBuf));
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_P(AllReduceTest, DISABLED_AllToAll) {
  const auto& params = GetParam();

  if (params.dataType == ncclInt32) {
    run<int>(
        NCCL_ALLREDUCE_ALGO::ctarg,
        params.dataType,
        params.enableDequant,
        params.count);
  } else if (params.dataType == ncclBfloat16) {
    run<__nv_bfloat16>(
        NCCL_ALLREDUCE_ALGO::ctarg,
        params.dataType,
        params.enableDequant,
        params.count);
  }
}

// Instantiate the test suite with different parameter combinations
std::list<AllReduceTestParams> createTestParams() {
  std::list<AllReduceTestParams> params;
  // Int32 tests without dequantization
  params.push_back(AllReduceTestParams{1, ncclInt32, false});
  params.push_back(AllReduceTestParams{2, ncclInt32, false});
  params.push_back(AllReduceTestParams{64, ncclInt32, false});
  params.push_back(AllReduceTestParams{(1 << 10) - 1, ncclInt32, false});
  params.push_back(AllReduceTestParams{1 << 20, ncclInt32, false});
  params.push_back(AllReduceTestParams{1 << 25, ncclInt32, false});
  params.push_back(
      AllReduceTestParams{1024 * 1024 * 1024 + 17, ncclInt32, false});

  // BFloat16 tests without dequantization
  params.push_back(AllReduceTestParams{64, ncclBfloat16, false});

  // BFloat16 tests with dequantization
  params.push_back(AllReduceTestParams{2, ncclBfloat16, true});
  params.push_back(AllReduceTestParams{64, ncclBfloat16, true});
  params.push_back(AllReduceTestParams{(1 << 10) - 1, ncclBfloat16, true});

  return params;
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceTests,
    AllReduceTest,
    ::testing::ValuesIn(createTestParams()));

class AllReduceHintOverrideTest : public AllReduceTest {};

TEST_F(AllReduceHintOverrideTest, TestWithHintOverride_orig) {
  enum NCCL_ALLREDUCE_ALGO algo = NCCL_ALLREDUCE_ALGO::orig;
  std::string hintVal = ncclx::algoconf::getAlgoHintValue(algo);
  ASSERT_TRUE(ncclx::setGlobalHint("algo_allreduce", hintVal.c_str()));
  run<int>(algo, ncclInt32, false, 1024);
  ASSERT_TRUE(ncclx::resetGlobalHint("algo_allreduce"));
}

TEST_F(AllReduceHintOverrideTest, TestWithHintOverride_ctdirect) {
  enum NCCL_ALLREDUCE_ALGO algo = NCCL_ALLREDUCE_ALGO::ctdirect;
  std::string hintVal = ncclx::algoconf::getAlgoHintValue(algo);
  ASSERT_TRUE(ncclx::setGlobalHint("algo_allreduce", hintVal.c_str()));
  run<int>(algo, ncclInt32, false, 1024);
  ASSERT_TRUE(ncclx::resetGlobalHint("algo_allreduce"));
}

TEST_F(AllReduceHintOverrideTest, TestWithHintOverride_null) {
  enum NCCL_ALLREDUCE_ALGO algo = NCCL_ALLREDUCE_ALGO::ctdirect;
  run<int>(algo, ncclInt32, false, 1024);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
