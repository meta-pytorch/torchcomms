// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "nccl.h"

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/commSpecs.h"

// Helper functions for parameterized tests to avoid code duplication
namespace {

auto getDevMemTypeValues() {
  return ::testing::Values(DevMemType::kCumem, DevMemType::kCudaMalloc);
}

auto getDevMemTypeNameGenerator() {
  return [](const ::testing::TestParamInfo<DevMemType>& info) {
    return std::string(devMemTypeStr(info.param));
  };
}

} // namespace

class IpcTest : public NcclxBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();

    // Ensure cuda driver functions have been loaded
    COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());

    if (!ctran::utils::CtranIpcSupport()) {
      GTEST_SKIP() << "CTran IPC is not supported on this platform. Skip test.";
    }

    CUDACHECK_TEST(cudaSetDevice(localRank));
    commDeprecated = createNcclComm(globalRank, numRanks, localRank);
    comm_ = commDeprecated->ctranComm_.get();

    commIpcCount = ctran::utils::getActiveIpcMemCount();
    commIpcRemCount = ctran::utils::getActiveIpcRemMemCount();
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated));
    // Check comm created IPC resources have been freed after ncclCommDestroy
    EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), 0);
    EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), 0);
    NcclxBaseTest::TearDown();
  }

 protected:
  CtranComm* comm_{nullptr};
  ncclComm_t commDeprecated{nullptr};
  const char* dummyDesc_{"dummy"};
  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      dummyDesc_,
      0,
      0};
  size_t commIpcCount{0};
  size_t commIpcRemCount{0};
  const bool kShouldSupportCudaMalloc = true;
};

class IpcAllocFreeTest : public IpcTest,
                         public ::testing::WithParamInterface<DevMemType> {};

TEST_P(IpcAllocFreeTest, AllocFree) {
  constexpr size_t size = 8192;
  DevMemType memType = GetParam();
  auto cuMemHandleType = ctran::utils::getCuMemAllocHandleType();

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem = nullptr;
  ASSERT_NO_THROW(
      ipcMem = std::make_unique<ctran::utils::CtranIpcMem>(
          size,
          this->localRank,
          &dummyLogMetaData_,
          dummyDesc_,
          memType,
          cuMemHandleType));

  ASSERT_NE(ipcMem, nullptr);
  EXPECT_GE(ipcMem->getRange(), size);
  EXPECT_NE(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Free both handle and memory
  EXPECT_EQ(ipcMem->free(), commSuccess);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);
}

INSTANTIATE_TEST_SUITE_P(
    DevMemTypes,
    IpcAllocFreeTest,
    getDevMemTypeValues(),
    getDevMemTypeNameGenerator());

class IpcLoadFreeTest : public IpcTest,
                        public ::testing::WithParamInterface<DevMemType> {};

TEST_P(IpcLoadFreeTest, LoadFree) {
  constexpr size_t size = 8192;
  DevMemType memType = GetParam();
  void* buf = nullptr;

  if (memType == DevMemType::kCumem) {
    NCCLCHECK_TEST(ncclMemAlloc(&buf, size));
  } else {
    CUDACHECK_TEST(cudaMalloc(&buf, size));
  }

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem =
      std::make_unique<ctran::utils::CtranIpcMem>(this->localRank, dummyDesc_);

  // Load an existing buffer
  bool supported = false;
  ASSERT_EQ(
      ipcMem->tryLoad(buf, size, supported, kShouldSupportCudaMalloc),
      commSuccess);
  EXPECT_TRUE(supported);
  EXPECT_GE(ipcMem->getRange(), size);
  EXPECT_NE(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Free only handle in load mode
  EXPECT_EQ(ipcMem->free(), commSuccess);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);

  // Free the buffer
  if (memType == DevMemType::kCumem) {
    EXPECT_EQ(ncclMemFree(buf), ncclSuccess);
  } else {
    EXPECT_EQ(cudaFree(buf), ncclSuccess);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DevMemTypes,
    IpcLoadFreeTest,
    getDevMemTypeValues(),
    getDevMemTypeNameGenerator());

TEST_F(IpcTest, InvalidLoadWithWrongProperty) {
  constexpr size_t size = 8192;
  constexpr bool setNcclHandleType = false;
  constexpr bool setRdmaSupport = false;
  void* buf = nullptr;

  // Single segment with insufficient property
  std::vector<TestMemSegment> segments;
  std::vector<size_t> segmentSizes(1);
  segmentSizes[0] = size;
  NCCLCHECK_TEST(ncclMemAllocDisjoint(
      &buf, segmentSizes, segments, setNcclHandleType, setRdmaSupport));

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem =
      std::make_unique<ctran::utils::CtranIpcMem>(this->localRank, dummyDesc_);

  bool supported = false;
  ASSERT_EQ(
      ipcMem->tryLoad(buf, size, supported, kShouldSupportCudaMalloc),
      commInvalidUsage);
  EXPECT_FALSE(supported);
  EXPECT_EQ(ipcMem->getRange(), 0);
  EXPECT_EQ(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);

  // Free the buffer
  ncclMemFreeDisjoint(buf, segmentSizes);
}

class IpcExportImportTest : public IpcTest,
                            public ::testing::WithParamInterface<DevMemType> {};

TEST_P(IpcExportImportTest, ExportImport) {
  constexpr size_t count = 8192;
  constexpr size_t size = count * sizeof(int);
  DevMemType memType = GetParam();

  const int rank = comm_->statex_->rank();
  const int nRanks = comm_->statex_->nRanks();
  if (nRanks < 2 || comm_->statex_->nNodes() > 1) {
    GTEST_SKIP()
        << "This test requires at least 2 ranks and only 1 node, but got nRanks="
        << nRanks << " and nNodes=" << comm_->statex_->nNodes();
  }

  // Allocate local memory
  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem = nullptr;
  ASSERT_NO_THROW(
      ipcMem = std::make_unique<ctran::utils::CtranIpcMem>(
          size, this->localRank, &dummyLogMetaData_, dummyDesc_, memType));

  // Assign value to local memory
  std::vector<int> vals(count);
  std::iota(vals.begin(), vals.end(), rank * size);
  CUDACHECK_TEST(
      cudaMemcpy(ipcMem->getBase(), vals.data(), size, cudaMemcpyHostToDevice));

  // Export
  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nRanks);
  auto res = ipcMem->ipcExport(ipcDescs[rank]);
  EXPECT_EQ(res, commSuccess);

  auto& ipcDesc = ipcDescs[rank];
  EXPECT_EQ(ipcDesc.range, ipcMem->getRange());
  EXPECT_EQ(ipcDesc.base, ipcMem->getBase());
  EXPECT_EQ(ipcDesc.pid, getpid());
  EXPECT_EQ(ipcDesc.cuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  EXPECT_EQ(ipcDesc.numSegments, 1);
  EXPECT_NE(ipcDesc.segments[0].sharedHandle.fd, 0);
  for (int i = 1; i < CTRAN_IPC_INLINE_SEGMENTS; i++) {
    EXPECT_EQ(ipcDesc.segments[i].sharedHandle.fd, 0);
  }

  intraNodeAllGather<ctran::utils::CtranIpcDesc>(commDeprecated, ipcDescs);

  // Check received descriptor
  int peerRank = (rank + 1) % nRanks;
  auto& peerIpcDesc = ipcDescs[peerRank];
  EXPECT_GE(peerIpcDesc.range, size);
  EXPECT_NE(peerIpcDesc.base, nullptr);
  EXPECT_NE(peerIpcDesc.pid, 0);
  EXPECT_EQ(peerIpcDesc.numSegments, 1);
  EXPECT_EQ(
      peerIpcDesc.cuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  EXPECT_NE(peerIpcDesc.segments[0].sharedHandle.fd, 0);
  for (int i = 1; i < CTRAN_IPC_INLINE_SEGMENTS; i++) {
    EXPECT_EQ(peerIpcDesc.segments[i].sharedHandle.fd, 0);
  }
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Import remote memory
  std::unique_ptr<ctran::utils::CtranIpcRemMem> ipcRemMem = nullptr;
  try {
    ipcRemMem = std::make_unique<ctran::utils::CtranIpcRemMem>(
        peerIpcDesc, this->localRank, &dummyLogMetaData_, dummyDesc_);
  } catch (std::exception& e) {
    GTEST_FAIL() << "Failed to import remote memory: " << e.what();
  }

  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount + 1);

  // Check access to remote memory
  std::vector<int> peerVals(count);
  std::vector<int> peerExpVals(count);
  std::iota(peerExpVals.begin(), peerExpVals.end(), peerRank * size);
  CUDACHECK_TEST(cudaMemcpy(
      peerVals.data(), ipcRemMem->getBase(), size, cudaMemcpyDeviceToHost));
  EXPECT_THAT(peerVals, ::testing::ElementsAreArray(peerExpVals));

  // Ensure the remote rank has imported the memory before local frees
  intraNodeBarrier(commDeprecated);

  // Free local memory
  EXPECT_EQ(ipcMem->free(), commSuccess);

  // Free imported remote memory
  EXPECT_EQ(ipcRemMem->release(), commSuccess);

  // Check IPC created in this test have been freed
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);
  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount);
}

INSTANTIATE_TEST_SUITE_P(
    DevMemTypes,
    IpcExportImportTest,
    getDevMemTypeValues(),
    getDevMemTypeNameGenerator());

TEST_F(IpcTest, DisjointExportImport) {
  std::vector<size_t> disjointSegmentSizes;
  int numSegments = std::min(CTRAN_IPC_INLINE_SEGMENTS, 4);
  for (int i = 0; i < numSegments; i++) {
    disjointSegmentSizes.push_back(1UL << (21 + i));
  }
  size_t size = std::accumulate(
      disjointSegmentSizes.begin(), disjointSegmentSizes.end(), 0UL);
  EXPECT_TRUE(size % sizeof(int) == 0);
  size_t count = size / sizeof(int);

  const int rank = comm_->statex_->rank();
  const int nRanks = comm_->statex_->nRanks();
  if (nRanks < 2 || comm_->statex_->nNodes() > 1) {
    GTEST_SKIP()
        << "This test requires at least 2 ranks and only 1 node, but got nRanks="
        << nRanks << " and nNodes=" << comm_->statex_->nNodes();
  }

  void* buf = nullptr;
  std::vector<TestMemSegment> segments;
  NCCLCHECK_TEST(ncclMemAllocDisjoint(&buf, disjointSegmentSizes, segments));

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem =
      std::make_unique<ctran::utils::CtranIpcMem>(this->localRank, dummyDesc_);

  // Load an existing buffer
  bool supported = false;
  ASSERT_EQ(
      ipcMem->tryLoad(buf, size, supported, kShouldSupportCudaMalloc),
      commSuccess);
  EXPECT_TRUE(supported);
  EXPECT_GE(ipcMem->getRange(), size);
  EXPECT_NE(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Assign value to local memory
  std::vector<int> vals(count);
  std::iota(vals.begin(), vals.end(), rank * size);
  CUDACHECK_TEST(
      cudaMemcpy(ipcMem->getBase(), vals.data(), size, cudaMemcpyHostToDevice));

  // Export
  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nRanks);
  auto res = ipcMem->ipcExport(ipcDescs[rank]);
  EXPECT_EQ(res, commSuccess);

  auto& ipcDesc = ipcDescs[rank];
  EXPECT_EQ(ipcDesc.memType, DevMemType::kCumem);
  EXPECT_EQ(ipcDesc.range, ipcMem->getRange());
  EXPECT_EQ(ipcDesc.base, ipcMem->getBase());
  EXPECT_EQ(ipcDesc.pid, getpid());
  EXPECT_EQ(ipcDesc.cuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  EXPECT_EQ(ipcDesc.numSegments, disjointSegmentSizes.size());
  for (int i = 0; i < ipcDesc.numSegments; i++) {
    EXPECT_NE(ipcDesc.segments[i].sharedHandle.fd, 0);
    EXPECT_EQ(ipcDesc.segments[i].range, disjointSegmentSizes[i]);
  }

  intraNodeAllGather<ctran::utils::CtranIpcDesc>(commDeprecated, ipcDescs);

  // Check received descriptor
  int peerRank = (rank + 1) % nRanks;
  auto& peerIpcDesc = ipcDescs[peerRank];
  EXPECT_GE(peerIpcDesc.range, size);
  EXPECT_NE(peerIpcDesc.base, nullptr);
  EXPECT_NE(peerIpcDesc.pid, 0);
  EXPECT_EQ(peerIpcDesc.numSegments, disjointSegmentSizes.size());
  EXPECT_EQ(
      peerIpcDesc.cuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  for (int i = 0; i < peerIpcDesc.numSegments; i++) {
    EXPECT_NE(peerIpcDesc.segments[i].sharedHandle.fd, 0);
    EXPECT_EQ(peerIpcDesc.segments[i].range, disjointSegmentSizes[i]);
  }

  // Import remote memory
  std::unique_ptr<ctran::utils::CtranIpcRemMem> ipcRemMem = nullptr;
  try {
    ipcRemMem = std::make_unique<ctran::utils::CtranIpcRemMem>(
        peerIpcDesc, this->localRank, &dummyLogMetaData_, dummyDesc_);
  } catch (std::exception& e) {
    GTEST_FAIL() << "Failed to import remote memory: " << e.what();
  }

  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount + 1);

  // Check access to remote memory
  std::vector<int> peerVals(count);
  std::vector<int> peerExpVals(count);
  std::iota(peerExpVals.begin(), peerExpVals.end(), peerRank * size);
  CUDACHECK_TEST(cudaMemcpy(
      peerVals.data(), ipcRemMem->getBase(), size, cudaMemcpyDeviceToHost));
  EXPECT_THAT(peerVals, ::testing::ElementsAreArray(peerExpVals));

  // Ensure the remote rank has imported the memory before local frees
  intraNodeBarrier(commDeprecated);

  // Free only handle in load mode
  EXPECT_EQ(ipcMem->free(), commSuccess);

  // Free the buffer
  EXPECT_EQ(ncclMemFreeDisjoint(buf, disjointSegmentSizes), ncclSuccess);

  // Free imported remote memory
  EXPECT_EQ(ipcRemMem->release(), commSuccess);
  // Check IPC created in this test have been freed
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);
  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount);
}

TEST_F(IpcTest, DisjointExportOffset) {
  size_t segmentSize = (1UL << 20) * 20;
  // Buffer spans across two segments
  int numSegments = std::min(CTRAN_IPC_INLINE_SEGMENTS, 2);
  std::vector<size_t> disjointSegmentSizes(numSegments, segmentSize);
  size_t size = std::accumulate(
      disjointSegmentSizes.begin(), disjointSegmentSizes.end(), 0UL);
  EXPECT_TRUE(size % sizeof(int) == 0);

  const int rank = comm_->statex_->rank();
  const int nRanks = comm_->statex_->nRanks();
  if (nRanks < 2 || comm_->statex_->nNodes() > 1) {
    GTEST_SKIP()
        << "This test requires at least 2 ranks and only 1 node, but got nRanks="
        << nRanks << " and nNodes=" << comm_->statex_->nNodes();
  }

  void* buf = nullptr;
  std::vector<TestMemSegment> segments;
  NCCLCHECK_TEST(ncclMemAllocDisjoint(&buf, disjointSegmentSizes, segments));

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem =
      std::make_unique<ctran::utils::CtranIpcMem>(this->localRank, dummyDesc_);

  // Load an existing buffer offset from the start of the buffer
  bool supported = false;
  void* subbufOffset = (char*)buf + (1UL << 20) * 7;
  size_t subbufSize = (1UL << 20) * 16;
  ASSERT_EQ(
      ipcMem->tryLoad(
          subbufOffset, subbufSize, supported, kShouldSupportCudaMalloc),
      commSuccess);
  EXPECT_TRUE(supported);
  EXPECT_GE(ipcMem->getRange(), size);
  EXPECT_NE(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Export
  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nRanks);
  auto res = ipcMem->ipcExport(ipcDescs[rank]);
  EXPECT_EQ(res, commSuccess);

  auto& ipcDesc = ipcDescs[rank];
  EXPECT_EQ(ipcDesc.range, ipcMem->getRange());
  EXPECT_EQ(ipcDesc.base, ipcMem->getBase());
  EXPECT_EQ(ipcDesc.pid, getpid());
  EXPECT_EQ(ipcDesc.cuMemHandleType, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  EXPECT_EQ(ipcDesc.numSegments, disjointSegmentSizes.size());
  for (int i = 0; i < ipcDesc.numSegments; i++) {
    EXPECT_NE(ipcDesc.segments[i].sharedHandle.fd, 0);
    EXPECT_EQ(ipcDesc.segments[i].range, disjointSegmentSizes[i]);
  }

  // Free only handle in load mode
  EXPECT_EQ(ipcMem->free(), commSuccess);

  // Free the buffer
  EXPECT_EQ(ncclMemFreeDisjoint(buf, disjointSegmentSizes), ncclSuccess);

  // Check IPC created in this test have been freed
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);
  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount);
}

TEST_F(IpcTest, DisjointExportTooManyPhysicalBackings) {
  std::vector<size_t> disjointSegmentSizes(
      CTRAN_IPC_INLINE_SEGMENTS + 1, 1UL << 21);
  size_t size = std::accumulate(
      disjointSegmentSizes.begin(), disjointSegmentSizes.end(), 0UL);
  EXPECT_TRUE(size % sizeof(int) == 0);

  const int rank = comm_->statex_->rank();
  const int nRanks = comm_->statex_->nRanks();
  if (nRanks < 2 || comm_->statex_->nNodes() > 1) {
    GTEST_SKIP()
        << "This test requires at least 2 ranks and only 1 node, but got nRanks="
        << nRanks << " and nNodes=" << comm_->statex_->nNodes();
  }

  void* buf = nullptr;
  std::vector<TestMemSegment> segments;
  NCCLCHECK_TEST(ncclMemAllocDisjoint(&buf, disjointSegmentSizes, segments));

  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem =
      std::make_unique<ctran::utils::CtranIpcMem>(this->localRank, dummyDesc_);

  // Load an existing buffer
  bool supported = false;
  ASSERT_EQ(
      ipcMem->tryLoad(buf, size, supported, kShouldSupportCudaMalloc),
      commSuccess);
  EXPECT_TRUE(supported);
  EXPECT_GE(ipcMem->getRange(), size);
  EXPECT_NE(ipcMem->getBase(), nullptr);
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount + 1);

  // Export
  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nRanks);
  auto res = ipcMem->ipcExport(ipcDescs[rank]);
  EXPECT_EQ(res, commInternalError);

  EXPECT_EQ(ipcMem->free(), commSuccess);
  // Check IPC created in this test have been freed
  EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), commIpcCount);
  EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), commIpcRemCount);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
