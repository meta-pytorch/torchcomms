// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranUtUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

ncclComm_t CtranDistBaseTest::commWorld = NCCL_COMM_NULL;
std::unique_ptr<c10d::TCPStore> CtranDistBaseTest::tcpStoreServer = nullptr;

void CtranDistBaseTest::TearDownTestSuite() {
  LOG(INFO) << "CtranBaseTest::TearDownTestSuite: Release commWorld "
            << commWorld << " tcpStoreServer " << tcpStoreServer;
  // Clean up commWorld
  if (commWorld != NCCL_COMM_NULL) {
    const int cudaDev = commWorld->ctranComm_->statex_->rank();
    NCCLCHECK_TEST(ncclCommDestroy(commWorld));
    commWorld = NCCL_COMM_NULL;

    logGpuMemoryStats(cudaDev);
  }

  // Reset tcpStore server
  if (tcpStoreServer) {
    tcpStoreServer.reset();
  }
}

void CtranDistBaseTest::SetUp() {
  setenv("NCCL_CTRAN_PROFILING", "none", 1);
  setenv("NCCL_DEBUG", "WARN", 0);
  setenv("NCCL_CTRAN_ENABLE", "1", 0);
  setenv("NCCL_COLLTRACE", "trace", 0);
  setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);

  // Create single tcpStore and commWorld shared by all tests running in
  // this test suite.
  if (commWorld == NCCL_COMM_NULL) {
    NcclxBaseTest::SetUp();
    // Handover tcpStore server to CtranBaseTest so that we control to release
    // it only at global TearDownTestSuite()
    if (server) {
      tcpStoreServer = std::move(server);
    }

    commWorld = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, tcpStoreServer.get());
    LOG(INFO) << "CtranBaseTest::SetUp: New commWorld " << commWorld
              << " numRanks " << numRanks << " tcpStoreServer "
              << tcpStoreServer;
  }

  // Reinitialize rank info since each test will reset the value
  numRanks = commWorld->ctranComm_->statex_->nRanks();
  localRank = commWorld->ctranComm_->statex_->localRank();
  globalRank = commWorld->ctranComm_->statex_->rank();
  localSize = commWorld->ctranComm_->statex_->nLocalRanks();

  // Reset the value of enableNolocal since each test will reset
  // the value and we set them only in NcclxBaseTest::SetUp()
  enableNolocal =
      NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal;

  ASSERT_TRUE(ctranInitialized(commWorld->ctranComm_.get()));

  if (commWorld->ctranComm_->ctran_->mapper->ctranIbPtr() == nullptr &&
      commWorld->ctranComm_->ctran_->mapper->ctranSockPtr() == nullptr) {
    GTEST_SKIP() << "No IB or Socket Backend found, skip test";
  }

  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Reset backends used counter
  resetBackendsUsed(commWorld->ctranComm_->ctran_.get());
}

void CtranDistBaseTest::TearDown() {
  finalizeNcclComm(globalRank, tcpStoreServer.get());
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

void* CtranBaseTest::prepareBuf(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments) {
  void* buf = nullptr;
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    segments.emplace_back(buf, bufSize);
  } else if (memType == kMemNcclMemAlloc) {
    NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    segments.emplace_back(buf, bufSize);
  } else {
    std::vector<size_t> disjointSegmentSizes(2);
    disjointSegmentSizes[0] = bufSize / 2;
    disjointSegmentSizes[1] = bufSize / 2;
    NCCLCHECK_TEST(ncclMemAllocDisjoint(&buf, disjointSegmentSizes, segments));
  }
  return buf;
}

void CtranBaseTest::releaseBuf(
    void* buf,
    size_t bufSize,
    MemAllocType memType) {
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaFree(buf));
  } else if (memType == kMemNcclMemAlloc) {
    NCCLCHECK_TEST(ncclMemFreeWithRefCheck(buf));
  } else {
    std::vector<size_t> disjointSegmentSizes(2);
    disjointSegmentSizes[0] = bufSize / 2;
    disjointSegmentSizes[1] = bufSize / 2;
    NCCLCHECK_TEST(ncclMemFreeDisjoint(buf, disjointSegmentSizes));
  }
}
