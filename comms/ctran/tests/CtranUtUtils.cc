// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranUtUtils.h"
#include "comms/ctran/tests/CtranNcclTestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

ncclComm_t CtranDistBaseTest::commWorld = NCCL_COMM_NULL;
std::unique_ptr<c10d::TCPStore> CtranDistBaseTest::tcpStoreServer = nullptr;

// Static helper instance for NCCL memory allocation
static ctran::CtranNcclTestHelpers ncclHelpers;

void CtranDistBaseTest::TearDownTestSuite() {
  LOG(INFO) << "CtranBaseTest::TearDownTestSuite: Release commWorld "
            << commWorld << " tcpStoreServer " << tcpStoreServer;
  // Clean up commWorld
  if (commWorld != NCCL_COMM_NULL) {
    const int cudaDev = commWorld->ctranComm_->statex_->rank();
    NCCLCHECK_TEST(ncclCommDestroy(commWorld));
    commWorld = NCCL_COMM_NULL;

    ctran::logGpuMemoryStats(cudaDev);
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

    // FIXME: this should be replaced with standalone ctranComm
    commWorld = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, tcpStoreServer.get());
    LOG(INFO) << "CtranBaseTest::SetUp: New commWorld " << commWorld
              << " numRanks " << numRanks << " tcpStoreServer "
              << tcpStoreServer;
  }

  // Reinitialize rank info since each test will reset the value
  ctranComm_ = commWorld->ctranComm_.get();
  numRanks = ctranComm_->statex_->nRanks();
  localRank = ctranComm_->statex_->localRank();
  globalRank = ctranComm_->statex_->rank();
  localSize = ctranComm_->statex_->nLocalRanks();

  // Reset the value of enableNolocal since each test will reset
  // the value and we set them only in NcclxBaseTest::SetUp()
  enableNolocal =
      NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal;

  ASSERT_TRUE(ctranInitialized(ctranComm_));

  if (ctranComm_->ctran_->mapper->ctranIbPtr() == nullptr &&
      ctranComm_->ctran_->mapper->ctranSockPtr() == nullptr) {
    GTEST_SKIP() << "No IB or Socket Backend found, skip test";
  }

  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Reset backends used counter
  resetBackendsUsed(ctranComm_->ctran_.get());
}

void CtranDistBaseTest::TearDown() {
  ctranComm_ = nullptr;
  finalizeNcclComm(globalRank, tcpStoreServer.get());
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

void* CtranBaseTest::prepareBuf(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments,
    size_t numSegments) {
  // Delegate to CtranNcclTestHelpers for all memory types
  return ncclHelpers.prepareBuf(bufSize, memType, segments, numSegments);
}

void CtranBaseTest::releaseBuf(
    void* buf,
    size_t bufSize,
    MemAllocType memType,
    size_t numSegments) {
  // Delegate to CtranNcclTestHelpers for all memory types
  ncclHelpers.releaseBuf(buf, bufSize, memType, numSegments);
}

void CtranBaseTest::allocDevArg(const size_t nbytes, void*& ptr) {
  CUDACHECK_ASSERT(cudaMalloc(&ptr, nbytes));
  // store argPtr to release at the end of test
  devArgs_.insert(ptr);
}

void CtranBaseTest::releaseDevArgs() {
  for (auto ptr : devArgs_) {
    CUDACHECK_TEST(cudaFree(ptr));
  }
  devArgs_.clear();
}

void CtranBaseTest::releaseDevArg(void* ptr) {
  cudaFree(ptr);
  devArgs_.erase(ptr);
}
