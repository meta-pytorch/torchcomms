// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTBase.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/wrapper/MetaFactory.h"

void CtranDistAlgoDevTest::SetUp() {
  // Require EAGER load to support concurrent kernels on two streasm since
  // cuda 12. Otherwise test may hang.
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  setenv("NCCL_CTRAN_ENABLE", "1", 0);

  NcclxBaseTest::SetUp();
  comm_ = createNcclComm(globalRank, numRanks, localRank);

  CUDACHECK_TEST(cudaSetDevice(localRank));

  ASSERT_NE(nullptr, comm_);
  ASSERT_TRUE(ctranInitialized(comm_->ctranComm_.get()));

  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  const int localRanks = comm_->ctranComm_->statex_->nLocalRanks();

  if (nLocalRanks < 2 || localRanks != nLocalRanks) {
    GTEST_SKIP()
        << "Skip test because it requires all ranks on the same node, but got "
        << "nLocalRanks=" << nLocalRanks << ", localRanks=" << localRanks;
  }
}

void CtranDistAlgoDevTest::TearDown() {
  NCCLCHECK_TEST(ncclCommDestroy(comm_));
  NcclxBaseTest::TearDown();
}

template <typename T>
void CtranDistAlgoDevTest::assignVal(
    void* buf,
    size_t count,
    T seedVal,
    bool inc) {
  if (inc) {
    std::vector<T> vals(count);
    std::iota(std::begin(vals), std::end(vals), seedVal);
    CUDACHECK_TEST(cudaMemcpy(
        buf, vals.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  } else {
    CUDACHECK_TEST(cudaMemset(buf, seedVal, count * sizeof(T)));
  }
}

template <typename T>
void CtranDistAlgoDevTest::initIpcBufs(size_t srcCount, size_t dstCount) {
  // Allocate local memory
  NCCLCHECK_TEST(ncclMemAlloc(&localBuf_, dstCount * sizeof(T)));
  ASSERT_NO_THROW(
      ipcMem_ = std::make_unique<ctran::utils::CtranIpcMem>(
          srcCount * sizeof(T), localRank, &dummyLogMetaData_, "Test"));
  ipcBuf_ = ipcMem_->getBase();

  // Export recvBuf
  int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  int localRank = comm_->ctranComm_->statex_->localRank();

  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nLocalRanks);
  COMMCHECK_TEST(ipcMem_->ipcExport(ipcDescs[localRank]));

  // Exchange with the other ranks on the same node.
  // (SetUp already checked all ranks on the same node)
  auto resFuture = comm_->ctranComm_->bootstrap_->allGatherIntraNode(
      ipcDescs.data(),
      sizeof(ctran::utils::CtranIpcDesc),
      localRank,
      nLocalRanks,
      comm_->ctranComm_->statex_->localRankToRanks());
  COMMCHECK_TEST(static_cast<commResult_t>(std::move(resFuture).get()));

  // Import remote recvBuf from all other peers
  ipcRemMem_.resize(nLocalRanks);
  try {
    for (int peer = 0; peer < nLocalRanks; peer++) {
      if (peer == localRank) {
        continue;
      }
      ipcRemMem_[peer] = std::make_unique<ctran::utils::CtranIpcRemMem>(
          ipcDescs[peer], localRank, &dummyLogMetaData_, "Test");
    }
  } catch (std::exception& e) {
    GTEST_FAIL() << "Failed to import remote memory: " << e.what();
  }
}

template <typename T>
void CtranDistAlgoDevTest::checkVals(size_t count, T seedVal, size_t offset) {
  size_t nBytes = count * sizeof(T);
  std::vector<T> recvVals(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      recvVals.data(),
      (char*)ipcBuf_ + offset,
      nBytes,
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> expVals(count);
  std::iota(expVals.begin(), expVals.end(), seedVal);
  EXPECT_THAT(recvVals, ::testing::ElementsAreArray(expVals))
      << " compared with seedVal " << seedVal << " at ipcBuf_ " << ipcBuf_
      << " offset " << offset << " with count " << count << " on rank "
      << globalRank;
}

void CtranDistAlgoDevTest::freeIpcBufs() {
  COMMCHECK_TEST(ipcMem_->free());

  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  const int localRank = comm_->ctranComm_->statex_->localRank();
  for (int peer = 0; peer < nLocalRanks; peer++) {
    if (peer != localRank) {
      COMMCHECK_TEST(ipcRemMem_.at(peer)->release());
    }
  }
  NCCLCHECK_TEST(ncclMemFree(localBuf_));
}

// TODO: add more types when needed
DECLAR_ALGO_UT_FUNCS(int);
