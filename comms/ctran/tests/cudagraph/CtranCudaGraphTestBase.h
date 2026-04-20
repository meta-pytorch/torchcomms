// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>
#include <vector>

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranNcclTestUtils.h"
#include "comms/ctran/tests/cudagraph/CudaGraphTestBuilder.h"
#include "comms/utils/test_utils/CudaGraphTestUtils.h"

class CtranCudaGraphEnvironment : public ctran::CtranDistEnvironment {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ALLOW_CUDA_GRAPH", "1", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
#endif
#ifdef NCCL_COMM_STATE_DEBUG_TOPO_VNODE
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "vnode", 1);
#endif
#ifdef TEST_FAST_INIT_MODE_NONE
    setenv("NCCL_FASTINIT_MODE", "none", 1);
#endif
    ctran::CtranDistEnvironment::SetUp();
  }
};

class CtranCudaGraphTestBase : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }

 public:
  static constexpr size_t kDefaultCount = 1024;

  void waitAndVerifyGpeClean(CtranComm* comm) {
    ASSERT_NE(comm->ctran_, nullptr);
    constexpr int kMaxSpinMs = 5000;
    constexpr int kSpinIntervalMs = 10;
    int elapsedMs = 0;
    while (elapsedMs < kMaxSpinMs) {
      if (comm->ctran_->gpe->numInUseKernelFlags() == 0 &&
          comm->ctran_->gpe->numInUseKernelElems() == 0 &&
          comm->ctran_->gpe->numInUseChecksums() == 0 &&
          comm->ctran_->gpe->numInUseGpeKernelSyncs() == 0) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kSpinIntervalMs));
      elapsedMs += kSpinIntervalMs;
    }
    EXPECT_EQ(comm->ctran_->gpe->numInUseKernelFlags(), 0)
        << "KernelFlag leak after graph destroy";
    EXPECT_EQ(comm->ctran_->gpe->numInUseKernelElems(), 0)
        << "KernelElem leak after graph destroy";
    EXPECT_EQ(comm->ctran_->gpe->numInUseChecksums(), 0)
        << "Checksum leak after graph destroy";
    EXPECT_EQ(comm->ctran_->gpe->numInUseGpeKernelSyncs(), 0)
        << "GpeKernelSync leak after graph destroy";
  }

  static void fillSendBuf(void* buf, size_t count, int32_t val) {
    std::vector<int32_t> h(count, val);
    CUDACHECK_TEST(
        cudaMemcpy(buf, h.data(), count * sizeof(int32_t), cudaMemcpyDefault));
  }

  void verifyAllGather(const void* recvbuf, size_t sendcount, int nRanks) {
    size_t totalCount = sendcount * nRanks;
    std::vector<int32_t> h(totalCount);
    CUDACHECK_TEST(cudaMemcpy(
        h.data(), recvbuf, totalCount * sizeof(int32_t), cudaMemcpyDefault));
    for (int r = 0; r < nRanks; ++r) {
      for (size_t i = 0; i < sendcount; ++i) {
        ASSERT_EQ(h[r * sendcount + i], r)
            << "AllGather mismatch at rank segment " << r << " index " << i;
      }
    }
  }

  void verifyAllReduce(const void* recvbuf, size_t count, int nRanks) {
    std::vector<int32_t> h(count);
    CUDACHECK_TEST(cudaMemcpy(
        h.data(), recvbuf, count * sizeof(int32_t), cudaMemcpyDefault));
    int32_t expected = nRanks * (nRanks - 1) / 2; // sum of 0..nRanks-1
    for (size_t i = 0; i < count; ++i) {
      ASSERT_EQ(h[i], expected) << "AllReduce mismatch at index " << i;
    }
  }

  void verifyReduceScatter(
      const void* recvbuf,
      size_t recvcount,
      int rank,
      int nRanks) {
    std::vector<int32_t> h(recvcount);
    CUDACHECK_TEST(cudaMemcpy(
        h.data(), recvbuf, recvcount * sizeof(int32_t), cudaMemcpyDefault));
    int32_t expected = nRanks * (nRanks - 1) / 2;
    for (size_t i = 0; i < recvcount; ++i) {
      ASSERT_EQ(h[i], expected)
          << "ReduceScatter mismatch at index " << i << " on rank " << rank;
    }
  }

  // AllToAll with uniform counts: same layout as AllGather (each rank's
  // segment filled with that rank's value).
  void verifyAllToAll(const void* recvbuf, size_t countPerRank, int nRanks) {
    verifyAllGather(recvbuf, countPerRank, nRanks);
  }

  // AllToAllv with variable counts/displacements: recvbuf at
  // rdispls[p]..rdispls[p]+recvcounts[p]-1 should equal p (the sender's rank).
  void verifyAllToAllv(
      const void* recvbuf,
      const size_t* recvcounts,
      const size_t* rdispls,
      int nRanks) {
    size_t totalRecv = 0;
    for (int p = 0; p < nRanks; ++p) {
      totalRecv = std::max(totalRecv, rdispls[p] + recvcounts[p]);
    }
    std::vector<int32_t> h(totalRecv);
    CUDACHECK_TEST(cudaMemcpy(
        h.data(), recvbuf, totalRecv * sizeof(int32_t), cudaMemcpyDefault));
    for (int p = 0; p < nRanks; ++p) {
      for (size_t i = 0; i < recvcounts[p]; ++i) {
        ASSERT_EQ(h[rdispls[p] + i], p)
            << "AllToAllv mismatch from peer " << p << " index " << i;
      }
    }
  }

  // Returns a GraphAssertionsFn that verifies each captured graph has at least
  // the expected number of HOST and KERNEL nodes.
  static ctran::testing::GraphAssertionsFn expectGraphNodes(
      size_t minHostNodes = 1,
      size_t minKernelNodes = 1) {
    return [=](const std::vector<GraphTopology>& topos) {
      for (size_t i = 0; i < topos.size(); ++i) {
        EXPECT_GE(
            topos[i].nodesOfType(cudaGraphNodeTypeHost).size(), minHostNodes)
            << "Graph " << i << ": expected at least " << minHostNodes
            << " HOST node(s)";
        EXPECT_GE(
            topos[i].nodesOfType(cudaGraphNodeTypeKernel).size(),
            minKernelNodes)
            << "Graph " << i << ": expected at least " << minKernelNodes
            << " KERNEL node(s)";
      }
    };
  }

  void verifyBroadcast(const void* recvbuf, size_t count, int32_t rootVal) {
    std::vector<int32_t> h(count);
    CUDACHECK_TEST(cudaMemcpy(
        h.data(), recvbuf, count * sizeof(int32_t), cudaMemcpyDefault));
    for (size_t i = 0; i < count; ++i) {
      ASSERT_EQ(h[i], rootVal) << "Broadcast mismatch at index " << i;
    }
  }
};
