// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <optional>

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ibverbx::createAddressHandle;
using ibverbx::createDCI;
using ibverbx::createDCT;
using ibverbx::createSRQ;
using ibverbx::DC_KEY;
using ibverbx::DcBusinessCard;
using ibverbx::kGidIndex;
using ibverbx::kPortNum;
using ibverbx::pollCqForCompletions;
using ibverbx::transitionDCIToRts;
using ibverbx::transitionDCTToRtr;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

class DcMultiRankTestFixture : public meta::comms::MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
    ASSERT_TRUE(ibverbx::ibvInit());

    CUDA_CHECK(cudaSetDevice(localRank));
    CUDA_CHECK(cudaGetDevice(&myDevId_));

    // Get device and allocate PD
    auto devices =
        ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
    ASSERT_TRUE(devices);
    devices_.emplace(std::move(*devices));
    auto& device = devices_->at(myDevId_);

    auto pd = device.allocPd();
    ASSERT_TRUE(pd);
    pd_.emplace(std::move(*pd));

    // Create CQ and SRQ
    auto cq = device.createCq(kCqe, nullptr, nullptr, 0);
    ASSERT_TRUE(cq);
    cq_.emplace(std::move(*cq));

    auto srqResult = createSRQ(*pd_, kSrqMaxWr);
    ASSERT_TRUE(srqResult) << "Failed to create SRQ: "
                           << srqResult.error().errStr;
    srq_.emplace(std::move(*srqResult));

    // Create DCI and DCT
    auto dciResult = createDCI(*pd_, *cq_);
    auto dctResult = createDCT(*pd_, *cq_, *srq_);

    // Check if QP creation succeeded across all ranks
    int localSuccess = (dciResult.hasValue() && dctResult.hasValue()) ? 1 : 0;
    int globalSuccess = 0;
    MPI_CHECK(MPI_Allreduce(
        &localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    if (globalSuccess == 0) {
      GTEST_SKIP() << "DC QP creation failed on one or more ranks";
    }

    dci_.emplace(std::move(*dciResult));
    dct_.emplace(std::move(*dctResult));

    // Get extended QP interfaces for DC send operations
    exQp_ = ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dci_->qp());
    dvQp_ = ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp_);
    ASSERT_NE(exQp_, nullptr);
    ASSERT_NE(dvQp_, nullptr);

    // Transition QPs to operational states
    auto dciTransitionResult =
        transitionDCIToRts(*dci_, kPortNum, ibverbx::IBV_MTU_4096);
    ASSERT_TRUE(dciTransitionResult) << "Failed to transition DCI to RTS: "
                                     << dciTransitionResult.error().errStr;
    auto dctTransitionResult =
        transitionDCTToRtr(*dct_, kPortNum, ibverbx::IBV_MTU_4096);
    ASSERT_TRUE(dctTransitionResult) << "Failed to transition DCT to RTR: "
                                     << dctTransitionResult.error().errStr;

    // Query GID
    auto gid = device.queryGid(kPortNum, kGidIndex);
    ASSERT_TRUE(gid);
    gid_ = *gid;
  }

  ibverbx::IbvDevice& device() {
    return devices_->at(myDevId_);
  }

  static constexpr int kCqe = 2048;
  static constexpr int kSrqMaxWr = 2048;

  int myDevId_{-1};
  std::optional<std::vector<ibverbx::IbvDevice>> devices_;
  std::optional<ibverbx::IbvPd> pd_;
  std::optional<ibverbx::IbvCq> cq_;
  std::optional<ibverbx::IbvSrq> srq_;
  std::optional<ibverbx::IbvQp> dci_;
  std::optional<ibverbx::IbvQp> dct_;
  ibverbx::ibv_qp_ex* exQp_{nullptr};
  struct mlx5dv_qp_ex* dvQp_{nullptr};
  ibverbx::ibv_gid gid_{};
};

// Ring communication test: each rank sends to (rank+1) % numRanks
TEST_F(DcMultiRankTestFixture, RingRdmaWrite) {
  // Allocate and initialize device buffer
  size_t devBufSize = 1024 * 1024; // 1MB
  void* devBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));
  size_t numElements = devBufSize / sizeof(int64_t);
  std::vector<int64_t> hostBuf(numElements);

  // Initialize with zeros (will be overwritten by sender)
  std::fill(hostBuf.begin(), hostBuf.end(), 0);
  CUDA_CHECK(cudaMemcpy(devBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Register memory
  auto access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);
  auto mr = pd_->regMr(devBuf, devBufSize, access);
  ASSERT_TRUE(mr);

  // Create and exchange business cards
  DcBusinessCard localCard = {
      .mtu = 5, // IBV_MTU_4096 = 5
      .dctNum = dct_->qp()->qp_num,
      .port = kPortNum,
      .subnetPrefix = gid_.global.subnet_prefix,
      .interfaceId = gid_.global.interface_id,
      .rank = globalRank,
      .remoteAddr = reinterpret_cast<uint64_t>(devBuf),
      .rkey = mr->mr()->rkey,
  };

  std::vector<DcBusinessCard> cards(numRanks);
  MPI_CHECK(MPI_Allgather(
      &localCard,
      sizeof(DcBusinessCard),
      MPI_BYTE,
      cards.data(),
      sizeof(DcBusinessCard),
      MPI_BYTE,
      MPI_COMM_WORLD));

  for (int i = 0; i < numRanks; i++) {
    const auto& card = cards.at(i);
    XLOG(INFO) << "rank " << globalRank << ": got card " << card;
  }

  // Send to next rank, receive from previous rank
  int sendToRank = (globalRank + 1) % numRanks;
  int recvFromRank = (globalRank + numRanks - 1) % numRanks;
  const auto& targetCard = cards.at(sendToRank);

  XLOGF(
      INFO,
      "rank {}: sending to rank {}, receiving from rank {}",
      globalRank,
      sendToRank,
      recvFromRank);

  // Synchronize before posting work
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Prepare data to send: fill with sender's rank as pattern
  std::fill(hostBuf.begin(), hostBuf.end(), static_cast<int64_t>(globalRank));
  void* sendBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&sendBuf, devBufSize));
  CUDA_CHECK(
      cudaMemcpy(sendBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto sendMr = pd_->regMr(sendBuf, devBufSize, access);
  ASSERT_TRUE(sendMr);

  // Create address handle for target DCT
  auto ahResult = createAddressHandle(*pd_, targetCard);
  ASSERT_TRUE(ahResult) << "Failed to create Address Handle: "
                        << ahResult.error().errStr;
  auto ah = std::move(*ahResult);

  // Post RDMA write to next rank
  ibverbx::ibv_sge sendSge{};
  sendSge.addr = reinterpret_cast<uint64_t>(sendBuf);
  sendSge.length = static_cast<uint32_t>(devBufSize);
  sendSge.lkey = sendMr->mr()->lkey;

  ibverbx::ibvSymbols.ibv_internal_wr_start(exQp_);
  exQp_->wr_id = static_cast<uint64_t>(sendToRank);
  exQp_->wr_flags = ibverbx::IBV_SEND_SIGNALED;
  ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(
      exQp_, targetCard.rkey, targetCard.remoteAddr);
  ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp_, 1, &sendSge);
  ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
      dvQp_, ah.ah(), targetCard.dctNum, DC_KEY);
  int ret = ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp_);
  ASSERT_EQ(ret, 0) << "Failed to post DC RDMA write";

  XLOGF(INFO, "Rank {}: Posted RDMA write to rank {}", globalRank, sendToRank);

  // Wait for send completion only (plain RDMA write has no receiver completion)
  auto pollResult = pollCqForCompletions(globalRank, *cq_, 1);
  ASSERT_TRUE(pollResult) << "CQ poll failed: " << pollResult.error().errStr;

  // Synchronize to ensure all writes have landed before verification
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data - should be filled with recvFromRank's value
  std::vector<int64_t> recvBuf(numElements);
  CUDA_CHECK(cudaMemcpy(recvBuf.data(), devBuf, devBufSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int64_t> expectedBuf(
      numElements, static_cast<int64_t>(recvFromRank));
  ASSERT_EQ(recvBuf, expectedBuf)
      << "Data mismatch: expected data from rank " << recvFromRank;

  XLOGF(
      INFO,
      "Rank {}: Ring RDMA write test passed - verified data from rank {}",
      globalRank,
      recvFromRank);

  // Cleanup - RAII handles ibverbx objects
  CUDA_CHECK(cudaFree(sendBuf));
  CUDA_CHECK(cudaFree(devBuf));
}

// All-to-all communication test: each rank sends to all other ranks
TEST_F(DcMultiRankTestFixture, AllToAllRdmaWrite) {
  // Allocate receive buffer - one slot per sender
  size_t slotSize = 1024; // 1KB per sender
  size_t totalRecvSize = slotSize * numRanks;
  void* recvBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&recvBuf, totalRecvSize));

  // Initialize receive buffer to zeros
  CUDA_CHECK(cudaMemset(recvBuf, 0, totalRecvSize));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Allocate send buffer
  void* sendBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&sendBuf, slotSize));

  // Fill send buffer with sender's rank
  size_t numElements = slotSize / sizeof(int64_t);
  std::vector<int64_t> hostSendBuf(numElements);
  std::fill(
      hostSendBuf.begin(), hostSendBuf.end(), static_cast<int64_t>(globalRank));
  CUDA_CHECK(
      cudaMemcpy(sendBuf, hostSendBuf.data(), slotSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Register memory
  auto access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);
  auto recvMr = pd_->regMr(recvBuf, totalRecvSize, access);
  ASSERT_TRUE(recvMr);
  auto sendMr = pd_->regMr(sendBuf, slotSize, access);
  ASSERT_TRUE(sendMr);

  // Create and exchange business cards
  DcBusinessCard localCard = {
      .mtu = 5, // IBV_MTU_4096 = 5
      .dctNum = dct_->qp()->qp_num,
      .port = kPortNum,
      .subnetPrefix = gid_.global.subnet_prefix,
      .interfaceId = gid_.global.interface_id,
      .rank = globalRank,
      .remoteAddr = reinterpret_cast<uint64_t>(recvBuf),
      .rkey = recvMr->mr()->rkey,
  };

  std::vector<DcBusinessCard> cards(numRanks);
  MPI_CHECK(MPI_Allgather(
      &localCard,
      sizeof(DcBusinessCard),
      MPI_BYTE,
      cards.data(),
      sizeof(DcBusinessCard),
      MPI_BYTE,
      MPI_COMM_WORLD));

  // Create address handles for all remote DCTs
  std::vector<ibverbx::IbvAh> addressHandles;
  addressHandles.reserve(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    if (i != globalRank) {
      auto ahResult = createAddressHandle(*pd_, cards.at(i));
      ASSERT_TRUE(ahResult) << "Failed to create AH for rank " << i;
      addressHandles.emplace_back(std::move(*ahResult));
    } else {
      // Placeholder for self - create a dummy AH that won't be used
      addressHandles.emplace_back();
    }
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  for (int i = 0; i < numRanks; i++) {
    if (i == globalRank) {
      continue;
    }

    int recvFromRank = i;
    int sendToRank = i;
    XLOG(INFO) << "rank " << globalRank << ": sending to rank " << sendToRank
               << ", receiving from rank " << recvFromRank;

    const auto& card = cards.at(i);
    XLOG(INFO) << "rank " << globalRank << ": got card " << card;

    // Each sender writes to their designated slot in the receiver's buffer
    uint64_t targetAddr = card.remoteAddr + (globalRank * slotSize);

    // Post RDMA write to target rank's designated slot
    ibverbx::ibv_sge sendSge{};
    sendSge.addr = reinterpret_cast<uint64_t>(sendBuf);
    sendSge.length = static_cast<uint32_t>(slotSize);
    sendSge.lkey = sendMr->mr()->lkey;

    ibverbx::ibvSymbols.ibv_internal_wr_start(exQp_);
    exQp_->wr_id = static_cast<uint64_t>(sendToRank);
    exQp_->wr_flags = ibverbx::IBV_SEND_SIGNALED;
    ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(
        exQp_, card.rkey, targetAddr);
    ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp_, 1, &sendSge);
    ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
        dvQp_, addressHandles[i].ah(), card.dctNum, DC_KEY);
    int ret = ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp_);
    ASSERT_EQ(ret, 0) << "Failed to post DC RDMA write to rank "
                      << recvFromRank;
  }

  XLOGF(INFO, "Rank {}: Posted RDMA write to all other ranks", globalRank);

  // Wait for send completions only (plain RDMA write has no receiver
  // completion)
  int expectedCompletions = numRanks - 1;
  auto pollResult = pollCqForCompletions(globalRank, *cq_, expectedCompletions);
  ASSERT_TRUE(pollResult) << "CQ poll failed: " << pollResult.error().errStr;

  // Synchronize to ensure all writes have landed before verification
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data from each sender
  std::vector<int64_t> recvHostBuf(totalRecvSize / sizeof(int64_t));
  CUDA_CHECK(cudaMemcpy(
      recvHostBuf.data(), recvBuf, totalRecvSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Construct expected buffer and compare as a whole
  std::vector<int64_t> expectedBuf(totalRecvSize / sizeof(int64_t));
  for (int recvFromRank = 0; recvFromRank < numRanks; recvFromRank++) {
    int64_t expectedValue =
        (recvFromRank == globalRank) ? 0 : static_cast<int64_t>(recvFromRank);
    std::fill(
        expectedBuf.begin() + recvFromRank * numElements,
        expectedBuf.begin() + (recvFromRank + 1) * numElements,
        expectedValue);
  }
  ASSERT_EQ(recvHostBuf, expectedBuf)
      << "Data mismatch in all-to-all verification at rank " << globalRank;

  XLOGF(
      INFO,
      "Rank {}: All-to-all RDMA write test passed - verified data from all {} peers",
      globalRank,
      numRanks - 1);

  // Cleanup - RAII handles ibverbx objects
  CUDA_CHECK(cudaFree(sendBuf));
  CUDA_CHECK(cudaFree(recvBuf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
