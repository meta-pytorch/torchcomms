// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstring>

#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

/**
 * Test fixture for MultiPeerTransport.
 *
 * For multi-node tests that exercise mixed NVL + IBGDA topology
 * (cross-node peers), see MultiPeerTransportMultiNodeTest.cc.
 */
class MultiPeerTransportTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    detectLocalSize();
  }

  std::unique_ptr<MultiPeerTransport> createTransport() {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .dataBufferSize = 256 * 1024,
                .chunkSize = 512,
                .pipelineDepth = 4,
                .p2pSignalCount = 4,
            },
        .ibgdaConfig =
            {
                .cudaDevice = localRank,
            },
    };
    return std::make_unique<MultiPeerTransport>(
        globalRank,
        numRanks,
        localRank,
        std::make_shared<MpiBootstrap>(),
        config);
  }

  // Returns true when NVL peers span multiple hosts (MNNVL).
  // cudaIpc is host-local and cannot cross host boundaries.
  bool nvlSpansMultipleHosts(const MultiPeerTransport& t) const {
    return t.nvl_n_ranks() > localSize_;
  }

  // Verify exchangeNvlBuffer mappedPtrs: self entry == localBuf,
  // peer entries non-null and readable.
  void verifyMappedPtrs(
      const MultiPeerTransport& t,
      const std::vector<void*>& mappedPtrs,
      void* localBuf) {
    ASSERT_EQ(static_cast<int>(mappedPtrs.size()), t.nvl_n_ranks());
    EXPECT_EQ(mappedPtrs[t.nvl_local_rank()], localBuf);

    for (int rank = 0; rank < t.nvl_n_ranks(); ++rank) {
      if (rank == t.nvl_local_rank()) {
        continue;
      }
      ASSERT_NE(mappedPtrs[rank], nullptr)
          << "mapped ptr for NVL rank " << rank << " is null";

      char peerByte = 0;
      CUDACHECK_TEST(
          cudaMemcpy(&peerByte, mappedPtrs[rank], 1, cudaMemcpyDeviceToHost));
      EXPECT_NE(peerByte, 0)
          << "peer data at NVL rank " << rank << " should be non-zero";
    }
  }

 private:
  void detectLocalSize() {
    char myHostname[64]{};
    gethostname(myHostname, sizeof(myHostname));

    std::vector<char> allHostnames(numRanks * 64);
    MPI_Allgather(
        myHostname,
        64,
        MPI_BYTE,
        allHostnames.data(),
        64,
        MPI_BYTE,
        MPI_COMM_WORLD);

    localSize_ = 0;
    for (int r = 0; r < numRanks; ++r) {
      if (std::strcmp(myHostname, &allHostnames[r * 64]) == 0) {
        ++localSize_;
      }
    }
  }

  int localSize_{0};
};

TEST_F(MultiPeerTransportTestFixture, TopologyDiscovery) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  int peer = (globalRank == 0) ? 1 : 0;

  EXPECT_EQ(transport->get_transport_type(globalRank), TransportType::SELF);
  EXPECT_EQ(transport->get_transport_type(peer), TransportType::P2P_NVL);
  EXPECT_FALSE(transport->nvl_peer_ranks().empty());
  EXPECT_EQ(
      static_cast<int>(transport->ibgda_peer_ranks().size()), numRanks - 1);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, SelfTransportType) {
  auto transport = createTransport();
  EXPECT_EQ(transport->get_transport_type(globalRank), TransportType::SELF);
  EXPECT_EQ(transport->my_rank(), globalRank);
  EXPECT_EQ(transport->n_ranks(), numRanks);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, ExchangeSucceeds) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  EXPECT_NO_THROW(transport->exchange());

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, HostNvlAccessor) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  transport->exchange();

  int peer = (globalRank == 0) ? 1 : 0;
  auto p2p = transport->get_p2p_nvl_transport_device(peer);
  EXPECT_NE(p2p.getLocalState().dataBuffer, nullptr);
  EXPECT_NE(p2p.getRemoteState().dataBuffer, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, SelfAccessor) {
  auto transport = createTransport();
  (void)transport->get_p2p_self_transport_device();

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, DeviceHandleMetadata) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  transport->exchange();

  auto handle = transport->get_device_handle();
  EXPECT_EQ(handle.myRank, globalRank);
  EXPECT_EQ(handle.nRanks, numRanks);
  EXPECT_EQ(handle.transports.size(), static_cast<uint32_t>(numRanks));
  EXPECT_GT(handle.numNvlPeers, 0);
  EXPECT_EQ(handle.numIbPeers, numRanks - 1);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, DeviceHandleBeforeExchange) {
  auto transport = createTransport();
  EXPECT_THROW(transport->get_device_handle(), std::runtime_error);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportTestFixture, HostIbgdaAccessorForNvlPeer) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  transport->exchange();

  int peer = (globalRank == 0) ? 1 : 0;
  ASSERT_TRUE(transport->is_nvl_peer(peer));
  EXPECT_TRUE(transport->has_ibgda(peer));
  EXPECT_NE(transport->get_p2p_ibgda_transport_device(peer), nullptr);

  MPI_Barrier(MPI_COMM_WORLD);
}

// cudaIpc path — skips on MNNVL where NVL peers span hosts.
TEST_F(MultiPeerTransportTestFixture, ExchangeNvlBufferCudaMalloc) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  transport->exchange();

  if (transport->nvl_peer_ranks().empty()) {
    GTEST_SKIP() << "No NVL peers available";
  }
  if (nvlSpansMultipleHosts(*transport)) {
    GTEST_SKIP() << "cudaIpc does not work across hosts; "
                 << "NVL domain spans multiple hosts (MNNVL). "
                 << "Use ExchangeNvlBufferFabric for cross-host NVL exchange.";
  }

  const size_t nbytes = 4096;
  void* localBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&localBuf, nbytes));
  CUDACHECK_TEST(cudaMemset(localBuf, globalRank + 1, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  auto mappedPtrs = transport->exchangeNvlBuffer(localBuf, nbytes);
  verifyMappedPtrs(*transport, mappedPtrs, localBuf);

  transport->unmapNvlBuffers(mappedPtrs);
  CUDACHECK_TEST(cudaFree(localBuf));

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verifies exchange + unmap round-trip works twice (no state leaks).
TEST_F(MultiPeerTransportTestFixture, ExchangeNvlBufferMultipleRoundTrips) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto transport = createTransport();
  transport->exchange();

  if (transport->nvl_peer_ranks().empty()) {
    GTEST_SKIP() << "No NVL peers available";
  }
  if (nvlSpansMultipleHosts(*transport)) {
    GTEST_SKIP() << "cudaIpc does not work across hosts; "
                 << "NVL domain spans multiple hosts (MNNVL).";
  }

  const size_t nbytes = 1024;
  for (int iter = 0; iter < 2; ++iter) {
    void* localBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&localBuf, nbytes));
    CUDACHECK_TEST(cudaMemset(localBuf, iter + 1, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    auto mappedPtrs = transport->exchangeNvlBuffer(localBuf, nbytes);
    EXPECT_EQ(static_cast<int>(mappedPtrs.size()), transport->nvl_n_ranks());
    for (int rank = 0; rank < transport->nvl_n_ranks(); ++rank) {
      EXPECT_NE(mappedPtrs[rank], nullptr);
    }

    transport->unmapNvlBuffers(mappedPtrs);
    CUDACHECK_TEST(cudaFree(localBuf));

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// Fabric handle path — mimics ncclMemAlloc on GB200/GB300.
TEST_F(MultiPeerTransportTestFixture, ExchangeNvlBufferFabric) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }
  if (!GpuMemHandler::isFabricHandleSupported()) {
    GTEST_SKIP() << "Fabric handles not supported on this GPU/CUDA version";
  }

  auto transport = createTransport();
  transport->exchange();

  if (transport->nvl_peer_ranks().empty()) {
    GTEST_SKIP() << "No NVL peers available";
  }

#if CUDART_VERSION >= 12030
  const size_t requestedSize = 4096;

  int cudaDev = 0;
  CUdevice cuDev;
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));
  ASSERT_EQ(cuDeviceGet(&cuDev, cudaDev), CUDA_SUCCESS);

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, cuDev};
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  int rdmaFlag = 0;
  cuDeviceGetAttribute(
      &rdmaFlag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      cuDev);
  if (rdmaFlag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  size_t granularity = 0;
  ASSERT_EQ(
      cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      CUDA_SUCCESS);
  size_t allocSize =
      ((requestedSize + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle allocHandle;
  ASSERT_EQ(cuMemCreate(&allocHandle, allocSize, &prop, 0), CUDA_SUCCESS);

  CUdeviceptr devPtr = 0;
  ASSERT_EQ(
      cuMemAddressReserve(&devPtr, allocSize, granularity, 0, 0), CUDA_SUCCESS);
  ASSERT_EQ(cuMemMap(devPtr, allocSize, 0, allocHandle, 0), CUDA_SUCCESS);

  CUmemAccessDesc accessDesc = {};
  accessDesc.location = {CU_MEM_LOCATION_TYPE_DEVICE, cuDev};
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ASSERT_EQ(cuMemSetAccess(devPtr, allocSize, &accessDesc, 1), CUDA_SUCCESS);

  void* localBuf = reinterpret_cast<void*>(devPtr);
  CUDACHECK_TEST(cudaMemset(localBuf, globalRank + 1, requestedSize));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  auto mappedPtrs = transport->exchangeNvlBuffer(localBuf, requestedSize);
  verifyMappedPtrs(*transport, mappedPtrs, localBuf);

  transport->unmapNvlBuffers(mappedPtrs);
  cuMemUnmap(devPtr, allocSize);
  cuMemAddressFree(devPtr, allocSize);
  cuMemRelease(allocHandle);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
