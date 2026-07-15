// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdint>
#include <vector>

#include "comms/prims/tests/MultipeerIbgdaDeviceTransportTest.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaDeviceTransport.cuh"

namespace comms::prims::tests {

// =============================================================================
// Device-side test kernel for rank mapping logic
// =============================================================================

__global__ void testRankMappingKernel(
    int myRank,
    int nRanks,
    int* results,
    int* expectedResults,
    int numTestCases,
    bool* success) {
  *success = true;

  // Create transport with empty peer transports (we only test rank mapping)
  MultipeerIbgdaDeviceTransport transport(
      myRank, nRanks, DeviceSpan<P2pIbgdaTransportDevice>());

  // Verify basic properties
  if (transport.myRank != myRank) {
    *success = false;
    return;
  }
  if (transport.nRanks != nRanks) {
    *success = false;
    return;
  }
  if (transport.numPeers() != nRanks - 1) {
    *success = false;
    return;
  }

  // Test indexToRank mapping for all peer indices
  for (int i = 0; i < numTestCases; ++i) {
    results[i] = transport.indexToRank(i);
    if (results[i] != expectedResults[i]) {
      *success = false;
    }
  }
}

// =============================================================================
// Wrapper function to launch the kernel (called from .cc test file)
// =============================================================================

void runTestRankMappingKernel(
    int myRank,
    int nRanks,
    int* d_results,
    int* d_expected,
    int numTestCases,
    bool* d_success) {
  testRankMappingKernel<<<1, 1>>>(
      myRank, nRanks, d_results, d_expected, numTestCases, d_success);
}

// =============================================================================
// V4 QP-ownership lane-mapping test
//
// Verifies that select_put_lane() maps a ThreadGroup to a lane per the
// block-owned-QP + per-group-staggered-NIC model (no IB hardware needed —
// select_put_lane only reads QP-pointer values and the per-channel cursor):
//   - physical QP slot is owned by group.block_id (groups in a block share it)
//   - the NIC lane is staggered by group.group_id (co-resident groups spread)
// =============================================================================

// Grants the test kernel access to the private select_put_lane(). Returns the
// chosen lane's (nic, qp-slot) as a public V4LaneResult so callers never name
// the private IbgdaLane type.
struct IbgdaLaneTestAccess {
  __device__ static V4LaneResult
  select(P2pIbgdaTransportDevice& t, ThreadGroup& g, IbDirection dir) {
    const auto lane = t.select_put_lane(g, dir);
    return V4LaneResult{
        static_cast<int>(lane.nic_id), static_cast<int>(lane.qp_slot_per_nic)};
  }
};

__global__ void testV4LaneMappingKernel(
    NicDeviceIbgdaResources* nicResources,
    int numNics,
    IbLocalChannel* localChannels,
    int maxChannels,
    int qpsPerConn,
    int qpDirCount,
    int numBlocks,
    int groupsPerBlock,
    int* outNicId, // indexed by group_id, size maxChannels
    int* outQpSlot) {
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>(nicResources, numNics),
      /*ownedRemoteSignalBuf=*/{},
      /*ownedLocalSignalBuf=*/{},
      /*ownedCounterBuf=*/{},
      /*numSignalSlots=*/0,
      /*numCounterSlots=*/0,
      maxChannels,
      qpsPerConn,
      qpDirCount,
      DeviceSpan<IbLocalChannel>(localChannels, maxChannels),
      /*channelLayout=*/{});

  for (int b = 0; b < numBlocks; ++b) {
    for (int j = 0; j < groupsPerBlock; ++j) {
      ThreadGroup g{};
      g.thread_id_in_group = 0;
      g.group_size = 1;
      g.group_id = static_cast<uint32_t>(b * groupsPerBlock + j);
      g.block_id = static_cast<uint32_t>(b);
      g.total_groups = static_cast<uint32_t>(maxChannels);
      g.scope = SyncScope::MULTIWARP;
      const V4LaneResult r =
          IbgdaLaneTestAccess::select(transport, g, IbDirection::Send);
      outNicId[g.group_id] = r.nic_id;
      outQpSlot[g.group_id] = r.qp_slot_per_nic;
    }
  }
}

void runV4LaneMappingTest(
    int numNics,
    int numBlocks,
    int groupsPerBlock,
    int qpsPerConn,
    std::vector<V4LaneResult>& outResults) {
  const int maxChannels = numBlocks * groupsPerBlock;
  const int qpDirCount = kIbDirections;
  const int perNicQps = maxChannels * qpDirCount * qpsPerConn;

  // Fake, non-null QP pointers. select_put_lane only reads the pointer value
  // (stored into IbgdaLane.qp) and never dereferences it.
  std::vector<doca_gpu_dev_verbs_qp*> h_qps(
      static_cast<std::size_t>(numNics) * perNicQps);
  for (std::size_t i = 0; i < h_qps.size(); ++i) {
    h_qps[i] = reinterpret_cast<doca_gpu_dev_verbs_qp*>(
        static_cast<std::uintptr_t>(i + 1));
  }
  const std::size_t qpBytes = h_qps.size() * sizeof(doca_gpu_dev_verbs_qp*);
  doca_gpu_dev_verbs_qp** d_qps = nullptr;
  doca_gpu_dev_verbs_qp** d_comp = nullptr;
  cudaMalloc(&d_qps, qpBytes);
  cudaMalloc(&d_comp, qpBytes);
  cudaMemcpy(d_qps, h_qps.data(), qpBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_comp, h_qps.data(), qpBytes, cudaMemcpyHostToDevice);

  std::vector<NicDeviceIbgdaResources> h_nic;
  h_nic.reserve(numNics);
  for (int n = 0; n < numNics; ++n) {
    h_nic.push_back(
        NicDeviceIbgdaResources{
            DeviceSpan<doca_gpu_dev_verbs_qp*>(
                d_qps + n * perNicQps, perNicQps),
            DeviceSpan<doca_gpu_dev_verbs_qp*>(
                d_comp + n * perNicQps, perNicQps),
            NetworkLKey{},
            n});
  }
  NicDeviceIbgdaResources* d_nic = nullptr;
  cudaMalloc(&d_nic, numNics * sizeof(NicDeviceIbgdaResources));
  cudaMemcpy(
      d_nic,
      h_nic.data(),
      numNics * sizeof(NicDeviceIbgdaResources),
      cudaMemcpyHostToDevice);

  IbLocalChannel* d_channels = nullptr;
  cudaMalloc(&d_channels, maxChannels * sizeof(IbLocalChannel));
  cudaMemset(d_channels, 0, maxChannels * sizeof(IbLocalChannel));

  int* d_nicId = nullptr;
  int* d_qpSlot = nullptr;
  cudaMalloc(&d_nicId, maxChannels * sizeof(int));
  cudaMalloc(&d_qpSlot, maxChannels * sizeof(int));

  testV4LaneMappingKernel<<<1, 1>>>(
      d_nic,
      numNics,
      d_channels,
      maxChannels,
      qpsPerConn,
      qpDirCount,
      numBlocks,
      groupsPerBlock,
      d_nicId,
      d_qpSlot);
  cudaDeviceSynchronize();

  std::vector<int> nicId(maxChannels);
  std::vector<int> qpSlot(maxChannels);
  cudaMemcpy(
      nicId.data(), d_nicId, maxChannels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      qpSlot.data(),
      d_qpSlot,
      maxChannels * sizeof(int),
      cudaMemcpyDeviceToHost);
  outResults.resize(maxChannels);
  for (int i = 0; i < maxChannels; ++i) {
    outResults[i] = V4LaneResult{nicId[i], qpSlot[i]};
  }

  cudaFree(d_qps);
  cudaFree(d_comp);
  cudaFree(d_nic);
  cudaFree(d_channels);
  cudaFree(d_nicId);
  cudaFree(d_qpSlot);
}

} // namespace comms::prims::tests
