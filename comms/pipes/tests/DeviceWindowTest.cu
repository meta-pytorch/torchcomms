// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/DeviceWindowTest.cuh"

#include <algorithm>
#include <memory>
#include <vector>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes::test {

// Helper: build a minimal NVL-only DeviceWindow for unit tests.
// All IBGDA params are zeroed/null since these tests run NVL-only.
struct NvlOnlyDeviceWindowBuffers {
  std::unique_ptr<meta::comms::DeviceBuffer> nvlPeerSignalInboxBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> nvlPeerSignalSpansBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> peerIndexMapsBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> transportsBuf;

  DeviceWindow create(int myRank, int nRanks, int peerSignalCount) {
    int nPeers = nRanks - 1;

    // Transports array (needed for handle_.get_type dispatching)
    transportsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nRanks) * sizeof(Transport));
    CUDACHECK_TEST(cudaMemset(
        transportsBuf->get(), 0, std::max(1, nRanks) * sizeof(Transport)));
    auto* transportsPtr = static_cast<Transport*>(transportsBuf->get());
    for (int i = 0; i < nRanks; ++i) {
      TransportType type =
          (i == myRank) ? TransportType::SELF : TransportType::P2P_NVL;
      CUDACHECK_TEST(cudaMemcpy(
          &transportsPtr[i].type,
          &type,
          sizeof(TransportType),
          cudaMemcpyHostToDevice));
    }

    // NVL per-peer signal inbox: nPeers * peerSignalCount entries
    std::size_t peerInboxSlots =
        static_cast<std::size_t>(std::max(1, nPeers)) * peerSignalCount;
    nvlPeerSignalInboxBuf = std::make_unique<meta::comms::DeviceBuffer>(
        peerInboxSlots * sizeof(SignalState));
    CUDACHECK_TEST(cudaMemset(
        nvlPeerSignalInboxBuf->get(), 0, peerInboxSlots * sizeof(SignalState)));

    // NVL per-peer signal spans: build on host, copy to device.
    // Each span[i] points to inbox + i * peerSignalCount with size
    // peerSignalCount.
    nvlPeerSignalSpansBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nPeers) * sizeof(DeviceSpan<SignalState>));
    {
      auto* inboxBase = static_cast<SignalState*>(nvlPeerSignalInboxBuf->get());
      std::vector<DeviceSpan<SignalState>> hostSpans(std::max(1, nPeers));
      for (int i = 0; i < nPeers; ++i) {
        new (&hostSpans[i]) DeviceSpan<SignalState>(
            inboxBase + i * peerSignalCount, peerSignalCount);
      }
      CUDACHECK_TEST(cudaMemcpy(
          nvlPeerSignalSpansBuf->get(),
          hostSpans.data(),
          std::max(1, nPeers) * sizeof(DeviceSpan<SignalState>),
          cudaMemcpyHostToDevice));
    }

    // Pre-computed peer index maps: rankToNvlPeerIndex only
    // (IBGDA uses rank_to_peer_index() arithmetic, no table needed)
    peerIndexMapsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nRanks) * sizeof(int));
    {
      std::vector<int> nvlMap(nRanks, -1);
      int nvlIdx = 0;
      for (int r = 0; r < nRanks; ++r) {
        if (r != myRank) {
          nvlMap[r] = nvlIdx++;
        }
      }
      auto* base = static_cast<int*>(peerIndexMapsBuf->get());
      CUDACHECK_TEST(cudaMemcpy(
          base, nvlMap.data(), nRanks * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Build DeviceWindow field-by-field.
    // DeviceSpan has deleted copy-assignment, so we use placement new.
    DeviceWindow dw{};
    new (&dw.handle_) MultiPeerDeviceHandle{
        myRank,
        nRanks,
        DeviceSpan<Transport>(transportsPtr, nRanks),
        nPeers,
        0};
    dw.nNvlPeers_ = nPeers;
    dw.nIbgdaPeers_ = 0;
    dw.peerSignalCount_ = peerSignalCount;

    auto* indexBase = static_cast<int*>(peerIndexMapsBuf->get());
    new (&dw.rankToNvlPeerIndex_) DeviceSpan<int>(indexBase, nRanks);

    new (&dw.nvlPeerSignalInbox_) DeviceSpan<SignalState>(
        static_cast<SignalState*>(nvlPeerSignalInboxBuf->get()),
        static_cast<std::size_t>(nPeers) * peerSignalCount);
    new (&dw.nvlPeerSignalSpans_) DeviceSpan<DeviceSpan<SignalState>>(
        static_cast<DeviceSpan<SignalState>*>(nvlPeerSignalSpansBuf->get()),
        nPeers);

    return dw;
  }
};

// =============================================================================
// DeviceWindow Construction Test
// =============================================================================

__global__ void deviceWindowConstructionKernel(DeviceWindow dw, int* results) {
  results[0] = dw.rank();
  results[1] = dw.n_ranks();
  results[2] = dw.num_nvl_peers();
}

void testDeviceWindowConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  deviceWindowConstructionKernel<<<1, 1>>>(dw, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Basic Accessors Test
// =============================================================================

__global__ void deviceWindowBasicAccessorsKernel(
    DeviceWindow dw,
    int* results) {
  results[0] = dw.rank();
  results[1] = dw.n_ranks();
}

void testDeviceWindowBasicAccessors(int myRank, int nRanks, int* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, 1);

  deviceWindowBasicAccessorsKernel<<<1, 1>>>(dw, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Self-Transport Put Test
// =============================================================================

__global__ void selfTransportPutKernel(
    Transport* transport,
    char* dst_d,
    const char* src_d,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport->self.put_group(group, dst_d, src_d, nbytes);
}

void testSelfTransportPut(
    void* transport_d,
    char* dst_d,
    const char* src_d,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  selfTransportPutKernel<<<numBlocks, blockSize>>>(
      static_cast<Transport*>(transport_d), dst_d, src_d, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Get Transport Type Test
// =============================================================================

__global__ void getTransportTypeKernel(Transport* transport, int* results) {
  results[0] = (transport->type == TransportType::SELF) ? 1 : 0;
}

void testGetTransportType(void* transport_d, int* results) {
  getTransportTypeKernel<<<1, 1>>>(
      static_cast<Transport*>(transport_d), results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Peer Iteration Helpers Test
// =============================================================================

__global__ void peerIterationHelpersKernel(DeviceWindow dw, int* results) {
  results[0] = dw.num_peers();

  int numPeers = dw.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    results[1 + i] = dw.peer_index_to_rank(i);
  }
}

void testPeerIterationHelpers(int myRank, int nRanks, int* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, 1);

  peerIterationHelpersKernel<<<1, 1>>>(dw, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Peer Index Conversion Roundtrip Test
// =============================================================================

__global__ void peerIndexConversionRoundtripKernel(
    DeviceWindow dw,
    int nRanks,
    int myRank,
    int* results) {
  int numPeers = dw.num_peers();
  int idx = 0;

  results[idx++] = numPeers;

  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    results[idx++] = dw.rank_to_peer_index(rank);
  }

  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    int peerIdx = dw.rank_to_peer_index(rank);
    results[idx++] = dw.peer_index_to_rank(peerIdx);
  }

  for (int i = 0; i < numPeers; ++i) {
    int rank = dw.peer_index_to_rank(i);
    results[idx++] = dw.rank_to_peer_index(rank);
  }

  results[idx++] = static_cast<int>(dw.get_handle().transports[myRank].type);

  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank)
      continue;
    results[idx++] = static_cast<int>(dw.get_handle().transports[rank].type);
  }
}

void testPeerIndexConversionRoundtrip(int myRank, int nRanks, int* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, 1);

  peerIndexConversionRoundtripKernel<<<1, 1>>>(dw, nRanks, myRank, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow NVL Signal Write+Read Test
// =============================================================================

__global__ void deviceSignalWriteReadKernel(
    DeviceWindow dw,
    int targetPeerRank,
    int signalId,
    uint64_t* results) {
  auto group = make_block_group();

  dw.signal_peer(group, targetPeerRank, signalId);

  if (group.is_global_leader()) {
    results[0] = dw.read_signal_from(targetPeerRank, signalId);
    results[1] = dw.read_signal(signalId);
  }
}

void testDeviceWindowSignalWriteRead(
    int myRank,
    int nRanks,
    int signalCount,
    int targetPeerRank,
    int signalId,
    uint64_t* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  deviceSignalWriteReadKernel<<<1, 32>>>(dw, targetPeerRank, signalId, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow read_signal Test
// =============================================================================

__global__ void readSignalKernel(
    DeviceWindow dw,
    int targetPeerRank,
    int signalId,
    uint64_t* results) {
  auto group = make_block_group();

  // Signal a peer (thread-level, from leader)
  if (group.is_global_leader()) {
    dw.signal_peer(targetPeerRank, signalId, SignalOp::SIGNAL_ADD, 3);
  }
  group.sync();

  // Read aggregate (thread-level API)
  if (group.is_leader()) {
    results[0] = dw.read_signal(signalId);
  }
}

void testDeviceWindowReadSignalGroup(
    int myRank,
    int nRanks,
    int signalCount,
    uint64_t* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  // Signal peer rank 1 on signal slot 0
  int targetPeerRank = (myRank == 0) ? 1 : 0;
  int signalId = 0;
  readSignalKernel<<<1, 32>>>(dw, targetPeerRank, signalId, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Offset-Based NVL Put Test
// =============================================================================

__global__ void nvlOffsetPutKernel(
    DeviceWindow dw,
    int targetPeerRank,
    std::size_t dst_offset,
    LocalBufferRegistration src_buf,
    std::size_t src_offset,
    std::size_t nbytes) {
  auto group = make_block_group();
  dw.put(group, targetPeerRank, dst_offset, src_buf, src_offset, nbytes);
}

// Extended NVL helper that also sets up window buffer pointers
// for offset-based put/put_signal tests.
struct NvlOffsetPutDeviceWindowBuffers : NvlOnlyDeviceWindowBuffers {
  std::unique_ptr<meta::comms::DeviceBuffer> windowPtrsBuf;

  DeviceWindow
  create_with_offset_put(int myRank, int nRanks, void* windowBuf_d) {
    auto dw = create(myRank, nRanks, 1);
    int nPeers = nRanks - 1;

    // windowNvlPeerPtrs_: each NVL peer's "window buffer" pointer
    windowPtrsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nPeers) * sizeof(void*));
    {
      std::vector<void*> hostPtrs(nPeers, windowBuf_d);
      CUDACHECK_TEST(cudaMemcpy(
          windowPtrsBuf->get(),
          hostPtrs.data(),
          nPeers * sizeof(void*),
          cudaMemcpyHostToDevice));
    }
    new (&dw.windowNvlPeerPtrs_)
        DeviceSpan<void*>(static_cast<void**>(windowPtrsBuf->get()), nPeers);

    return dw;
  }
};

void testDeviceWindowNvlOffsetPut(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  int targetPeerRank = (myRank == 0) ? 1 : 0;
  LocalBufferRegistration src_buf{srcBuf_d, srcBufSize, NetworkLKey{}};
  nvlOffsetPutKernel<<<4, 256>>>(
      dw, targetPeerRank, dst_offset, src_buf, src_offset, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Offset-Based NVL Per-Group Put Test
// =============================================================================

__global__ void nvlOffsetPutPerGroupKernel(
    DeviceWindow dw,
    int targetPeerRank,
    LocalBufferRegistration src_buf,
    std::size_t tileSize) {
  auto group = make_block_group();
  std::size_t offset = group.group_id * tileSize;
  dw.put(group, targetPeerRank, offset, src_buf, offset, tileSize);
}

void testDeviceWindowNvlOffsetPutPerGroup(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t tileSize,
    int numTiles) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  int targetPeerRank = (myRank == 0) ? 1 : 0;
  LocalBufferRegistration src_buf{srcBuf_d, srcBufSize, NetworkLKey{}};
  nvlOffsetPutPerGroupKernel<<<numTiles, 256>>>(
      dw, targetPeerRank, src_buf, tileSize);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Offset-Based NVL Put + Signal Test
// =============================================================================

__global__ void nvlOffsetPutSignalKernel(
    DeviceWindow dw,
    int targetPeerRank,
    std::size_t dst_offset,
    LocalBufferRegistration src_buf,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId) {
  auto group = make_block_group();
  dw.put_signal(
      group, targetPeerRank, dst_offset, src_buf, src_offset, nbytes, signalId);
}

void testDeviceWindowNvlOffsetPutSignal(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  int targetPeerRank = (myRank == 0) ? 1 : 0;
  LocalBufferRegistration src_buf{srcBuf_d, srcBufSize, NetworkLKey{}};
  nvlOffsetPutSignalKernel<<<4, 256>>>(
      dw, targetPeerRank, dst_offset, src_buf, src_offset, nbytes, signalId);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Bidirectional Offset-Based NVL Put + Signal Test
// =============================================================================

void testDeviceWindowNvlBidirectionalOffsetPutSignal(
    char* windowBuf0_d,
    char* windowBuf1_d,
    const char* srcBuf0_d,
    const char* srcBuf1_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId) {
  const int nRanks = 2;

  // Rank 0's DeviceWindow: peer is rank 1, peer's window = windowBuf1_d
  NvlOffsetPutDeviceWindowBuffers bufs0;
  auto dw0 = bufs0.create_with_offset_put(0, nRanks, windowBuf1_d);

  // Rank 1's DeviceWindow: peer is rank 0, peer's window = windowBuf0_d
  NvlOffsetPutDeviceWindowBuffers bufs1;
  auto dw1 = bufs1.create_with_offset_put(1, nRanks, windowBuf0_d);

  LocalBufferRegistration srcReg0{srcBuf0_d, srcBufSize, NetworkLKey{}};
  LocalBufferRegistration srcReg1{srcBuf1_d, srcBufSize, NetworkLKey{}};

  // Rank 0 puts to rank 1's window buffer
  nvlOffsetPutSignalKernel<<<4, 256>>>(
      dw0, 1, dst_offset, srcReg0, src_offset, nbytes, signalId);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Rank 1 puts to rank 0's window buffer
  nvlOffsetPutSignalKernel<<<4, 256>>>(
      dw1, 0, dst_offset, srcReg1, src_offset, nbytes, signalId);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow signal_all + read_signal Aggregate Test
// =============================================================================

__global__ void
signalAllAggregateKernel(DeviceWindow dw, int signalId, uint64_t* results) {
  auto group = make_block_group();

  // signal_all signals every peer with value 1
  dw.signal_all(group, signalId, SignalOp::SIGNAL_ADD, 1);

  // Read aggregate via thread-level API from leader
  if (group.is_global_leader()) {
    results[0] = dw.read_signal(signalId);
  }
}

void testDeviceWindowSignalAllAggregate(
    int myRank,
    int nRanks,
    int signalCount,
    int signalId,
    uint64_t* results) {
  NvlOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  signalAllAggregateKernel<<<1, 32>>>(dw, signalId, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// IBGDA-Only DeviceWindow Helper
// =============================================================================

// Helper: build a DeviceWindow with IBGDA-only peers for unit tests.
// The IBGDA inbox is a flat uint64_t array (no real NIC/QP needed).
// This lets us test read_signal_from / read_signal on the IBGDA path
// by writing known values directly into the local inbox.
struct IbgdaOnlyDeviceWindowBuffers {
  std::unique_ptr<meta::comms::DeviceBuffer> ibgdaPeerSignalInboxBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> ibgdaPeerSignalRemoteBufsBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> peerIndexMapsBuf;
  std::unique_ptr<meta::comms::DeviceBuffer> transportsBuf;

  DeviceWindow create(int myRank, int nRanks, int peerSignalCount) {
    int nPeers = nRanks - 1;

    // Transports array: all peers are IBGDA
    transportsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nRanks) * sizeof(Transport));
    CUDACHECK_TEST(cudaMemset(
        transportsBuf->get(), 0, std::max(1, nRanks) * sizeof(Transport)));
    auto* transportsPtr = static_cast<Transport*>(transportsBuf->get());
    for (int i = 0; i < nRanks; ++i) {
      TransportType type =
          (i == myRank) ? TransportType::SELF : TransportType::P2P_IBGDA;
      CUDACHECK_TEST(cudaMemcpy(
          &transportsPtr[i].type,
          &type,
          sizeof(TransportType),
          cudaMemcpyHostToDevice));
    }

    // IBGDA per-peer signal inbox: nPeers * peerSignalCount uint64_t slots
    std::size_t inboxSlots =
        static_cast<std::size_t>(std::max(1, nPeers)) * peerSignalCount;
    ibgdaPeerSignalInboxBuf = std::make_unique<meta::comms::DeviceBuffer>(
        inboxSlots * sizeof(uint64_t));
    CUDACHECK_TEST(cudaMemset(
        ibgdaPeerSignalInboxBuf->get(), 0, inboxSlots * sizeof(uint64_t)));

    // IBGDA remote bufs (dummy — no real QP, but needed for DeviceSpan)
    ibgdaPeerSignalRemoteBufsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nPeers) * sizeof(IbgdaRemoteBuffer));
    CUDACHECK_TEST(cudaMemset(
        ibgdaPeerSignalRemoteBufsBuf->get(),
        0,
        std::max(1, nPeers) * sizeof(IbgdaRemoteBuffer)));

    // Pre-computed peer index maps (NVL only; IBGDA uses rank_to_peer_index())
    peerIndexMapsBuf = std::make_unique<meta::comms::DeviceBuffer>(
        std::max(1, nRanks) * sizeof(int));
    {
      std::vector<int> nvlMap(nRanks, -1);
      auto* base = static_cast<int*>(peerIndexMapsBuf->get());
      CUDACHECK_TEST(cudaMemcpy(
          base, nvlMap.data(), nRanks * sizeof(int), cudaMemcpyHostToDevice));
    }

    DeviceWindow dw{};
    new (&dw.handle_) MultiPeerDeviceHandle{
        myRank,
        nRanks,
        DeviceSpan<Transport>(transportsPtr, nRanks),
        0,
        nPeers};
    dw.nNvlPeers_ = 0;
    dw.nIbgdaPeers_ = nPeers;
    dw.peerSignalCount_ = peerSignalCount;

    auto* indexBase = static_cast<int*>(peerIndexMapsBuf->get());
    new (&dw.rankToNvlPeerIndex_) DeviceSpan<int>(indexBase, nRanks);

    dw.ibgdaPeerSignalInbox_ =
        static_cast<uint64_t*>(ibgdaPeerSignalInboxBuf->get());
    new (&dw.ibgdaPeerSignalRemoteBufs_) DeviceSpan<IbgdaRemoteBuffer>(
        static_cast<IbgdaRemoteBuffer*>(ibgdaPeerSignalRemoteBufsBuf->get()),
        nPeers);

    return dw;
  }

  // Get host-accessible pointer to the raw inbox for seeding test values
  uint64_t* getInboxPtr() {
    return static_cast<uint64_t*>(ibgdaPeerSignalInboxBuf->get());
  }
};

// =============================================================================
// IBGDA Signal: read_signal_from + read_signal Test
// =============================================================================

__global__ void ibgdaSignalReadKernel(
    DeviceWindow dw,
    int sourceRank,
    int signalId,
    uint64_t* results) {
  results[0] = dw.read_signal_from(sourceRank, signalId);
  results[1] = dw.read_signal(signalId);
}

void testIbgdaSignalRead(
    int myRank,
    int nRanks,
    int signalCount,
    int sourceRank,
    int signalId,
    uint64_t seedValue,
    uint64_t* results) {
  IbgdaOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  // Seed the inbox: write seedValue at the slot for (sourceRank, signalId).
  // The skip-self peer index for sourceRank is:
  int peerIdx = (sourceRank < myRank) ? sourceRank : (sourceRank - 1);
  int slot = peerIdx * signalCount + signalId;
  CUDACHECK_TEST(cudaMemcpy(
      bufs.getInboxPtr() + slot,
      &seedValue,
      sizeof(uint64_t),
      cudaMemcpyHostToDevice));

  ibgdaSignalReadKernel<<<1, 1>>>(dw, sourceRank, signalId, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// IBGDA Signal: Multi-peer aggregate read_signal Test
// =============================================================================

__global__ void ibgdaSignalAggregateReadKernel(
    DeviceWindow dw,
    int signalId,
    uint64_t* result) {
  *result = dw.read_signal(signalId);
}

void testIbgdaSignalAggregateRead(
    int myRank,
    int nRanks,
    int signalCount,
    int signalId,
    const uint64_t* peerValues,
    int nPeers,
    uint64_t* result) {
  IbgdaOnlyDeviceWindowBuffers bufs;
  auto dw = bufs.create(myRank, nRanks, signalCount);

  // Seed the inbox for each peer at the given signalId
  for (int i = 0; i < nPeers; ++i) {
    int slot = i * signalCount + signalId;
    CUDACHECK_TEST(cudaMemcpy(
        bufs.getInboxPtr() + slot,
        &peerValues[i],
        sizeof(uint64_t),
        cudaMemcpyHostToDevice));
  }

  ibgdaSignalAggregateReadKernel<<<1, 1>>>(dw, signalId, result);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow get_nvlink_address() Test
// =============================================================================

__global__ void
getNvlinkAddressKernel(DeviceWindow dw, int nRanks, int64_t* results) {
  for (int r = 0; r < nRanks; ++r) {
    void* addr = dw.get_nvlink_address(r);
    results[r] = reinterpret_cast<int64_t>(addr);
  }
}

void testDeviceWindowGetNvlinkAddress(
    int myRank,
    int nRanks,
    void* windowBuf_d,
    int64_t* results) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  getNvlinkAddressKernel<<<1, 1>>>(dw, nRanks, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Offset-Based NVL Put + Signal + Counter Test
// =============================================================================

__global__ void nvlOffsetPutSignalCounterKernel(
    DeviceWindow dw,
    int targetPeerRank,
    std::size_t dst_offset,
    LocalBufferRegistration src_buf,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  auto group = make_block_group();
  dw.put_signal_counter(
      group,
      targetPeerRank,
      dst_offset,
      src_buf,
      src_offset,
      nbytes,
      signalId,
      signalVal,
      counterId,
      counterVal);
}

void testDeviceWindowNvlOffsetPutSignalCounter(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  int targetPeerRank = (myRank == 0) ? 1 : 0;
  LocalBufferRegistration src_buf{srcBuf_d, srcBufSize, NetworkLKey{}};
  nvlOffsetPutSignalCounterKernel<<<4, 256>>>(
      dw,
      targetPeerRank,
      dst_offset,
      src_buf,
      src_offset,
      nbytes,
      signalId,
      signalVal,
      counterId,
      counterVal);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindow Offset-Based NVL Put + Counter (No Signal) Test
// =============================================================================

__global__ void nvlOffsetPutCounterKernel(
    DeviceWindow dw,
    int targetPeerRank,
    std::size_t dst_offset,
    LocalBufferRegistration src_buf,
    std::size_t src_offset,
    std::size_t nbytes,
    int counterId,
    uint64_t counterVal) {
  auto group = make_block_group();
  dw.put_counter(
      group,
      targetPeerRank,
      dst_offset,
      src_buf,
      src_offset,
      nbytes,
      counterId,
      counterVal);
}

void testDeviceWindowNvlOffsetPutCounter(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int counterId,
    uint64_t counterVal) {
  NvlOffsetPutDeviceWindowBuffers bufs;
  auto dw = bufs.create_with_offset_put(myRank, nRanks, windowBuf_d);

  int targetPeerRank = (myRank == 0) ? 1 : 0;
  LocalBufferRegistration src_buf{srcBuf_d, srcBufSize, NetworkLKey{}};
  nvlOffsetPutCounterKernel<<<4, 256>>>(
      dw,
      targetPeerRank,
      dst_offset,
      src_buf,
      src_offset,
      nbytes,
      counterId,
      counterVal);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

} // namespace comms::pipes::test
