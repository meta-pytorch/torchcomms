// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/intranode/Runtime.h"

#include <stdexcept>
#include <string>

// Heavy includes that the public IntranodeRuntime.h forward-declares.
// Must come before any use of GpuMemHandler / MultiPeerNvlTransport types.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"

namespace comms::prims::moe_ep {

namespace {

void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

} // namespace

IntranodeRuntime::IntranodeRuntime(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int rank,
    int numRanks,
    std::size_t numNvlBytes)
    : rank_(rank), numRanks_(numRanks), numNvlBytes_(numNvlBytes) {
  if (numRanks_ <= 0 || numRanks_ > NUM_MAX_NVL_PEERS) {
    throw std::invalid_argument(
        "IntranodeRuntime: numRanks must be in (0, NUM_MAX_NVL_PEERS]");
  }
  if (rank_ < 0 || rank_ >= numRanks_) {
    throw std::invalid_argument("IntranodeRuntime: rank out of bounds");
  }

  // 1. NVL data buffer via GpuMemHandler (kCudaIpcUncached for BNXT
  //    dma-buf compatibility — see N1 in the design plan).
  memHandler_ = std::make_unique<comms::prims::GpuMemHandler>(
      bootstrap,
      rank_,
      numRanks_,
      numNvlBytes_,
      comms::prims::MemSharingMode::kCudaIpcUncached);
  memHandler_->exchangeMemPtrs();

  // 2. Pipes NVL transport — per-peer P2P state, signal slots, barrier
  //    slots. The actual `dataBufferSize` here is for the transport's own
  //    state buffers (chunked-send pipelining) — not the dispatch/combine
  //    data block (which lives in `memHandler_`).
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{
      .pipelineDepth = 4,
      .maxNumChannels = 64,
      .perChannelSize = (1024UL * 1024UL) / 64,
      // numSignalSlots is dimensioned by callers; for Phase 1 dispatch +
      // combine we need ~num_channels*num_peers*2 slots. With num_sms=64,
      // num_channels=32 and 7 peers → ~448 slots. Bump to 1024 to avoid
      // re-tuning per config.
      .p2pSignalCount = 1024,
      .p2pBarrierCount = 64,
  };
  transport_ = std::make_unique<comms::prims::MultiPeerNvlTransport>(
      rank_, numRanks_, bootstrap, nvlConfig);
  transport_->exchange();

  // 3. Persistent workspace.
  checkCuda(
      cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES),
      "IntranodeRuntime: cudaMalloc(workspace) failed");
  checkCuda(
      cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES),
      "IntranodeRuntime: cudaMemset(workspace) failed");

  // 4. Per-runtime atomic counters (4 bytes each; only AMD strictly needs
  //    them, but we always allocate so the kernel signatures don't need
  //    AMD/NVIDIA gating).
  checkCuda(
      cudaMalloc(&dispatchGlobalAtomicCounter_, sizeof(int)),
      "IntranodeRuntime: cudaMalloc(dispatchGlobalAtomicCounter) failed");
  checkCuda(
      cudaMemset(dispatchGlobalAtomicCounter_, 0, sizeof(int)),
      "IntranodeRuntime: cudaMemset(dispatchGlobalAtomicCounter) failed");
  checkCuda(
      cudaMalloc(&combineGlobalAtomicCounter_, sizeof(int)),
      "IntranodeRuntime: cudaMalloc(combineGlobalAtomicCounter) failed");
  checkCuda(
      cudaMemset(combineGlobalAtomicCounter_, 0, sizeof(int)),
      "IntranodeRuntime: cudaMemset(combineGlobalAtomicCounter) failed");

  // 5. Communication stream.
  checkCuda(
      cudaStreamCreate(&commStream_),
      "IntranodeRuntime: cudaStreamCreate(comm) failed");

  // 6. Peer-mapped pointer table. Fill the
  //    host-side staging vector with each peer's NVL data buffer pointer
  //    (self entry == local) and copy to the device-resident pointer array
  //    that kernels consume.
  std::vector<void*> peerPtrsHost(numRanks_, nullptr);
  for (int i = 0; i < numRanks_; ++i) {
    peerPtrsHost[i] = (i == rank_) ? memHandler_->getLocalDeviceMemPtr()
                                   : memHandler_->getPeerDeviceMemPtr(i);
  }
  checkCuda(
      cudaMalloc(&peerDataPtrsDevice_, numRanks_ * sizeof(void*)),
      "IntranodeRuntime: cudaMalloc(peerDataPtrsDevice) failed");
  checkCuda(
      cudaMemcpy(
          peerDataPtrsDevice_,
          peerPtrsHost.data(),
          numRanks_ * sizeof(void*),
          cudaMemcpyHostToDevice),
      "IntranodeRuntime: cudaMemcpy(peerDataPtrsDevice) failed");

  // 7. Barrier-signal pointer table — carved out of the workspace so each
  //    rank's int* slot lives in the peer-mapped IPC region. With `numRanks`
  //    barrier slots × `NUM_MAX_FIFO_SLOTS` rotating positions, each
  //    barrier costs `numRanks * sizeof(int)` of workspace.
  //
  //    On a single-rank smoke setup peer pointers point back at us, so the
  //    star-pattern barrier degenerates to a same-buffer ping (still
  //    correct; just no cross-rank semantics).
  std::vector<int*> barrierPtrsHost(numRanks_, nullptr);
  for (int i = 0; i < numRanks_; ++i) {
    // First slice of each peer's data buffer (offset 0) is reserved for the
    // FIFO barrier slots. The dispatch / combine kernels
    // start their own region after `kNumRanks * kNumRanks * sizeof(int)`,
    // so reusing the same region is a TODO for production multi-rank: in
    // Phase 1 we colocate barrier slots with the rank prefix matrix because
    // notify_dispatch zeros them via the memset stride (num_memset_int).
    barrierPtrsHost[i] = reinterpret_cast<int*>(peerPtrsHost[i]);
  }
  checkCuda(
      cudaMalloc(&barrierSignalPtrsDevice_, numRanks_ * sizeof(int*)),
      "IntranodeRuntime: cudaMalloc(barrierSignalPtrsDevice) failed");
  checkCuda(
      cudaMemcpy(
          barrierSignalPtrsDevice_,
          barrierPtrsHost.data(),
          numRanks_ * sizeof(int*),
          cudaMemcpyHostToDevice),
      "IntranodeRuntime: cudaMemcpy(barrierSignalPtrsDevice) failed");

  // 8. Host-pinned MoE recv counters. notify_dispatch's block 0 writes the
  //    aggregated count back into `*moe_recv_counter_mapped`; the host loop
  //    in Buffer::intranode_dispatch polls `moeRecvCounterHost_`.
  void* hostPtr = nullptr;
  void* devicePtr = nullptr;
  checkCuda(
      cudaHostAlloc(&hostPtr, sizeof(int), cudaHostAllocMapped),
      "IntranodeRuntime: cudaHostAlloc(moeRecvCounter) failed");
  moeRecvCounterHost_ = static_cast<int*>(hostPtr);
  checkCuda(
      cudaHostGetDevicePointer(&devicePtr, hostPtr, 0),
      "IntranodeRuntime: cudaHostGetDevicePointer(moeRecvCounter) failed");
  moeRecvCounterDevice_ = static_cast<int*>(devicePtr);
  *moeRecvCounterHost_ = -1;

  const std::size_t expertCounterBytes =
      kernels::NUM_MAX_LOCAL_EXPERTS * sizeof(int);
  hostPtr = nullptr;
  devicePtr = nullptr;
  checkCuda(
      cudaHostAlloc(&hostPtr, expertCounterBytes, cudaHostAllocMapped),
      "IntranodeRuntime: cudaHostAlloc(moeRecvExpertCounter) failed");
  moeRecvExpertCounterHost_ = static_cast<int*>(hostPtr);
  checkCuda(
      cudaHostGetDevicePointer(&devicePtr, hostPtr, 0),
      "IntranodeRuntime: cudaHostGetDevicePointer(moeRecvExpertCounter) failed");
  moeRecvExpertCounterDevice_ = static_cast<int*>(devicePtr);
}

IntranodeRuntime::IntranodeRuntime(
    int rank,
    int numRanks,
    std::size_t numNvlBytes,
    void* localBuffer,
    const cudaIpcMemHandle_t& /*localHandle*/,
    std::vector<void*> peerDataPtrs)
    : rank_(rank), numRanks_(numRanks), numNvlBytes_(numNvlBytes) {
  if (numRanks_ <= 0 || numRanks_ > NUM_MAX_NVL_PEERS) {
    throw std::invalid_argument(
        "IntranodeRuntime: numRanks must be in (0, NUM_MAX_NVL_PEERS]");
  }
  if (rank_ < 0 || rank_ >= numRanks_) {
    throw std::invalid_argument("IntranodeRuntime: rank out of bounds");
  }
  if (static_cast<int>(peerDataPtrs.size()) != numRanks_) {
    throw std::invalid_argument(
        "IntranodeRuntime: peerDataPtrs size must equal numRanks");
  }
  if (localBuffer == nullptr) {
    throw std::invalid_argument("IntranodeRuntime: localBuffer is null");
  }
  if (peerDataPtrs[rank_] != localBuffer) {
    throw std::invalid_argument(
        "IntranodeRuntime: peerDataPtrs[rank] must equal localBuffer");
  }

  // Skip steps 1 + 2: memHandler_ and transport_ stay nullptr. Caller already
  // allocated the local NVL buffer and exchanged peer pointers Python-side.
  localBufferPtr_ = localBuffer;
  peerDataPtrsHost_ = peerDataPtrs;

  // 3. Persistent workspace.
  checkCuda(
      cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES),
      "IntranodeRuntime: cudaMalloc(workspace) failed");
  checkCuda(
      cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES),
      "IntranodeRuntime: cudaMemset(workspace) failed");

  // 4. Per-runtime atomic counters.
  checkCuda(
      cudaMalloc(&dispatchGlobalAtomicCounter_, sizeof(int)),
      "IntranodeRuntime: cudaMalloc(dispatchGlobalAtomicCounter) failed");
  checkCuda(
      cudaMemset(dispatchGlobalAtomicCounter_, 0, sizeof(int)),
      "IntranodeRuntime: cudaMemset(dispatchGlobalAtomicCounter) failed");
  checkCuda(
      cudaMalloc(&combineGlobalAtomicCounter_, sizeof(int)),
      "IntranodeRuntime: cudaMalloc(combineGlobalAtomicCounter) failed");
  checkCuda(
      cudaMemset(combineGlobalAtomicCounter_, 0, sizeof(int)),
      "IntranodeRuntime: cudaMemset(combineGlobalAtomicCounter) failed");

  // 5. Communication stream.
  checkCuda(
      cudaStreamCreate(&commStream_),
      "IntranodeRuntime: cudaStreamCreate(comm) failed");

  // 6. Peer-mapped pointer table — copy caller-supplied peerDataPtrs
  //    (entry rank == localBuffer; others are cudaIpcOpenMemHandle results)
  //    to the device-resident pointer array that kernels consume.
  checkCuda(
      cudaMalloc(&peerDataPtrsDevice_, numRanks_ * sizeof(void*)),
      "IntranodeRuntime: cudaMalloc(peerDataPtrsDevice) failed");
  checkCuda(
      cudaMemcpy(
          peerDataPtrsDevice_,
          peerDataPtrsHost_.data(),
          numRanks_ * sizeof(void*),
          cudaMemcpyHostToDevice),
      "IntranodeRuntime: cudaMemcpy(peerDataPtrsDevice) failed");

  // 7. Barrier-signal pointer table — barrier slots live in a DEDICATED
  //    region AFTER `numNvlBytes_` of each peer's data buffer. Putting them
  //    at offset 0 collides with notify_dispatch's per-rank/per-expert
  //    prefix matrix writes — the barrier_device spin-wait then never sees
  //    the completion signal and the kernel hangs until GPU watchdog fires.
  //
  //    The Buffer ctor pre-allocates `numNvlBytes_ + NUM_MAX_FIFO_SLOTS *
  //    sizeof(int)` for each peer's IPC region so this offset is valid.
  std::vector<int*> barrierPtrsHost(numRanks_, nullptr);
  for (int i = 0; i < numRanks_; ++i) {
    barrierPtrsHost[i] = reinterpret_cast<int*>(
        reinterpret_cast<std::uint8_t*>(peerDataPtrsHost_[i]) + numNvlBytes_);
  }
  checkCuda(
      cudaMalloc(&barrierSignalPtrsDevice_, numRanks_ * sizeof(int*)),
      "IntranodeRuntime: cudaMalloc(barrierSignalPtrsDevice) failed");
  checkCuda(
      cudaMemcpy(
          barrierSignalPtrsDevice_,
          barrierPtrsHost.data(),
          numRanks_ * sizeof(int*),
          cudaMemcpyHostToDevice),
      "IntranodeRuntime: cudaMemcpy(barrierSignalPtrsDevice) failed");

  // 8. Host-pinned MoE recv counters.
  void* hostPtr = nullptr;
  void* devicePtr = nullptr;
  checkCuda(
      cudaHostAlloc(&hostPtr, sizeof(int), cudaHostAllocMapped),
      "IntranodeRuntime: cudaHostAlloc(moeRecvCounter) failed");
  moeRecvCounterHost_ = static_cast<int*>(hostPtr);
  checkCuda(
      cudaHostGetDevicePointer(&devicePtr, hostPtr, 0),
      "IntranodeRuntime: cudaHostGetDevicePointer(moeRecvCounter) failed");
  moeRecvCounterDevice_ = static_cast<int*>(devicePtr);
  *moeRecvCounterHost_ = -1;

  const std::size_t expertCounterBytes =
      kernels::NUM_MAX_LOCAL_EXPERTS * sizeof(int);
  hostPtr = nullptr;
  devicePtr = nullptr;
  checkCuda(
      cudaHostAlloc(&hostPtr, expertCounterBytes, cudaHostAllocMapped),
      "IntranodeRuntime: cudaHostAlloc(moeRecvExpertCounter) failed");
  moeRecvExpertCounterHost_ = static_cast<int*>(hostPtr);
  checkCuda(
      cudaHostGetDevicePointer(&devicePtr, hostPtr, 0),
      "IntranodeRuntime: cudaHostGetDevicePointer(moeRecvExpertCounter) failed");
  moeRecvExpertCounterDevice_ = static_cast<int*>(devicePtr);
}

IntranodeRuntime::~IntranodeRuntime() {
  if (moeRecvExpertCounterHost_ != nullptr) {
    (void)cudaFreeHost(moeRecvExpertCounterHost_);
  }
  if (moeRecvCounterHost_ != nullptr) {
    (void)cudaFreeHost(moeRecvCounterHost_);
  }
  if (barrierSignalPtrsDevice_ != nullptr) {
    (void)cudaFree(barrierSignalPtrsDevice_);
  }
  if (peerDataPtrsDevice_ != nullptr) {
    (void)cudaFree(peerDataPtrsDevice_);
  }
  if (commStream_ != nullptr) {
    (void)cudaStreamDestroy(commStream_);
  }
  if (combineGlobalAtomicCounter_ != nullptr) {
    (void)cudaFree(combineGlobalAtomicCounter_);
  }
  if (dispatchGlobalAtomicCounter_ != nullptr) {
    (void)cudaFree(dispatchGlobalAtomicCounter_);
  }
  if (workspace_ != nullptr) {
    (void)cudaFree(workspace_);
  }
  // memHandler_ + transport_ destructors handle their own cleanup.
}

void IntranodeRuntime::moveFifoSlots(int n) {
  // Rotate `head` by `n * numRanks_`
  // (each barrier consumes one slot per peer) modulo NUM_MAX_FIFO_SLOTS.
  head_ = (head_ + n * numRanks_) % kernels::NUM_MAX_FIFO_SLOTS;
}

void* IntranodeRuntime::getLocalDataPtr() const {
  if (memHandler_ == nullptr) {
    return localBufferPtr_;
  }
  return memHandler_->getLocalDeviceMemPtr();
}

void* IntranodeRuntime::getPeerDataPtr(int peerRank) const {
  if (memHandler_ == nullptr) {
    if (peerRank < 0 || peerRank >= numRanks_) {
      throw std::out_of_range(
          "IntranodeRuntime::getPeerDataPtr: peerRank out of bounds");
    }
    return peerDataPtrsHost_[peerRank];
  }
  return memHandler_->getPeerDeviceMemPtr(peerRank);
}

comms::prims::MultiPeerNvlTransport& IntranodeRuntime::transport() {
  return *transport_;
}

const comms::prims::MultiPeerNvlTransport& IntranodeRuntime::transport() const {
  return *transport_;
}

comms::prims::GpuMemHandler& IntranodeRuntime::memHandler() {
  return *memHandler_;
}

} // namespace comms::prims::moe_ep
