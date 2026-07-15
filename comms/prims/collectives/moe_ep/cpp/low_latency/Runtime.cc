// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/low_latency/Runtime.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <folly/CppAttributes.h>

#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

namespace comms::prims::moe_ep {

namespace {

void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

} // namespace

LowLatencyRuntime::LowLatencyRuntime(
    std::shared_ptr<meta::comms::IBootstrap> /*bootstrap*/,
    int rank,
    int numRanks,
    std::size_t numRdmaBytes,
    int numMaxDispatchTokensPerRank,
    int hidden,
    int numExperts,
    int numQpsPerRank,
    void* externalRdmaBuffer)
    : rank_(rank),
      numRanks_(numRanks),
      numExperts_(numExperts),
      numMaxDispatchTokensPerRank_(numMaxDispatchTokensPerRank),
      hidden_(hidden),
      numQpsPerRank_(numQpsPerRank),
      numRdmaBytes_(numRdmaBytes),
      layout_(
          LowLatencyLayout::compute(
              numMaxDispatchTokensPerRank,
              hidden,
              numRanks,
              numExperts)) {
  if (numRanks_ <= 0) {
    throw std::invalid_argument("LowLatencyRuntime: numRanks must be > 0");
  }
  if (rank_ < 0 || rank_ >= numRanks_) {
    throw std::invalid_argument("LowLatencyRuntime: rank out of bounds");
  }
  if (numExperts_ % numRanks_ != 0) {
    throw std::invalid_argument(
        "LowLatencyRuntime: numExperts must be divisible by numRanks");
  }

  // 1. Use external buffer if provided, otherwise allocate.
  if (externalRdmaBuffer != nullptr) {
    rdmaBufferPtr_ = externalRdmaBuffer;
    ownsRdmaBuffer_ = false;
  } else {
#ifdef __HIP_PLATFORM_AMD__
    checkCuda(
        hipExtMallocWithFlags(
            &rdmaBufferPtr_, numRdmaBytes_, hipDeviceMallocUncached),
        "LowLatencyRuntime: hipExtMallocWithFlags(uncached) failed");
#else
    checkCuda(
        cudaMalloc(&rdmaBufferPtr_, numRdmaBytes_),
        "LowLatencyRuntime: cudaMalloc(rdma) failed");
#endif
    ownsRdmaBuffer_ = true;
    checkCuda(
        cudaMemset(rdmaBufferPtr_, 0, numRdmaBytes_),
        "LowLatencyRuntime: cudaMemset(rdma) failed");
  }

  // 2. Persistent atomic counters in workspace. Per-expert + per-rank
  //    counters are dimensioned by `numRanks * numLocalExperts`.
  const int numLocalExperts = numExperts_ / numRanks_;
  const std::size_t counterArrayBytes =
      static_cast<std::size_t>(numRanks_) * numLocalExperts * sizeof(int);

  checkCuda(
      cudaMalloc(&globalAtomicCounter_, sizeof(int)),
      "LowLatencyRuntime: cudaMalloc(globalAtomicCounter) failed");
  checkCuda(
      cudaMemset(globalAtomicCounter_, 0, sizeof(int)),
      "LowLatencyRuntime: cudaMemset(globalAtomicCounter) failed");

  checkCuda(
      cudaMalloc(&atomicCounterPerExpert_, counterArrayBytes),
      "LowLatencyRuntime: cudaMalloc(atomicCounterPerExpert) failed");
  checkCuda(
      cudaMemset(atomicCounterPerExpert_, 0, counterArrayBytes),
      "LowLatencyRuntime: cudaMemset(atomicCounterPerExpert) failed");

  checkCuda(
      cudaMalloc(&atomicFinishCounterPerExpert_, counterArrayBytes),
      "LowLatencyRuntime: cudaMalloc(atomicFinishCounterPerExpert) failed");
  checkCuda(
      cudaMemset(atomicFinishCounterPerExpert_, 0, counterArrayBytes),
      "LowLatencyRuntime: cudaMemset(atomicFinishCounterPerExpert) failed");

  // 3. `next_clean` persistent buffer — used by combine to defer the next
  //    dispatch's clean. Sized at numRanks*numLocalExperts*8B (one int64 slot
  //    per (rank, local expert)).
  nextCleanBufferIntCount_ = numRanks_ * numLocalExperts;
  checkCuda(
      cudaMalloc(
          &nextCleanBuffer_,
          static_cast<std::size_t>(nextCleanBufferIntCount_) *
              sizeof(std::int64_t)),
      "LowLatencyRuntime: cudaMalloc(nextCleanBuffer) failed");
  checkCuda(
      cudaMemset(
          nextCleanBuffer_,
          0,
          static_cast<std::size_t>(nextCleanBufferIntCount_) *
              sizeof(std::int64_t)),
      "LowLatencyRuntime: cudaMemset(nextCleanBuffer) failed");

  // 4. Combine workspace — single int for atomic_clean_flag.
  checkCuda(
      cudaMalloc(&combineWorkspace_, sizeof(int)),
      "LowLatencyRuntime: cudaMalloc(combineWorkspace) failed");
  checkCuda(
      cudaMemset(combineWorkspace_, 0, sizeof(int)),
      "LowLatencyRuntime: cudaMemset(combineWorkspace) failed");

  // 5. Communication stream.
  checkCuda(
      cudaStreamCreate(&commStream_),
      "LowLatencyRuntime: cudaStreamCreate(comm) failed");

  // 6. MultipeerIbgdaTransport setup is deferred to setupIbgda(), which needs
  //    `bootstrap->allGather` for per-peer QP exchange.
  (void)numQpsPerRank_;
}

LowLatencyRuntime::~LowLatencyRuntime() {
  if (commStream_ != nullptr) {
    (void)cudaStreamDestroy(commStream_);
  }
  if (combineWorkspace_ != nullptr) {
    (void)cudaFree(combineWorkspace_);
  }
  if (nextCleanBuffer_ != nullptr) {
    (void)cudaFree(nextCleanBuffer_);
  }
  if (atomicFinishCounterPerExpert_ != nullptr) {
    (void)cudaFree(atomicFinishCounterPerExpert_);
  }
  if (atomicCounterPerExpert_ != nullptr) {
    (void)cudaFree(atomicCounterPerExpert_);
  }
  if (globalAtomicCounter_ != nullptr) {
    (void)cudaFree(globalAtomicCounter_);
  }
  if (peerDataPtrsDevice_ != nullptr) {
    (void)cudaFree(peerDataPtrsDevice_);
  }
  if (rdmaBufferPtr_ != nullptr && ownsRdmaBuffer_) {
    (void)cudaFree(rdmaBufferPtr_);
  }
  // Free IBGDA device-side arrays. ibgdaTransport_ destructor cleans up
  // the host-side transport (QPs, MRs, NIC contexts).
  if (ibgdaPeerRemoteCombineRecvFlagDevice_ != nullptr) {
    (void)cudaFree(ibgdaPeerRemoteCombineRecvFlagDevice_);
  }
  if (ibgdaPeerRemoteCombineRecvXDevice_ != nullptr) {
    (void)cudaFree(ibgdaPeerRemoteCombineRecvXDevice_);
  }
  if (ibgdaPeerRemoteRecvCountDevice_ != nullptr) {
    (void)cudaFree(ibgdaPeerRemoteRecvCountDevice_);
  }
  if (ibgdaPeerRemoteRecvXDevice_ != nullptr) {
    (void)cudaFree(ibgdaPeerRemoteRecvXDevice_);
  }
  if (ibgdaLocalCombineXBufDevice_ != nullptr) {
    (void)cudaFree(ibgdaLocalCombineXBufDevice_);
  }
  if (ibgdaLocalRdmaXBufDevice_ != nullptr) {
    (void)cudaFree(ibgdaLocalRdmaXBufDevice_);
  }
  // ibgdaDeviceTransportPtr_ is the wrapper struct (small) allocated
  // explicitly above; the per-peer P2pIbgdaTransportDevice array it points
  // to is owned by the transport (allocated via buildDeviceTransportsOnGpu),
  // freed by transport dtor.
  if (ibgdaDeviceTransportPtr_ != nullptr) {
    (void)cudaFree(ibgdaDeviceTransportPtr_);
  }
}

void LowLatencyRuntime::setPeerDataPtrs(const std::vector<void*>& peerPtrs) {
  if (peerDataPtrsDevice_ != nullptr) {
    (void)cudaFree(peerDataPtrsDevice_);
    peerDataPtrsDevice_ = nullptr;
  }
  if (peerPtrs.empty()) {
    return;
  }
  checkCuda(
      cudaMalloc(&peerDataPtrsDevice_, peerPtrs.size() * sizeof(void*)),
      "LowLatencyRuntime: cudaMalloc(peerDataPtrsDevice) failed");
  checkCuda(
      cudaMemcpy(
          peerDataPtrsDevice_,
          peerPtrs.data(),
          peerPtrs.size() * sizeof(void*),
          cudaMemcpyHostToDevice),
      "LowLatencyRuntime: cudaMemcpy(peerDataPtrsDevice) failed");
}

namespace {

// Helper: copy a host vector of T into a freshly-allocated device array.
// Returns nullptr if vec is empty.
template <typename T>
T* FOLLY_NULLABLE uploadToDevice(const std::vector<T>& vec, const char* what) {
  if (vec.empty()) {
    return nullptr;
  }
  T* devPtr = nullptr;
  if (cudaMalloc(&devPtr, vec.size() * sizeof(T)) != cudaSuccess) {
    throw std::runtime_error(
        std::string("LowLatencyRuntime: cudaMalloc(") + what + ") failed");
  }
  if (cudaMemcpy(
          devPtr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    (void)cudaFree(devPtr);
    throw std::runtime_error(
        std::string("LowLatencyRuntime: cudaMemcpy(") + what + ") failed");
  }
  return devPtr;
}

// Helper: pad peer-rank-indexed remote buffer vector (size = numRanks - 1,
// excludes self) to a numRanks-indexed array (self slot zero-init). Lets
// the kernel index by global rank without remap.
std::vector<comms::prims::IbgdaRemoteBuffer> padPeerRemote(
    const std::vector<comms::prims::IbgdaRemoteBuffer>& peerBufs,
    int rank,
    int numRanks) {
  std::vector<comms::prims::IbgdaRemoteBuffer> out(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    if (i == rank) {
      continue; // self slot left default-initialized
    }
    int peerIndex = (i < rank) ? i : (i - 1);
    out[i] = peerBufs[peerIndex];
  }
  return out;
}

} // namespace

void LowLatencyRuntime::setupIbgda(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap) {
  if (numRanks_ < 2) {
    // Single-rank: no peers, nothing to set up.
    return;
  }
  if (ibgdaTransport_ != nullptr) {
    throw std::runtime_error("LowLatencyRuntime::setupIbgda: already set up");
  }

  // 1. Construct the transport. NIC discovery happens here (local op).
  // Use the currently-active CUDA device (set by the caller via
  // `torch.cuda.set_device(local_rank)`) — using `rank_` directly only works
  // single-host, since on multi-host rank ≥ num_local_ranks has no matching
  // GPU ordinal on the local host.
  int currentDevice = 0;
  cudaError_t devErr = cudaGetDevice(&currentDevice);
  if (devErr != cudaSuccess) {
    throw std::runtime_error(
        std::string("LowLatencyRuntime::setupIbgda: cudaGetDevice failed: ") +
        cudaGetErrorString(devErr));
  }
  comms::prims::MultipeerIbgdaTransportConfig cfg{};
  cfg.cudaDevice = currentDevice;
  // The kernel routes each block's puts through the QP indexed by its block_id,
  // which it sets to dst_expert_local_idx (range [0, numLocalExperts)), so
  // maxGroups must span the expert-index range. numQpsPerRank_ carries
  // numLocalExperts.
  cfg.maxGroups = numQpsPerRank_;
  // Give each expert as many QPs/NIC as the transport's eager-exchange budget
  // allows (maxGroups * qpsPerBlockPerNic <= kMaxEagerExchangeQpsPerPeerPerNic)
  // to spread per-QP spinlock contention across more QPs and recover
  // throughput. Safe with >1 QP/expert: the data->count (dispatch) and
  // data->flag (combine) ordering is completion-based — the synchronous BNXT
  // put drains each CQE before the per-expert finish counter / __syncthreads
  // that gates the count/flag put — not same-QP FIFO.
  const int qpsPerExpert =
      comms::prims::kMaxEagerExchangeQpsPerPeerPerNic / numQpsPerRank_;
  cfg.qpsPerBlockPerNic = qpsPerExpert > 0 ? qpsPerExpert : 1;
  // Honor NCCL_IB_HCA from env so deployments can pin the NIC family /
  // specific NICs. MultipeerIbgdaTransport reads `cfg.ibHca` and passes
  // it to NicDiscovery as the filter; an empty string means "any NIC
  // by best affinity".
  if (const char* hca = std::getenv("NCCL_IB_HCA"); hca != nullptr) {
    cfg.ibHca = hca;
  }
  ibgdaTransport_ = std::make_unique<comms::prims::MultipeerIbgdaTransport>(
      rank_, numRanks_, std::move(bootstrap), cfg);

  // 2. Exchange QP info (calls bootstrap->allGather under the hood).
  ibgdaTransport_->exchange();

  // 3. Register the LL RDMA buffer with the transport. The four LL regions
  //    (dispatch_recv_x, dispatch_recv_count, combine_recv_x,
  //    combine_recv_flag) live inside numRdmaBytes_ at known offsets; we
  //    register the whole buffer once and use sub-buffer offsets.
  auto localBuf =
      ibgdaTransport_->registerBuffer(rdmaBufferPtr_, numRdmaBytes_);

  // 4. Exchange remote buffer descriptors. One round-trip — every peer learns
  //    every other peer's buffer addr+rkey for the same registered region.
  auto peerRemote = ibgdaTransport_->exchangeBuffer(localBuf);

  // 5. Build per-region per-peer arrays. The kernel indexes into these by
  //    global rank, so pad with self-rank zero-init.
  std::vector<comms::prims::IbgdaRemoteBuffer> peerRemoteRecvX(numRanks_);
  std::vector<comms::prims::IbgdaRemoteBuffer> peerRemoteRecvCount(numRanks_);
  std::vector<comms::prims::IbgdaRemoteBuffer> peerRemoteCombineRecvX(
      numRanks_);
  std::vector<comms::prims::IbgdaRemoteBuffer> peerRemoteCombineRecvFlag(
      numRanks_);
  // padPeerRemote returns exactly numRanks_ entries; the per-rank accesses
  // below use padded.at(i) so an unexpected size throws instead of reading OOB.
  auto padded = padPeerRemote(peerRemote, rank_, numRanks_);
  for (int i = 0; i < numRanks_; ++i) {
    if (i == rank_) {
      continue;
    }
    peerRemoteRecvX[i] = padded.at(i).subBuffer(layout_.dispatchRecvXOffset);
    peerRemoteRecvCount[i] =
        padded.at(i).subBuffer(layout_.dispatchRecvCountOffset);
    peerRemoteCombineRecvX[i] =
        padded.at(i).subBuffer(layout_.combineRecvXOffset);
    peerRemoteCombineRecvFlag[i] =
        padded.at(i).subBuffer(layout_.combineRecvFlagOffset);
  }

  // 6. Build the local-buffer descriptors at the relevant offsets.
  std::vector<comms::prims::IbgdaLocalBuffer> localRdmaXVec{
      localBuf.subBuffer(layout_.dispatchSendXOffset)};
  // Combine's RDMA source is the registered `combine_send_x` region (NOT the
  // user `x` tensor, which isn't RDMA-registered). The combine kernel stages
  // each token's hidden data into a `combine_send_x` slot and RDMAs from
  // there. Register a local descriptor over that region so the kernel's
  // cross-node branch has a valid lkey.
  std::vector<comms::prims::IbgdaLocalBuffer> localCombineXVec{
      localBuf.subBuffer(layout_.combineSendXOffset)};

  // 7. Upload all device-resident arrays.
  ibgdaLocalRdmaXBufDevice_ =
      uploadToDevice(localRdmaXVec, "ibgdaLocalRdmaXBuf");
  ibgdaLocalCombineXBufDevice_ =
      uploadToDevice(localCombineXVec, "ibgdaLocalCombineXBuf");
  ibgdaPeerRemoteRecvXDevice_ =
      uploadToDevice(peerRemoteRecvX, "ibgdaPeerRemoteRecvX");
  ibgdaPeerRemoteRecvCountDevice_ =
      uploadToDevice(peerRemoteRecvCount, "ibgdaPeerRemoteRecvCount");
  ibgdaPeerRemoteCombineRecvXDevice_ =
      uploadToDevice(peerRemoteCombineRecvX, "ibgdaPeerRemoteCombineRecvX");
  ibgdaPeerRemoteCombineRecvFlagDevice_ = uploadToDevice(
      peerRemoteCombineRecvFlag, "ibgdaPeerRemoteCombineRecvFlag");

  // 8. Allocate a MultipeerIbgdaDeviceTransport on device for the kernel.
  //    getDeviceTransport() returns the wrapper by value (it just bundles
  //    the per-peer P2pIbgdaTransportDevice array span). Copy that wrapper
  //    to device memory so the kernel can dereference it.
  auto wrapper = ibgdaTransport_->getDeviceTransport();
  if (cudaMalloc(
          &ibgdaDeviceTransportPtr_,
          sizeof(comms::prims::MultipeerIbgdaDeviceTransport)) != cudaSuccess) {
    throw std::runtime_error(
        "LowLatencyRuntime: cudaMalloc(deviceTransport) failed");
  }
  if (cudaMemcpy(
          ibgdaDeviceTransportPtr_,
          &wrapper,
          sizeof(comms::prims::MultipeerIbgdaDeviceTransport),
          cudaMemcpyHostToDevice) != cudaSuccess) {
    (void)cudaFree(ibgdaDeviceTransportPtr_);
    ibgdaDeviceTransportPtr_ = nullptr;
    throw std::runtime_error(
        "LowLatencyRuntime: cudaMemcpy(deviceTransport) failed");
  }
}

} // namespace comms::prims::moe_ep
