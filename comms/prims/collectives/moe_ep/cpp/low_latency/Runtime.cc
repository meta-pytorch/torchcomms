// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/low_latency/Runtime.h"

#include <stdexcept>
#include <string>

#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

#include "comms/common/bootstrap/IBootstrap.h"
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
  //    dispatch's clean. Sized at numRanks*numLocalExperts*8B.
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

  // 6. MultipeerIbgdaTransport setup is deferred until the LL kernel
  //    implementations are wired through. The transport requires
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
  // ibgdaTransport_ destructor handles its own cleanup.
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

} // namespace comms::prims::moe_ep
