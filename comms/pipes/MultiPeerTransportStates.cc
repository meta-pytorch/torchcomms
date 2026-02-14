// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerTransportStates.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <unistd.h>

#include <cuda_runtime.h>

#include "comms/pipes/IntraNodeBootstrapAdapter.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/NvmlFabricInfo.h"

namespace comms::pipes {

namespace {

#define CUDA_CHECK(cmd)                                                    \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

/**
 * Per-rank info exchanged during topology discovery.
 */
struct RankInfo {
  char hostname[64];
  int cudaDevice;
  NvmlFabricInfo fabricInfo;
};

} // namespace

MultiPeerTransportStates::MultiPeerTransportStates(
    int myRank,
    int nRanks,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const MultiPeerTransportStatesConfig& config)
    : myRank_(myRank), nRanks_(nRanks), bootstrap_(std::move(bootstrap)) {
  discover_topology();

  // Create NVLink sub-transport with IntraNodeBootstrapAdapter
  if (!nvlPeerRanks_.empty()) {
    std::vector<int> localRankToCommRank(nvlNRanks_);
    for (const auto& [globalRank, nvlLocal] : globalToNvlLocal_) {
      localRankToCommRank[nvlLocal] = globalRank;
    }

    nvlBootstrapAdapter_ = std::make_shared<IntraNodeBootstrapAdapter>(
        bootstrap_, std::move(localRankToCommRank));

    nvlTransport_ = std::make_unique<MultiPeerNvlTransport>(
        nvlLocalRank_, nvlNRanks_, nvlBootstrapAdapter_, config.nvlConfig);
  }

  // Always create IBGDA transport — it is the universal fallback for all peers.
  // NVL is preferred when available, but IBGDA covers every non-self rank.
  if (nRanks_ > 1) {
    ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
        myRank_, nRanks_, bootstrap_, config.ibgdaConfig);
  }
}

MultiPeerTransportStates::~MultiPeerTransportStates() {
  free_device_handle();
}

void MultiPeerTransportStates::discover_topology() {
  int myDevice;
  CUDA_CHECK(cudaGetDevice(&myDevice));

  // AllGather per-rank info: hostname, CUDA device, and MNNVL fabric info.
  std::vector<RankInfo> allInfo(nRanks_);
  auto& myInfo = allInfo[myRank_];

  myInfo.cudaDevice = myDevice;
  gethostname(myInfo.hostname, sizeof(myInfo.hostname));

  // Query MNNVL fabric info via NVML (dlopen).
  char busId[detail::kNvmlBusIdLen];
  CUDA_CHECK(cudaDeviceGetPCIBusId(busId, detail::kNvmlBusIdLen, myDevice));
  myInfo.fabricInfo = query_nvml_fabric_info(busId);

  bootstrap_->allGather(allInfo.data(), sizeof(RankInfo), myRank_, nRanks_)
      .get();

  typePerRank_.resize(nRanks_);

  // Two-tier NVLink detection (following NCCL's MNNVL pattern):
  //
  //   Tier 1 — MNNVL fabric (GB200):
  //     If both ranks have fabric info and share the same clusterUuid
  //     + cliqueId, they are in the same NVLink domain → NVL peer.
  //
  //   Tier 2 — Same-host + cudaDeviceCanAccessPeer (H100 and earlier):
  //     If both ranks are on the same hostname, query CUDA peer access
  //     using local device ordinals (which are valid within a host).
  //
  //   Fallback → IBGDA.
  //
  std::vector<int> nvlGroupGlobalRanks;
  nvlGroupGlobalRanks.push_back(myRank_);

  for (int r = 0; r < nRanks_; ++r) {
    if (r == myRank_) {
      continue;
    }

    // Tier 1: MNNVL fabric match (GB200 cross-host NVLink).
    if (myInfo.fabricInfo.available && allInfo[r].fabricInfo.available &&
        std::memcmp(
            myInfo.fabricInfo.clusterUuid,
            allInfo[r].fabricInfo.clusterUuid,
            NvmlFabricInfo::kUuidLen) == 0 &&
        myInfo.fabricInfo.cliqueId == allInfo[r].fabricInfo.cliqueId) {
      nvlGroupGlobalRanks.push_back(r);
      continue;
    }

    // Tier 2: Same hostname → local cudaDeviceCanAccessPeer (H100).
    if (std::strcmp(myInfo.hostname, allInfo[r].hostname) == 0) {
      int canAccess = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(
          &canAccess, myInfo.cudaDevice, allInfo[r].cudaDevice));
      if (canAccess) {
        nvlGroupGlobalRanks.push_back(r);
        continue;
      }
    }

    // Fallback: preferred transport is IBGDA.
    typePerRank_[r] = TransportType::P2P_IBGDA;
  }

  // Sort NVL group by global rank so that NVL local indices are consistent
  // across all ranks (and match MPI local rank ordering).
  std::sort(nvlGroupGlobalRanks.begin(), nvlGroupGlobalRanks.end());

  for (int i = 0; i < static_cast<int>(nvlGroupGlobalRanks.size()); ++i) {
    int gRank = nvlGroupGlobalRanks[i];
    globalToNvlLocal_[gRank] = i;
    if (gRank == myRank_) {
      typePerRank_[gRank] = TransportType::SELF;
    } else {
      nvlPeerRanks_.push_back(gRank);
      typePerRank_[gRank] = TransportType::P2P_NVL;
    }
  }

  nvlLocalRank_ = globalToNvlLocal_[myRank_];
  nvlNRanks_ = static_cast<int>(nvlGroupGlobalRanks.size());

  // IBGDA is universal — populate ibgdaPeerRanks_ with ALL non-self ranks.
  for (int r = 0; r < nRanks_; ++r) {
    if (r != myRank_) {
      ibgdaPeerRanks_.push_back(r);
    }
  }
}

void MultiPeerTransportStates::exchange() {
  if (nvlTransport_) {
    nvlTransport_->exchange();
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
  }
  build_device_handle();
}

TransportType MultiPeerTransportStates::get_transport_type(int peerRank) const {
  return typePerRank_[peerRank];
}

bool MultiPeerTransportStates::is_nvl_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_NVL;
}

bool MultiPeerTransportStates::is_ibgda_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_IBGDA;
}

P2pNvlTransportDevice MultiPeerTransportStates::get_p2p_nvl_transport_device(
    int globalPeerRank) const {
  int nvlLocalPeerRank = globalToNvlLocal_.at(globalPeerRank);
  return nvlTransport_->getP2pTransportDevice(nvlLocalPeerRank);
}

P2pIbgdaTransportDevice*
MultiPeerTransportStates::get_p2p_ibgda_transport_device(
    int globalPeerRank) const {
  return ibgdaTransport_->getP2pTransportDevice(globalPeerRank);
}

P2pSelfTransportDevice MultiPeerTransportStates::get_p2p_self_transport_device()
    const {
  return P2pSelfTransportDevice{};
}

MultiPeerDeviceHandle MultiPeerTransportStates::get_device_handle() const {
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransportStates::get_device_handle() called before exchange()");
  }

  P2pNvlTransportDevice* nvlPtr = nvlTransportsGpu_;
  uint32_t nvlSize = static_cast<uint32_t>(nvlPeerRanks_.size());

  P2pIbgdaTransportDevice* ibgdaPtr = nullptr;
  uint32_t ibgdaSize = 0;
  if (ibgdaTransport_) {
    auto ibgdaDevTransport = ibgdaTransport_->getDeviceTransport();
    ibgdaPtr = ibgdaDevTransport.peerTransports.data();
    ibgdaSize = ibgdaDevTransport.peerTransports.size();
  }

  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {typePerRankGpu_, static_cast<uint32_t>(nRanks_)},
      {nvlPtr, nvlSize},
      {ibgdaPtr, ibgdaSize},
      {globalToNvlIndexGpu_, static_cast<uint32_t>(nRanks_)},
      {globalToIbgdaIndexGpu_, static_cast<uint32_t>(nRanks_)},
  };
}

void MultiPeerTransportStates::build_device_handle() {
  // 1. Build NVL transport array on GPU
  //    We collect P2pNvlTransportDevice handles from nvlTransport_ for each
  //    peer and copy them to GPU memory.
  if (nvlTransport_ && !nvlPeerRanks_.empty()) {
    std::vector<P2pNvlTransportDevice> nvlTransportsHost;
    nvlTransportsHost.reserve(nvlPeerRanks_.size());

    for (int globalRank : nvlPeerRanks_) {
      int nvlLocalPeerRank = globalToNvlLocal_.at(globalRank);
      nvlTransportsHost.push_back(
          nvlTransport_->getP2pTransportDevice(nvlLocalPeerRank));
    }

    size_t nvlArraySize =
        nvlTransportsHost.size() * sizeof(P2pNvlTransportDevice);
    CUDA_CHECK(cudaMalloc(&nvlTransportsGpu_, nvlArraySize));
    CUDA_CHECK(cudaMemcpy(
        nvlTransportsGpu_,
        nvlTransportsHost.data(),
        nvlArraySize,
        cudaMemcpyHostToDevice));
  }

  // 2. Build mapping arrays on host
  //    NVL: map global rank → index into nvlTransportsGpu_
  //    The array uses the order from nvlPeerRanks_ (sorted by global rank).
  std::vector<int> globalToNvlIndex(nRanks_, -1);
  std::vector<int> globalToIbgdaIndex(nRanks_, -1);

  for (size_t i = 0; i < nvlPeerRanks_.size(); ++i) {
    int globalRank = nvlPeerRanks_[i];
    globalToNvlIndex[globalRank] = static_cast<int>(i);
  }
  // IBGDA covers all peers — build index for every non-self rank
  // using the standard skip-self formula.
  for (int r = 0; r < nRanks_; ++r) {
    if (r != myRank_) {
      globalToIbgdaIndex[r] = (r < myRank_) ? r : (r - 1);
    }
  }

  // 3. Copy all mapping arrays to GPU
  CUDA_CHECK(cudaMalloc(&typePerRankGpu_, nRanks_ * sizeof(TransportType)));
  CUDA_CHECK(cudaMemcpy(
      typePerRankGpu_,
      typePerRank_.data(),
      nRanks_ * sizeof(TransportType),
      cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&globalToNvlIndexGpu_, nRanks_ * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(
      globalToNvlIndexGpu_,
      globalToNvlIndex.data(),
      nRanks_ * sizeof(int),
      cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&globalToIbgdaIndexGpu_, nRanks_ * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(
      globalToIbgdaIndexGpu_,
      globalToIbgdaIndex.data(),
      nRanks_ * sizeof(int),
      cudaMemcpyHostToDevice));

  deviceHandleBuilt_ = true;
}

void MultiPeerTransportStates::free_device_handle() {
  if (nvlTransportsGpu_) {
    cudaFree(nvlTransportsGpu_);
    nvlTransportsGpu_ = nullptr;
  }
  if (typePerRankGpu_) {
    cudaFree(typePerRankGpu_);
    typePerRankGpu_ = nullptr;
  }
  if (globalToNvlIndexGpu_) {
    cudaFree(globalToNvlIndexGpu_);
    globalToNvlIndexGpu_ = nullptr;
  }
  if (globalToIbgdaIndexGpu_) {
    cudaFree(globalToIbgdaIndexGpu_);
    globalToIbgdaIndexGpu_ = nullptr;
  }
  deviceHandleBuilt_ = false;
}

} // namespace comms::pipes
