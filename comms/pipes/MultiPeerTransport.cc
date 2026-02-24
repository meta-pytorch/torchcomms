// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerTransport.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/bootstrap/NvlBootstrapAdapter.h"

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

} // namespace

MultiPeerTransport::MultiPeerTransport(
    int myRank,
    int nRanks,
    int deviceId,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const MultiPeerTransportConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      deviceId_(deviceId),
      bootstrap_(std::move(bootstrap)) {
  TopologyDiscovery topoDiscovery;
  auto topo = topoDiscovery.discover(
      myRank_, nRanks_, deviceId_, *bootstrap_, config.topoConfig);
  nvlPeerRanks_ = std::move(topo.nvlPeerRanks);
  globalToNvlLocal_ = std::move(topo.globalToNvlLocal);

  // Derive fields from the slim TopologyResult.
  nvlNRanks_ = static_cast<int>(nvlPeerRanks_.size()) + 1;
  nvlLocalRank_ = globalToNvlLocal_.at(myRank_);

  typePerRank_.resize(nRanks_);
  for (int r = 0; r < nRanks_; ++r) {
    if (r == myRank_) {
      typePerRank_[r] = TransportType::SELF;
    } else if (globalToNvlLocal_.count(r)) {
      typePerRank_[r] = TransportType::P2P_NVL;
    } else {
      typePerRank_[r] = TransportType::P2P_IBGDA;
    }
  }

  for (int r = 0; r < nRanks_; ++r) {
    if (r != myRank_) {
      ibgdaPeerRanks_.push_back(r);
    }
  }

  // Create NVLink sub-transport with NvlBootstrapAdapter
  if (!nvlPeerRanks_.empty()) {
    std::vector<int> localRankToCommRank(nvlNRanks_);
    for (const auto& [globalRank, nvlLocal] : globalToNvlLocal_) {
      localRankToCommRank[nvlLocal] = globalRank;
    }

    nvlBootstrapAdapter_ = std::make_shared<NvlBootstrapAdapter>(
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

MultiPeerTransport::~MultiPeerTransport() {
  free_device_handle();
}

void MultiPeerTransport::exchange() {
  if (nvlTransport_) {
    nvlTransport_->exchange();
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
  }
  build_device_handle();
}

TransportType MultiPeerTransport::get_transport_type(int peerRank) const {
  return typePerRank_[peerRank];
}

bool MultiPeerTransport::is_nvl_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_NVL;
}

bool MultiPeerTransport::is_ibgda_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_IBGDA;
}

P2pNvlTransportDevice MultiPeerTransport::get_p2p_nvl_transport_device(
    int globalPeerRank) const {
  if (!nvlTransport_) {
    throw std::runtime_error(
        "get_p2p_nvl_transport_device: NVL transport not available");
  }
  int nvlLocalPeerRank = globalToNvlLocal_.at(globalPeerRank);
  return nvlTransport_->getP2pTransportDevice(nvlLocalPeerRank);
}

P2pIbgdaTransportDevice* MultiPeerTransport::get_p2p_ibgda_transport_device(
    int globalPeerRank) const {
  if (!ibgdaTransport_) {
    throw std::runtime_error(
        "get_p2p_ibgda_transport_device: IBGDA transport not available (nRanks == 1?)");
  }
  return ibgdaTransport_->getP2pTransportDevice(globalPeerRank);
}

P2pSelfTransportDevice MultiPeerTransport::get_p2p_self_transport_device()
    const {
  return P2pSelfTransportDevice{};
}

MultiPeerDeviceHandle MultiPeerTransport::get_device_handle() const {
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransport::get_device_handle() called before exchange()");
  }

  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {transportsGpu_, static_cast<uint32_t>(nRanks_)},
      static_cast<int>(nvlPeerRanks_.size()),
      static_cast<int>(ibgdaPeerRanks_.size()),
  };
}

void MultiPeerTransport::build_device_handle() {
  if (deviceHandleBuilt_) {
    free_device_handle();
  }

  // Build a host-side Transport array indexed by global rank, then cudaMemcpy
  // it to GPU. Since Transport has deleted copy constructor, we allocate raw
  // memory and use placement new.
  const size_t arrayBytes = nRanks_ * sizeof(Transport);
  auto* transportsHost = static_cast<Transport*>(
      std::aligned_alloc(alignof(Transport), arrayBytes));
  if (!transportsHost) {
    throw std::runtime_error("Failed to allocate host Transport array");
  }

  // Get IBGDA GPU pointers per-peer via getP2pTransportDevice()
  // which returns device-memory addresses suitable for Transport.p2p_ibgda

  for (int r = 0; r < nRanks_; ++r) {
    switch (typePerRank_[r]) {
      case TransportType::SELF:
        new (&transportsHost[r]) Transport(P2pSelfTransportDevice{});
        break;

      case TransportType::P2P_NVL: {
        int nvlLocal = globalToNvlLocal_.at(r);
        auto nvlDev = nvlTransport_->getP2pTransportDevice(nvlLocal);
        new (&transportsHost[r]) Transport(nvlDev);
        break;
      }

      case TransportType::P2P_IBGDA: {
        P2pIbgdaTransportDevice* devPtr = ibgdaTransport_
            ? ibgdaTransport_->getP2pTransportDevice(r)
            : nullptr;
        new (&transportsHost[r]) Transport(devPtr);
        break;
      }
    }
  }

  // Allocate GPU memory and raw-copy the Transport array.
  // Transport union members are standard-layout + trivially destructible,
  // so raw byte copy via cudaMemcpy produces valid device-side objects.
  CUDA_CHECK(cudaMalloc(&transportsGpu_, arrayBytes));
  CUDA_CHECK(cudaMemcpy(
      transportsGpu_, transportsHost, arrayBytes, cudaMemcpyHostToDevice));

  // Destroy host-side Transport objects and free
  for (int r = 0; r < nRanks_; ++r) {
    transportsHost[r].~Transport();
  }
  std::free(transportsHost);

  deviceHandleBuilt_ = true;
}

void MultiPeerTransport::free_device_handle() {
  if (transportsGpu_) {
    cudaFree(transportsGpu_);
    transportsGpu_ = nullptr;
  }
  deviceHandleBuilt_ = false;
}

} // namespace comms::pipes
