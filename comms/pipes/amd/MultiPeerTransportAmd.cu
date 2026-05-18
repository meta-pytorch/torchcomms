#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "MultiPeerTransportAmd.h"
#include "MultipeerIbgdaTransportAmd.h" // @manual

#include <hip/hip_runtime.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <folly/logging/xlog.h>

namespace comms::pipes {

namespace {

#define HIP_CHECK(call)                                               \
  do {                                                                \
    hipError_t err = (call);                                          \
    if (err != hipSuccess) {                                          \
      throw std::runtime_error(                                       \
          std::string(#call) + " failed: " + hipGetErrorString(err)); \
    }                                                                 \
  } while (0)

} // namespace

MultiPeerTransportAmd::MultiPeerTransportAmd(
    int myRank,
    int nRanks,
    int localRank,
    int localSize,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerTransportAmdConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      localRank_(localRank),
      localSize_(localSize),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  detectTopology();

  // Create NVL transport for local peers (filtered: only IPC-open local peers)
  if (!localPeerRanks_.empty()) {
    nvlTransport_ = std::make_unique<MultiPeerNvlTransportAmd>(
        myRank_, nRanks_, bootstrap_, config_.nvlConfig, localPeerRanks_);
  }

  // Create IBGDA transport for remote peers (filtered: only RDMA to remotes)
  if (!remotePeerRanks_.empty()) {
    pipes_gda::MultipeerIbgdaTransportAmdConfig ibgdaCfg{
        .hipDevice = config_.hipDevice,
        .qpDepth = config_.ibgdaQpDepth,
    };
    ibgdaTransport_ = std::make_unique<pipes_gda::MultipeerIbgdaTransportAmd>(
        myRank_, nRanks_, bootstrap_, ibgdaCfg, remotePeerRanks_);
  }
}

MultiPeerTransportAmd::~MultiPeerTransportAmd() {
  if (transportsDevice_) {
    hipFree(transportsDevice_);
    transportsDevice_ = nullptr;
  }
}

void MultiPeerTransportAmd::detectTopology() {
  // Exchange hostnames to determine which peers are on the same node.
  // Same approach as NVIDIA's TopologyDiscovery Tier 2 (hostname match).
  constexpr int kMaxHostnameLen = 256;

  struct RankHostInfo {
    char hostname[kMaxHostnameLen];
  };

  std::vector<RankHostInfo> allHostInfo(nRanks_);
  gethostname(allHostInfo[myRank_].hostname, kMaxHostnameLen);
  allHostInfo[myRank_].hostname[kMaxHostnameLen - 1] = '\0';

  auto result =
      bootstrap_
          ->allGather(
              allHostInfo.data(), sizeof(RankHostInfo), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultiPeerTransportAmd: hostname allGather failed");
  }

  const char* myHostname = allHostInfo[myRank_].hostname;
  for (int r = 0; r < nRanks_; ++r) {
    if (r == myRank_) {
      continue;
    }
    if (strncmp(myHostname, allHostInfo[r].hostname, kMaxHostnameLen) == 0) {
      localPeerRanks_.push_back(r);
    } else {
      remotePeerRanks_.push_back(r);
    }
  }

  if (myRank_ == 0) {
    XLOGF(
        INFO,
        "MultiPeerTransportAmd: rank {} topology: {} local NVL peers, "
        "{} remote IBGDA peers",
        myRank_,
        localPeerRanks_.size(),
        remotePeerRanks_.size());
  }
}

void MultiPeerTransportAmd::exchange() {
  // NVL transport auto-exchanges in constructor (via HipMemHandler)
  // IBGDA transport requires explicit exchange
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
  }
}

DeviceSpan<Transport> MultiPeerTransportAmd::getDeviceTransports() {
  if (!initialized_) {
    initializeTransportsArray();
    initialized_ = true;
  }
  return DeviceSpan<Transport>(
      static_cast<Transport*>(transportsDevice_), nRanks_);
}

void MultiPeerTransportAmd::initializeTransportsArray() {
  HIP_CHECK(hipMalloc(&transportsDevice_, nRanks_ * sizeof(Transport)));

  std::vector<Transport> hostTransports;
  hostTransports.reserve(nRanks_);

  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      // Self transport
      hostTransports.emplace_back(P2pSelfTransportDevice());
    } else {
      // Check if this rank is a local NVL peer
      bool isLocal = false;
      for (int lr : localPeerRanks_) {
        if (lr == rank) {
          isLocal = true;
          break;
        }
      }

      if (isLocal && nvlTransport_) {
        // NVL transport for same-node peer
        hostTransports.emplace_back(nvlTransport_->getP2pTransportDevice(rank));
      } else if (ibgdaTransport_) {
        // IBGDA transport for cross-node peer (stored as void*)
        void* devTransport =
            static_cast<void*>(ibgdaTransport_->getP2pTransportDevice(rank));
        hostTransports.emplace_back(
            Transport(devTransport, Transport::IbgdaAmdTag{}));
      } else {
        throw std::runtime_error(
            "MultiPeerTransportAmd: no transport available for rank " +
            std::to_string(rank));
      }
    }
  }

  HIP_CHECK(hipMemcpy(
      transportsDevice_,
      hostTransports.data(),
      nRanks_ * sizeof(Transport),
      hipMemcpyDefault));
}

} // namespace comms::pipes
#endif
