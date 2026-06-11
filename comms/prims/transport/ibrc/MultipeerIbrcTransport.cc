// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace comms::prims {

namespace {
[[noreturn]] void ibrcUnimplemented(const char* what) {
  throw std::runtime_error(
      std::string("MultipeerIbrcTransport: ") + what +
      " is not implemented yet");
}
} // namespace

MultipeerIbrcTransport::MultipeerIbrcTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbTransportConfig& config)
    : MultiPeerIbTransport<MultipeerIbrcTransport>(
          myRank,
          nRanks,
          std::move(bootstrap),
          config) {}

void MultipeerIbrcTransport::exchange() {
  ibrcUnimplemented("exchange()");
}

void MultipeerIbrcTransport::doMaterializePeer(int /*peerRank*/) {
  // Lazy per-peer materialization: create this peer's lane resources, do the
  // bilateral QP + buffer exchange, and build its device-transport slot. Not
  // yet implemented.
  ibrcUnimplemented("doMaterializePeer()");
}

void MultipeerIbrcTransport::cleanupPeerOnFailure(int /*peerIndex*/) {
  // Per-peer rollback hook. The base invokes this from a catch(...) block
  // during failure rollback, where throwing would call std::terminate — so it
  // must stay no-throw. No-op until lazy materialization is implemented.
}

} // namespace comms::prims
