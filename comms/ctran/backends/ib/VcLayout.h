// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

namespace ctran::ib {

// Uniform per-VC NIC assignment, valid for both legacy single-VC and
// multi-VC configurations.
//
// Two layout modes are supported:
//   - Pinned (maxVcsPerPeer >= numNics, maxVcsPerPeer % numNics == 0):
//     each NIC owns maxVcsPerNic = maxVcsPerPeer/numNics VCs, NIC-major.
//   - Striped (maxVcsPerPeer < numNics, numNics % maxVcsPerPeer == 0):
//     each VC spans K = numNics/maxVcsPerPeer contiguous NICs;
//     maxVcsPerNic = 1.
//
// `vcToActiveDevices[v]` is the set of IB device indices (active NICs)
// that VC v owns. CtranIbVirtualConn consumes this vector directly.
struct VcLayout {
  // Default-constructed layout is empty; populate via the
  // (numNics, maxVcsPerPeer) ctor.
  VcLayout() = default;

  // Build a VcLayout for the given (numNics, maxVcsPerPeer) configuration.
  //
  // Throws ctran::utils::Exception(commInvalidArgument) when:
  //   - numNics <= 0
  //   - maxVcsPerPeer <= 0 (callers must normalize the legacy sentinel to 1
  //     before constructing)
  //   - neither divisibility relationship between numNics and maxVcsPerPeer
  //     holds.
  VcLayout(int numNics, int maxVcsPerPeer);

  int maxVcsPerPeer{0};
  int maxVcsPerNic{0};
  std::vector<std::vector<int>> vcToActiveDevices;

  // Single-line summary string for logging (init).
  std::string describe() const;
};

} // namespace ctran::ib
