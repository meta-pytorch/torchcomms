// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/VcLayout.h"

#include <fmt/core.h>

#include "comms/ctran/utils/Exception.h"
#include "comms/utils/commSpecs.h"

namespace ctran::ib {

VcLayout::VcLayout(int numNics, int maxVcsPerPeer) {
  if (numNics <= 0) {
    throw ctran::utils::Exception(
        fmt::format(
            "CTRAN-IB: VcLayout: numNics must be positive, got {}", numNics),
        commInvalidArgument);
  }
  if (maxVcsPerPeer <= 0) {
    throw ctran::utils::Exception(
        fmt::format(
            "CTRAN-IB: VcLayout: maxVcsPerPeer must be positive, got {} "
            "(callers must normalize legacy sentinel to 1 before constructing)",
            maxVcsPerPeer),
        commInvalidArgument);
  }

  this->maxVcsPerPeer = maxVcsPerPeer;
  vcToActiveDevices.reserve(maxVcsPerPeer);

  if (maxVcsPerPeer >= numNics && maxVcsPerPeer % numNics == 0) {
    // Pinned: each NIC owns maxVcsPerNic VCs (NIC-major layout).
    maxVcsPerNic = maxVcsPerPeer / numNics;
    for (int v = 0; v < maxVcsPerPeer; ++v) {
      vcToActiveDevices.push_back({v / maxVcsPerNic});
    }
  } else if (maxVcsPerPeer < numNics && numNics % maxVcsPerPeer == 0) {
    // Striped: each VC spans K contiguous NICs.
    maxVcsPerNic = 1;
    const int k = numNics / maxVcsPerPeer;
    for (int v = 0; v < maxVcsPerPeer; ++v) {
      std::vector<int> devs;
      devs.reserve(k);
      for (int i = 0; i < k; ++i) {
        devs.push_back(v * k + i);
      }
      vcToActiveDevices.push_back(std::move(devs));
    }
  } else {
    throw ctran::utils::Exception(
        fmt::format(
            "CTRAN-IB: VcLayout: invalid (numNics={}, maxVcsPerPeer={}): "
            "requires either maxVcsPerPeer >= numNics with "
            "maxVcsPerPeer % numNics == 0, or maxVcsPerPeer < numNics with "
            "numNics % maxVcsPerPeer == 0.",
            numNics,
            maxVcsPerPeer),
        commInvalidArgument);
  }
}

std::string VcLayout::describe() const {
  std::string out = fmt::format(
      "maxVcsPerPeer={}, maxVcsPerNic={}", maxVcsPerPeer, maxVcsPerNic);
  for (size_t v = 0; v < vcToActiveDevices.size(); ++v) {
    out += fmt::format(", VC[{}]\u2192[", v);
    for (size_t i = 0; i < vcToActiveDevices[v].size(); ++i) {
      if (i > 0) {
        out += ",";
      }
      out += fmt::format("{}", vcToActiveDevices[v][i]);
    }
    out += "]";
  }
  return out;
}

} // namespace ctran::ib
