// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// RoCE source-GID discovery for the ionic (AMD Pensando) NIC.
//
// The shared MultiPeerIbTransport defaults to GID index 3 (kDefaultGidIndex),
// which matches mlx5/bnxt's dual-stack layout (routable RoCEv2 GID at index 3).
// ionic lays its table out differently: index 0 is an fe80:: link-local GID,
// the routable RoCEv2 GID is at index 1, and index 3 is an empty slot. A QP
// built with an empty source GID makes ibv_modify_qp(RTR) fail with ENODATA, so
// ionic must locate its real GID.

#include <fstream>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <glog/logging.h>

#include "comms/ctran/ibverbx/Ibverbx.h"

namespace comms::prims {

// True if a GID slot is unpopulated (all-zero) or an fe80:: link-local address.
// Such an entry cannot be used as the source GID for a routed RoCE connection.
inline bool gidIsEmptyOrLinkLocal(const ibverbx::ibv_gid& gid) {
  bool allZero = true;
  for (int b = 0; b < 16; ++b) {
    if (gid.raw[b] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return true;
  }
  return gid.raw[0] == 0xfe && gid.raw[1] == 0x80; // fe80:: link-local
}

// Read the RoCE GID type from sysfs. ibverbx exposes no gid_type query, so we
// read /sys/class/infiniband/<dev>/ports/<port>/gid_attrs/types/<index>, which
// prints e.g. "RoCE v2" / "IB/RoCE v1". Returns true for a v2 entry.
inline bool
gidIndexIsRoceV2(const std::string& deviceName, int port, int index) {
  std::ifstream f(
      fmt::format(
          "/sys/class/infiniband/{}/ports/{}/gid_attrs/types/{}",
          deviceName,
          port,
          index));
  if (!f.is_open()) {
    return false;
  }
  std::string type;
  std::getline(f, type);
  return type.find("v2") != std::string::npos;
}

// Scan the port's GID table for the best routable RoCE GID: skip empty and
// fe80:: link-local entries, prefer RoCE v2 over v1 (the standard RoCE
// source-GID selection rule). Returns the chosen index, or -1 if none found.
template <typename Symbols, typename Ctx>
int discoverRoceGidIndex(
    Symbols& symbols,
    Ctx* ctx,
    const std::string& deviceName,
    int port,
    int gidTblLen) {
  int firstGlobal = -1;
  int firstV2 = -1;
  for (int i = 0; i < gidTblLen; ++i) {
    ibverbx::ibv_gid gid{};
    if (symbols.ibv_internal_query_gid(ctx, port, i, &gid) != 0) {
      continue;
    }
    if (gidIsEmptyOrLinkLocal(gid)) {
      continue;
    }
    if (firstGlobal < 0) {
      firstGlobal = i;
    }
    if (firstV2 < 0 && gidIndexIsRoceV2(deviceName, port, i)) {
      firstV2 = i;
    }
  }
  return firstV2 >= 0 ? firstV2 : firstGlobal;
}

// Ensure `gidIndex`/`localGid` name a routable RoCEv2 source GID. When the
// caller did not pin an explicit index and the current slot is empty/link-local
// (ionic's default index 3 is empty), scan the table, then update `gidIndex` +
// `localGid` to the discovered GID. Throws if no routable GID can be found.
template <typename Symbols, typename Ctx>
void resolveRoceGidIndex(
    Symbols& symbols,
    Ctx* ctx,
    const std::string& deviceName,
    int port,
    bool callerPinnedIndex,
    int& gidIndex,
    ibverbx::ibv_gid& localGid) {
  if (!callerPinnedIndex && gidIsEmptyOrLinkLocal(localGid)) {
    ibverbx::ibv_port_attr probe{};
    int tblLen = (symbols.ibv_internal_query_port(ctx, port, &probe) == 0 &&
                  probe.gid_tbl_len > 0)
        ? probe.gid_tbl_len
        : 256;
    int discovered =
        discoverRoceGidIndex(symbols, ctx, deviceName, port, tblLen);
    ibverbx::ibv_gid discoveredGid{};
    if (discovered >= 0 && discovered != gidIndex &&
        symbols.ibv_internal_query_gid(ctx, port, discovered, &discoveredGid) ==
            0) {
      LOG(INFO) << "MultiPeerIbTransport: NIC " << deviceName << " GID index "
                << gidIndex
                << " is empty/link-local; auto-discovered routable RoCEv2 GID "
                   "at index "
                << discovered;
      gidIndex = discovered;
      localGid = discoveredGid;
    }
  }
  if (gidIsEmptyOrLinkLocal(localGid)) {
    throw std::runtime_error(
        fmt::format(
            "No routable RoCE GID found on NIC {} (index {} is empty/link-local and "
            "auto-discovery found no global RoCEv2 GID)",
            deviceName,
            gidIndex));
  }
}

} // namespace comms::prims
