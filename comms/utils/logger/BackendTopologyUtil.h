// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <optional>
#include <string>
#include <vector>

class BackendTopologyUtil {
 public:
  struct Topology {
    // Shared fate zone (group of regions), e.g. sfz_al1
    std::string sfz;
    // Region e.g. pci
    std::string region;
    // Building e.g. pci2
    std::string dc;
    // AI Zone e.g. pci2.2A.z085
    std::string zone;
    // Rack training switch e.g. rtsw172.c085.f00.pci2
    std::string rtsw;
    // Host name, e.g. twshared43957.01.pci2.facebook.com
    std::string host;
    // scale-up fields
    struct {
      // scale-up unit e.g. nao5.z081.u003
      std::string unit;
      // scale-up domain e.g. nvso.nao5.z081.d019
      std::string domain;
      // scale-up rack serial
      std::string rack;
    } scaleUp;

    // List containing all the scopes in hierarchical order
    // e.g. [sfz, region, dc, zone, scale-up domain, rack, host]
    std::vector<std::string> fullScopes;
  };

  static std::optional<Topology> getBackendTopology(
      const std::string& fileName);
};
