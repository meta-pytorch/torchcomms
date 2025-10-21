// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once
#include <fmt/core.h>

struct CtranNvlRemoteAccessKey {
  // use peerRank and basePtr on peerRank to lookup the imported memory handle
  // in local cache
  int peerRank;
  void* basePtr;

  std::string toString() const {
    return fmt::format("peerRank: {}, basePtr: {}", peerRank, basePtr);
  }
};
