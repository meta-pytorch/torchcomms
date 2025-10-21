// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/mapper/CtranMapperTypes.h"

namespace ctran::window {

enum OpCountType {
  // generic opCount for any RMA operation in a window;
  // used to track opCount for a given window
  kWinScope = 0,
  // optional operation specific opCount, used to match behaviors on
  // local/remote ranks
  kPut = 1,
  kWaitSignal = 2,
  kSignal = 3,
  kGet = 4,
};

struct RemWinInfo {
  void* addr{nullptr};
  uint64_t* signalAddr{nullptr};
  CtranMapperRemoteAccessKey rkey{CtranMapperBackend::UNSET};
};
} // namespace ctran::window
