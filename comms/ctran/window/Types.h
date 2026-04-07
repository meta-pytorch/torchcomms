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
  void* dataAddr{nullptr};
  uint64_t* signalAddr{nullptr};
  // Dedicated signal slot for CUDA graph capture/replay. Isolated from the
  // eager signalAddr so that graph signal values and eager monotonic
  // counters don't interfere.
  uint64_t* graphSignalAddr{nullptr};
  CtranMapperRemoteAccessKey dataRkey{CtranMapperBackend::UNSET};
  CtranMapperRemoteAccessKey signalRkey{CtranMapperBackend::UNSET};
  size_t dataBytes{0};
};
} // namespace ctran::window
