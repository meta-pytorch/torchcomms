// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(ENABLE_PIPES)

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/DirectNvlTypes.h"

namespace ctran::allgatherp::hierarchical_pipes {

inline constexpr int kBlockSize = 512;

struct KernArgs {
  comms::pipes::HierarchicalAllgatherFusedArgs args{};
  comms::pipes::Timeout timeout{};
};

} // namespace ctran::allgatherp::hierarchical_pipes

#endif // ENABLE_PIPES
