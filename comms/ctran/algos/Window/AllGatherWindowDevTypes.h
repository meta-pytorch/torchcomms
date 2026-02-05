// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/common/GpeKernelSync.h"

namespace ctran::allgatherwindow {

// Kernel args for pipeline sync (wait for GPE to complete a step)
struct PipeSyncKernArgs {
  int stepId;
  ctran::algos::GpeKernelSync* pipeSync;
};

// Kernel args for pipeline end (reset sync flags)
struct PipeEndKernArgs {
  ctran::algos::GpeKernelSync* pipeSync;
};

} // namespace ctran::allgatherwindow
