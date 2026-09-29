// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include "comms/prims/trace/PipesTraceTypes.h"

namespace comms::prims::test {

void launchWriteAndSpin(
    PipesTraceHandle trace,
    volatile int* releaseFlag,
    int numEvents,
    cudaStream_t stream);

void launchWriteEvents(
    PipesTraceHandle trace,
    int numEvents,
    cudaStream_t stream);

void launchWriteEventsMultiBlock(
    PipesTraceHandle trace,
    int numBlocks,
    int eventsPerBlock,
    cudaStream_t stream);

} // namespace comms::prims::test
