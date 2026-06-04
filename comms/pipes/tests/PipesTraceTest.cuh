// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/PipesTraceTypes.h"

namespace comms::pipes::test {

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

} // namespace comms::pipes::test
