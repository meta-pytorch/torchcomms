// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstddef>

namespace comms::utils::cumem {

/**
 *
 * This function checks if memory was allocated with cuMem.
 * If yes, it checks it it was allocated with several memory allocations.
 *
 */
bool isBackedByMultipleCuMemAllocations(
    const void* ptr,
    const int devId,
    const size_t len);

} // namespace comms::utils::cumem
