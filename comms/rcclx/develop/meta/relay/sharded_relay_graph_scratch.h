/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include "comm.h"

namespace rcclx {
namespace relay {

/**
 * Scratch allocation for a relay collective that is being captured into a HIP
 * graph.
 *
 * Each collective keeps its own ScratchBufferCache, and all four allocate with
 * hipMallocAsync on the caller's stream. That is correct right up until the
 * stream is capturing, and then it breaks three separate ways:
 *
 *   1. hipMallocAsync inside a capture records a graph allocation node. Its
 *      address is only valid while that graph is executing, but the cache goes
 *      on handing the same pointer to later uncaptured calls.
 *   2. A cache growth inside a capture records the hipFreeAsync of the previous
 *      pointer as a free node, so the buffer is freed on every replay and the
 *      second replay is a double free.
 *   3. A cache growth AFTER a capture hipFreeAsyncs a pointer the graph has
 *      already baked into its nodes, returning it to the pool while the graph
 * is still live and replayable.
 *
 * This hands a capture a buffer that is allocated OUTSIDE the capture, so no
 * allocation node is recorded; is private to that graph, so no growth on
 * another stream or graph can free it underneath; and is reclaimed once the
 * graph is destroyed.
 *
 * Only the graph path lives here. Uncaptured calls stay on each collective's
 * own cache, unchanged -- this is a correctness fix, and there is no reason to
 * also perturb the path that carries all of the measured performance.
 *
 * `cacheTag` distinguishes the calling cache so the four per-collective caches
 * do not share entries; pass the cache instance address. `key` is that cache's
 * own buffer key. Returns nullptr on failure, which the caller must treat the
 * same way it treats an allocation failure today.
 */
void* graphScratchGet(
    const void* cacheTag,
    int key,
    size_t requiredBytes,
    cudaStream_t stream,
    struct ncclCudaGraph graph);

} // namespace relay
} // namespace rcclx
