// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

// Single-kernel multi-writer test that DELIBERATELY delays block 0 so
// block 1 reaches phase 2 before block 0 finishes phase 1. With the
// pre-fix tail in `localReduceVectorized`, block 0 is the writer for
// part of the byte range that block 1 reads in `copyUnroll`'s tail.
// Because block 0 hasn't issued its writes yet (it's sleeping), block 1
// reads the pre-init sentinel from L2 — `out` then carries that
// sentinel instead of `src`, and the test fails.
//
// With the fix (single-designated-CTA tail in localReduce matching
// copyUnroll's), block 1 owns its entire copyUnroll-tail range in BOTH
// writers, so block 1's phase 2 read returns block 1's own phase 1
// write — no dependency on block 0 — and the test passes.
//
// `Unroll16` controls the unroll factor passed to `copyUnroll<Unroll16, T>`
// in phase 2. `localReduceVectorized` in phase 1 uses `kUnroll=4`
// internally. When `Unroll16 != 4`, the per-CTA partitions of the two
// writers disagree and the block-0 delay surfaces cross-CTA stale reads
// even at counts where copyUnroll<4>'s tail is empty.
template <typename T, int Unroll16>
__global__ void multiWriterTailRaceKernel(
    T* buf,
    T* out,
    const T* src,
    size_t count,
    int* perCtaFlag,
    unsigned long long block0DelayNs);
