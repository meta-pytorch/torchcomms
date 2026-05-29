// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/ctran/algos/common/GpeKernelSync.h"

// Single-kernel multi-writer test that DELIBERATELY delays one block
// (`delayedBlockIdx`) so other blocks reach phase 2 before that block
// finishes phase 1. `UseFallback` picks between
// `localReduceVectorized` (false) and `localReduceFallback` (true) as
// the phase-1 writer; `copyUnroll<4, T>` is always the phase-2 reader.
//
// Pre-fix, each writer family used its own per-CTA byte ownership that
// disagreed with `copyUnroll`'s in different ways:
//   - vectorized: tail partition disagreed (`UseFallback=false`,
//     count not on numPerBlock boundary).
//   - fallback: main-loop chunk = blockDim * 8 instead of
//     blockDim * (16/sizeof(T)) * 4 (`UseFallback=true`,
//     `sizeof(T) < 8`).
// In either case, the delayed block is the writer for some bytes the
// reader block then reads in phase 2 — and because the writer hasn't
// issued its writes yet, the reader hits the pre-init sentinel.
//
// Post-fix (single shared `ctaPartition<T, 4>` between all three), each
// CTA owns the same byte ranges in writer and reader — no cross-CTA
// dependency — so the test passes regardless of `UseFallback`.
//
// `sync` is the REAL `ctran::algos::GpeKernelSync` allocated via
// `cudaHostAlloc` and pre-posted by the host (post(1) for all workers
// before launch). The kernel uses `GpeKernelSyncDev::complete` /
// `GpeKernelSyncDev::waitPost` only as an intra-CTA release/acquire
// fence between the two phases — each CTA touches only its own slot
// (`completeFlag[blockIdx]`, `postFlag[blockIdx]`), so these calls do
// NOT provide cross-CTA visibility.
//
// Production cross-round visibility in ring AllReduce is mediated by
// the GPE host thread (host's `post`/`isComplete` aggregating across
// all workers on different sync structs across different ops), not by
// per-CTA flags. The test deliberately omits that host relay so the
// byte-ownership invariant between writer and reader is the only
// safety net — which is what we want to assert in isolation.
template <typename T, bool UseFallback>
__global__ void multiWriterTailRaceKernel(
    T* buf,
    T* out,
    const T* src,
    size_t count,
    ctran::algos::GpeKernelSync* sync,
    int delayedBlockIdx,
    unsigned long long delayNs);
