// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::benchmark {

/**
 * Access shape of the single-CTA memory-roofline kernel.
 *
 * Unfused   two live tiles (TileReduceStaged::recv). 3 bytes moved per payload
 *           byte: 2 reads + 1 write.
 * Fused     one live tile plus a streaming tile_load_accumulate, matching
 *           production's RingReduceForwardCopy -> tileReduceCopy. Full tiles
 *           take the UNMASKED tile_load path exactly as production does; only
 *           the remainder tile passes `valid`.
 * ReadOnly  loads only, storing just the final tile so the compiler cannot
 *           eliminate them. ~2 bytes per payload byte -- isolates load issue
 *           rate from store-port contention and the load->add->store chain.
 * WriteOnly One load hoisted out of the loop, then stores only. 1 byte per
 *           payload byte. Together with ReadOnly this brackets the mixed shape:
 *           if time splits across a load port L and a store port S, then
 *           3/rate = 2/L + 1/S. With L = 120 B/clk and the mixed shape at
 *           60 B/clk that predicts S ~= 30 B/clk. A materially higher S
 *           (~60 B/clk) instead implicates write-allocate -- stores costing two
 *           bytes of real traffic each -- rather than a narrow store port.
 * Copy      One load, one store per element, no reduce. 2 bytes per payload
 *           byte at a 1:1 load:store ratio -- more store-heavy than the 2:1
 *           reduce shape. The port model (3/rate = 2/L + 1/S generalised to
 *           n/rate = r/L + w/S) therefore predicts Copy lands BELOW the reduce
 *           shape despite doing strictly less work per byte: ~50 B/clk against
 *           the reduce's ~60. Confirming that inversion is the sharpest
 *           available test of the store-port model.
 * Pipelined Fused with the next tile's loads issued before the current tile's
 *           store, i.e. the double-buffer idiom documented in CopyOp.cuh that
 *           TileReduceStaged::recv does not use. Tests whether the per-tile
 *           latency bubble, rather than issue rate, is what caps R_copy.
 */
enum class CopyOpReduceShape {
  Unfused,
  Fused,
  ReadOnly,
  WriteOnly,
  Copy,
  Pipelined,
};

struct CopyOpReduceTiming {
  float timeUs;
  float payloadGBps;
  /** SM-issued HBM traffic, i.e. R_copy. Shape-dependent multiplier. */
  float memoryGBps;
  /** memoryGBps divided by the MEASURED SM clock -- no assumed boost clock. */
  float bytesPerClock;
  /** SM cycles for one kernel launch, from in-kernel clock64(). */
  unsigned long long cycles;
};

/**
 * Single-CTA memory roofline.
 *
 * @param shape      access shape, see CopyOpReduceShape
 * @param nbytes     payload bytes per operand buffer. Sized below ~85 KB the
 *                   working set is L1-resident, which removes memory latency
 *                   and leaves the pure issue-rate roof.
 * @param iterations timed launches
 * @param threads    threads per block; the launch is always <<<1, threads>>>
 * @param vpt        16-byte vectors per thread per tile -- the memory-level
 *                   parallelism knob. kTileElems = threads * vpt * 4 for fp32.
 * @param repeats    in-kernel passes over the buffer. Needed at L1-resident
 *                   sizes, where a single pass is short enough that per-launch
 *                   overhead would dominate the measurement.
 */
CopyOpReduceTiming runCopyOpReduceBenchmark(
    CopyOpReduceShape shape,
    std::size_t nbytes,
    int iterations,
    int threads,
    int vpt,
    int repeats);

} // namespace comms::prims::benchmark
