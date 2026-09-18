// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "comms/utils/collstats/CollStatsBank.h"
#include "comms/utils/collstats/CollStatsFinalize.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// The globally-resident stats block for one communicator. A collective kernel
// reaches it through a single pointer hung off the per-comm device state (which
// is itself copied into shared memory per block), so everything the device
// mutates atomically across blocks lives here in global memory, never in the
// shared-mem copy. The double-buffered bank and the per-stream span scratch
// both hang off this one block.

namespace meta::comms::collstats {

struct CollStatsDeviceBlock {
  CollStatDoubleBank bank;
  CollStatSpanScratch* span; // [numSlots]
  uint32_t numSlots;
  CollStatHistGeometry hist; // device-resident, per-comm config
  uint32_t numThresholds;
  uint64_t thresholdsNs[kMaxThresholds]; // device-resident, per-comm config
};

// The device finalizer's only entry point: fold one completed observation into
// the current epoch's bank at `keyId`, using the block's device-resident
// threshold vector. `keyId` comes from the host key registry, which resolved it
// before the launch, so the device indexes rather than searching. Ids past the
// key capacity land on the reserved trailing catch-all slot. Shared between
// host and device so a GPU test drives the exact path.
COLLSTATS_HD inline void collStatsRecordById(
    CollStatsDeviceBlock* block,
    uint32_t keyId,
    uint64_t durNs,
    uint64_t logicalBytes) {
  const uint32_t valueSlot =
      keyId > block->bank.numKeys ? block->bank.numKeys : keyId;
  CollStatValue* values = collStatCurrentValues(&block->bank);
  collStatAccumulate(
      &values[valueSlot],
      durNs,
      logicalBytes,
      block->hist,
      block->thresholdsNs,
      block->numThresholds);
}

/* Per-communicator bucketing configuration, resolved from cvars by the owner
 * and copied into the block so the device finalizer reads it without a host
 * round trip. Defaults reproduce the compiled-in behaviour, so a caller that
 * does not configure anything gets 8 sub-buckets per octave over [1us, 1024s]
 * and cut-points at 1s/10s/60s/600s. */
struct CollStatsBlockConfig {
  CollStatHistGeometry hist;
  uint64_t thresholdsNs[kMaxThresholds];
  uint32_t numThresholds;
  /* Host-only, unlike the fields above: bucketing a message size happens on the
   * enqueue thread, so the device never reads these and they are deliberately
   * not copied into the device block. They live here because this struct is
   * what a window's exported labels are resolved against. */
  CollStatSizeClasses sizeClasses;
};

inline CollStatsBlockConfig collStatDefaultBlockConfig() {
  CollStatsBlockConfig cfg{};
  cfg.hist = collStatDefaultHistGeometry();
  cfg.numThresholds = kMaxThresholds;
  for (uint32_t i = 0; i < kMaxThresholds; ++i) {
    cfg.thresholdsNs[i] = kDefaultThresholdsNs[i];
  }
  cfg.sizeClasses = collStatDefaultSizeClasses();
  return cfg;
}

// Host-side handle retaining every device allocation backing a block, so it can
// be freed without reading pointers back from the device. `dev` is the pointer
// stored into the per-comm device state.
//
// Members are default-initialized so the "empty handle frees nothing" contract
// holds for any handle, not only a brace-initialized one:
// collStatsFreeDeviceBlock frees all four pointers unconditionally, and
// cudaFree is a no-op on null but not on an indeterminate address.
struct CollStatsDeviceBlockHandle {
  CollStatsDeviceBlock* dev = nullptr;
  CollStatValue* values[2] = {nullptr, nullptr};
  CollStatSpanScratch* span = nullptr;
  uint32_t numSlots = 0;
  uint32_t keyCapacity = 0; // key-index capacity; banks hold keyCapacity + 1
  /* The configuration the block was built with. Kept host-side so a readout can
   * export the geometry its buckets were actually produced under, rather than
   * whatever the defaults happen to be. */
  CollStatsBlockConfig cfg;
};

/* Allocate a device-resident block: two zeroed value banks of keyCapacity + 1
 * slots (the trailing slot is the catch-all) and numSlots span-scratch entries
 * initialized to the atomicMin start sentinel rather than to zero.
 *
 * Returns a null `dev` on allocation failure, on numSlots == 0, or on a `cfg`
 * whose geometry does not re-derive to itself -- the histogram and threshold
 * arrays are fixed capacity, so a geometry that indexes past them would write
 * past them rather than merely lose resolution. The check is a re-derivation
 * through collStatMakeHistGeometry, not a range test on the declared bucket
 * count, because the index logBucketNs produces follows from the bounds and
 * not from that count. */
CollStatsDeviceBlockHandle collStatsAllocDeviceBlock(
    uint32_t keyCapacity,
    uint32_t numSlots,
    const CollStatsBlockConfig& cfg = collStatDefaultBlockConfig());

/* Frees the four device allocations. Takes the handle by const reference and so
 * cannot null it: the handle is a value type that readers and drivers copy
 * freely, and nulling one copy would leave the others just as dangling. Exactly
 * one CollStatsDeviceBlockOwner is therefore responsible for calling this, and
 * every handle -- the owner's included -- is dead afterwards. Calling it twice
 * on the same block double-frees. A no-op on the empty (all-null) handle. */
void collStatsFreeDeviceBlock(const CollStatsDeviceBlockHandle& handle);

// RAII owner of a device block: frees the allocations on destruction, so the
// communicator holds one owner and lets destruction order handle teardown
// instead of a manual free. Move-only, single-owner — the handle it hands out
// is a non-owning view that readers/drivers copy freely.
// `collStatsFreeDeviceBlock` is a no-op on the empty (all-null) handle, so a
// default or moved-from owner destroys cleanly.
class CollStatsDeviceBlockOwner {
 public:
  CollStatsDeviceBlockOwner() = default;
  explicit CollStatsDeviceBlockOwner(CollStatsDeviceBlockHandle handle)
      : handle_(handle) {}
  ~CollStatsDeviceBlockOwner() {
    collStatsFreeDeviceBlock(handle_);
  }

  CollStatsDeviceBlockOwner(CollStatsDeviceBlockOwner&& other) noexcept
      : handle_(other.handle_) {
    other.handle_ = {};
  }
  CollStatsDeviceBlockOwner& operator=(
      CollStatsDeviceBlockOwner&& other) noexcept {
    if (this != &other) {
      collStatsFreeDeviceBlock(handle_);
      handle_ = other.handle_;
      other.handle_ = {};
    }
    return *this;
  }
  CollStatsDeviceBlockOwner(const CollStatsDeviceBlockOwner&) = delete;
  CollStatsDeviceBlockOwner& operator=(const CollStatsDeviceBlockOwner&) =
      delete;

  const CollStatsDeviceBlockHandle& handle() const {
    return handle_;
  }
  bool valid() const {
    return handle_.dev != nullptr;
  }

 private:
  CollStatsDeviceBlockHandle handle_{};
};

// Enqueue a stream-ordered pre-sequence reset of one span slot on `stream`:
// start = UINT64_MAX, arrived = 0. It is the slot cleanup — it runs,
// in stream order, after every prior kernel on the slot has completed and
// before the next collective's first entry, so the slot is clean regardless of
// whether the previous sequence finalized (an aborted sequence cannot poison
// its successor). Two byte-fills, so it is graph-capturable and needs no
// kernel. No-op when the handle is null (instrumentation off).
void collStatsEnqueuePreReset(
    const CollStatsDeviceBlockHandle& handle,
    uint32_t slot,
    cudaStream_t stream);

} // namespace meta::comms::collstats
