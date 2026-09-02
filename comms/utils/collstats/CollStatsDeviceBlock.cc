// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsDeviceBlock.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/utils/collstats/CollStatsHistogram.h"

namespace meta::comms::collstats {

namespace {

/* Swallowing a CUDA failure hides it from us but not from the runtime: the
 * error stays latched per-thread until the next cudaGetLastError(), which is
 * usually a CUDACHECK in the collective library and would blame whatever ran
 * after us.
 *
 * Called only when our own call failed, but cudaGetLastError() clears whatever
 * is latched, not specifically ours -- an unrelated non-sticky error already
 * pending is consumed too. The runtime offers no way to clear selectively, and
 * between mis-attributing our failure to someone else's code and swallowing an
 * error nobody has observed yet, the first is the worse outcome. Takes the
 * status by value so a nodiscard call can be consumed by passing it in. */
void clearLatchedError(cudaError_t e) {
  if (e != cudaSuccess) {
    // Consume the return value to satisfy HIP's nodiscard on hipGetLastError.
    [[maybe_unused]] const cudaError_t cleared = cudaGetLastError();
  }
}

// Allocate `bytes` of device memory into *out. Returns false on failure.
bool alloc(void** out, std::size_t bytes) {
  const cudaError_t e = cudaMalloc(out, bytes);
  if (e != cudaSuccess) {
    clearLatchedError(e);
    *out = nullptr;
    return false;
  }
  return true;
}

// Allocate `bytes` of zeroed device memory into *out. Returns false on failure.
bool allocZeroed(void** out, std::size_t bytes) {
  if (!alloc(out, bytes)) {
    return false;
  }
  const cudaError_t e = cudaMemset(*out, 0, bytes);
  if (e != cudaSuccess) {
    clearLatchedError(e);
    clearLatchedError(cudaFree(*out));
    *out = nullptr;
    return false;
  }
  return true;
}

} // namespace

CollStatsDeviceBlockHandle collStatsAllocDeviceBlock(
    uint32_t keyCapacity,
    uint32_t numSlots,
    const CollStatsBlockConfig& cfg) {
  // Validate before building the handle, so every rejection returns an empty
  // one and no partially-populated handle can escape.
  if (keyCapacity == UINT32_MAX) {
    return CollStatsDeviceBlockHandle{};
  }
  // The block's histogram and threshold arrays are fixed capacity, so an
  // over-large configuration would write past them. Refuse rather than clamp:
  // silently reshaping the geometry would make the exported buckets disagree
  // with the geometry exported alongside them.
  //
  // Re-derive the bucket count instead of range-checking the one handed in.
  // logBucketNs bounds its index by tMinNs/tMaxNs/subBucketsPerOctave, not by
  // the declared numBuckets, so a geometry whose count is too small for its own
  // bounds passes every range check and still indexes past histogram[].
  // collStatMakeHistGeometry is the single source of truth the defaults are
  // built through, and it also rejects tMinNs == 0 -- which would make
  // log2(dur / tMinNs) infinite and the cast to a bucket index undefined.
  const CollStatHistGeometry derived = collStatMakeHistGeometry(
      cfg.hist.tMinNs, cfg.hist.tMaxNs, cfg.hist.subBucketsPerOctave);
  if (derived.numBuckets == 0 || derived.numBuckets != cfg.hist.numBuckets ||
      cfg.numThresholds > kMaxThresholds) {
    return CollStatsDeviceBlockHandle{};
  }
  // A zero-slot block allocates nothing for the span scratch but still reports
  // a valid handle, leaving every span entry pointing at unallocated memory.
  if (numSlots == 0) {
    return CollStatsDeviceBlockHandle{};
  }

  CollStatsDeviceBlockHandle handle{};
  handle.numSlots = numSlots;
  handle.keyCapacity = keyCapacity;
  handle.cfg = cfg;
  const uint32_t valueSlots = keyCapacity + 1; // trailing catch-all slot

  // The span is allocated but not zeroed: the spanInit copy below overwrites
  // every byte of it, so zeroing here would be discarded work and one more way
  // to fail. The value banks do need it -- nothing else initializes them.
  const std::size_t bankBytes =
      static_cast<std::size_t>(valueSlots) * sizeof(CollStatValue);
  const std::size_t spanBytes =
      static_cast<std::size_t>(numSlots) * sizeof(CollStatSpanScratch);
  const bool ok =
      allocZeroed(reinterpret_cast<void**>(&handle.values[0]), bankBytes) &&
      allocZeroed(reinterpret_cast<void**>(&handle.values[1]), bankBytes) &&
      alloc(reinterpret_cast<void**>(&handle.span), spanBytes);
  if (!ok) {
    collStatsFreeDeviceBlock(handle);
    return CollStatsDeviceBlockHandle{};
  }

  // Value-initialized; only start departs from zero, taking the atomicMin
  // sentinel so the first entry timestamp wins the min rather than 0.
  std::vector<CollStatSpanScratch> spanInit(numSlots);
  for (auto& s : spanInit) {
    s.start = kSpanStartInit;
  }
  const cudaError_t spanCopy = cudaMemcpy(
      handle.span, spanInit.data(), spanBytes, cudaMemcpyHostToDevice);
  if (spanCopy != cudaSuccess) {
    clearLatchedError(spanCopy);
    collStatsFreeDeviceBlock(handle);
    return CollStatsDeviceBlockHandle{};
  }

  // Build the block on the host with device pointers, then publish it to
  // device.
  CollStatsDeviceBlock hostBlock{};
  hostBlock.bank.numKeys = keyCapacity;
  hostBlock.bank.epoch = 0;
  hostBlock.bank.values[0] = handle.values[0];
  hostBlock.bank.values[1] = handle.values[1];
  hostBlock.span = handle.span;
  hostBlock.numSlots = numSlots;
  hostBlock.hist = cfg.hist;
  hostBlock.numThresholds = cfg.numThresholds;
  for (uint32_t i = 0; i < cfg.numThresholds; ++i) {
    hostBlock.thresholdsNs[i] = cfg.thresholdsNs[i];
  }

  if (!alloc(
          reinterpret_cast<void**>(&handle.dev),
          sizeof(CollStatsDeviceBlock))) {
    collStatsFreeDeviceBlock(handle);
    return CollStatsDeviceBlockHandle{};
  }
  const cudaError_t blockCopy = cudaMemcpy(
      handle.dev,
      &hostBlock,
      sizeof(CollStatsDeviceBlock),
      cudaMemcpyHostToDevice);
  if (blockCopy != cudaSuccess) {
    clearLatchedError(blockCopy);
    collStatsFreeDeviceBlock(handle);
    return CollStatsDeviceBlockHandle{};
  }
  return handle;
}

void collStatsFreeDeviceBlock(const CollStatsDeviceBlockHandle& handle) {
  // Fail-open: a failed free is ignored, but not left latched.
  clearLatchedError(cudaFree(handle.dev));
  clearLatchedError(cudaFree(handle.values[0]));
  clearLatchedError(cudaFree(handle.values[1]));
  clearLatchedError(cudaFree(handle.span));
}

void collStatsEnqueuePreReset(
    const CollStatsDeviceBlockHandle& handle,
    uint32_t slot,
    cudaStream_t stream) {
  if (handle.dev == nullptr || slot >= handle.numSlots) {
    return;
  }
  // Compute device addresses without dereferencing a device pointer on the
  // host.
  // `&handle.span[slot].start` looks like a host dereference of device memory
  // to sanitizers even though it is just offset arithmetic; use explicit byte
  // offsets from the base pointer.
  const char* slotBase = reinterpret_cast<const char*>(handle.span) +
      static_cast<std::size_t>(slot) * sizeof(CollStatSpanScratch);
  const char* startAddr = slotBase + offsetof(CollStatSpanScratch, start);
  const char* arrivedAddr = slotBase + offsetof(CollStatSpanScratch, arrived);
  // start = UINT64_MAX is all 0xFF bytes; arrived = 0 is all 0x00 bytes. Two
  // async fills, stream-ordered before the collective's first entry.
  clearLatchedError(cudaMemsetAsync(
      const_cast<char*>(startAddr), 0xFF, sizeof(uint64_t), stream));
  clearLatchedError(cudaMemsetAsync(
      const_cast<char*>(arrivedAddr), 0, sizeof(uint32_t), stream));
}

} // namespace meta::comms::collstats
