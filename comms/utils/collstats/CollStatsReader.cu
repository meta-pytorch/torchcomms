// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsReader.h"

#include <cstddef>

namespace meta::comms::collstats {

namespace {

// Advance the epoch by one, publishing the flip to finalizers. A finalizer that
// observes the new epoch also observes an initialized new bank because the
// previous window's zeroing memset is stream-ordered ahead of this launch, not
// because of the fence: the fence follows the increment, so it supplies
// system-scope visibility of the epoch word, not release ordering.
// One thread; the reader launches it on the reader stream.
__global__ void flipEpochKernel(CollStatsDeviceBlock* block) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&block->bank.epoch), 1ull);
  __threadfence_system();
}

// Step 1: make the reader stream wait until every old-epoch finalizer on each
// instrumented stream has retired, so the retired bank is quiescent before the
// copy. No-op when ungated.
cudaError_t gateOldFinalizers(
    cudaStream_t readerStream,
    const CollStatsReadGating* gating) {
  if (gating == nullptr) {
    return cudaSuccess;
  }
  for (uint32_t i = 0; i < gating->numStreams; ++i) {
    cudaError_t err = cudaEventRecord(
        gating->streamEvents[i], gating->instrumentedStreams[i]);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaStreamWaitEvent(readerStream, gating->streamEvents[i], 0);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

// Step 3: make each instrumented stream wait for the flip to be visible, so no
// new-window collective launches before the epoch write and thus none writes
// the retired bank. No-op when ungated.
cudaError_t gateNewLaunches(
    cudaStream_t readerStream,
    const CollStatsReadGating* gating) {
  if (gating == nullptr) {
    return cudaSuccess;
  }
  cudaError_t err = cudaEventRecord(gating->flipEvent, readerStream);
  if (err != cudaSuccess) {
    return err;
  }
  for (uint32_t i = 0; i < gating->numStreams; ++i) {
    err = cudaStreamWaitEvent(
        gating->instrumentedStreams[i], gating->flipEvent, 0);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

// Shared async sequence: wait old finalizers -> flip -> gate new launches ->
// copy retired bank + key index + cumulative counters into `out` -> zero the
// retired bank -> record copyDoneEvent (if non-null). No sync. `out` must be
// pre-sized to numKeys. The retired bank is quiescent between the flip and the
// copy, and new finalizers write the new bank, so the copy never races live
// atomics.
// `flipped`, when non-null, is set once the flip has been enqueued: past that
// point the epoch has advanced whatever happens next, and a caller that bails
// owns a retired bank nobody will copy or zero.
cudaError_t issueGatedReadout(
    const CollStatsDeviceBlockHandle& handle,
    cudaStream_t readerStream,
    const CollStatsReadGating* gating,
    uint64_t currentEpoch,
    const CollStatsKeyRegistry& keys,
    cudaEvent_t copyDoneEvent,
    CollStatsPinnedStaging& staging,
    bool* flipped) {
  const uint32_t retired = static_cast<uint32_t>(currentEpoch & 1u);
  const uint32_t capacity = handle.keyCapacity;

  cudaError_t e = gateOldFinalizers(readerStream, gating);
  if (e != cudaSuccess) {
    return e;
  }
  flipEpochKernel<<<1, 1, 0, readerStream>>>(handle.dev);
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    return e;
  }
  if (flipped != nullptr) {
    *flipped = true;
  }
  e = gateNewLaunches(readerStream, gating);
  if (e != cudaSuccess) {
    return e;
  }
  // Sampled here, after the gate, and not by the caller before the flip. A
  // collective enqueued between the old-finalizer event records and this gate
  // is waited on by neither: the gate's wait lands at the stream's tail, behind
  // that collective, so it can still execute against the retired bank. Its key
  // was resolved before it was enqueued, so a count taken earlier would exclude
  // it and its observation would be neither copied nor counted -- just zeroed
  // by the memset below. Everything enqueued after the gate writes the new bank
  // and is not our business.
  //
  // Clamped because the registry's capacity is its own; ids past the bank's
  // capacity are folded onto the catch-all by collStatsRecordById and never
  // occupy a slot of their own.
  const uint32_t liveKeys = keys.size();
  const uint32_t numKeys = liveKeys < capacity ? liveKeys : capacity;
  staging.setStaged(numKeys);

  // Occupied prefix only: ids are dense, so everything past numKeys has never
  // been written.
  if (numKeys > 0) {
    e = cudaMemcpyAsync(
        staging.values(),
        handle.values[retired],
        static_cast<std::size_t>(numKeys) * sizeof(CollStatValue),
        cudaMemcpyDeviceToHost,
        readerStream);
    if (e != cudaSuccess) {
      return e;
    }
  }
  // The catch-all sits at the bank's capacity, not at numKeys, so it needs its
  // own copy into the staging buffer's trailing slot.
  e = cudaMemcpyAsync(
      staging.values() + numKeys,
      handle.values[retired] + capacity,
      sizeof(CollStatValue),
      cudaMemcpyDeviceToHost,
      readerStream);
  if (e != cudaSuccess) {
    return e;
  }
  // Zero the whole bank rather than the copied prefix: a device-side memset
  // costs no transfer, and it keeps the bank clean if the id space grows.
  e = cudaMemsetAsync(
      handle.values[retired],
      0,
      static_cast<std::size_t>(capacity + 1) * sizeof(CollStatValue),
      readerStream);
  if (e != cudaSuccess) {
    return e;
  }
  if (copyDoneEvent != nullptr) {
    e = cudaEventRecord(copyDoneEvent, readerStream);
    if (e != cudaSuccess) {
      return e;
    }
  }
  return cudaSuccess;
}

} // namespace

CollStatsPinnedStaging::~CollStatsPinnedStaging() {
  release();
}

void CollStatsPinnedStaging::release() {
  if (values_ != nullptr) {
    // Consume the return value to satisfy HIP's nodiscard on hipHostFree.
    [[maybe_unused]] const cudaError_t e = cudaFreeHost(values_);
    values_ = nullptr;
  }
  capacity_ = 0;
  staged_ = 0;
}

bool CollStatsPinnedStaging::allocate(uint32_t capacity) {
  release();
  if (cudaMallocHost(
          reinterpret_cast<void**>(&values_),
          static_cast<std::size_t>(capacity + 1) * sizeof(CollStatValue)) !=
      cudaSuccess) {
    values_ = nullptr;
    return false;
  }
  capacity_ = capacity;
  return true;
}

void CollStatsPinnedStaging::publish(
    uint64_t windowEpoch,
    const CollStatsKeyRegistry& keys,
    const CollStatsBlockConfig& cfg,
    CollStatSnapshot& out) const {
  out.windowEpoch = windowEpoch;
  out.hist = cfg.hist;
  out.sizeClasses = cfg.sizeClasses;
  // Clamped rather than trusted: the destination array is fixed capacity, and a
  // config that reached here with a larger count would write past it. Callers
  // reuse one snapshot across windows, so a partial overwrite would also leave
  // the tail describing the previous window's bucketing.
  out.numThresholds =
      cfg.numThresholds < kMaxThresholds ? cfg.numThresholds : kMaxThresholds;
  for (uint32_t i = 0; i < out.numThresholds; ++i) {
    out.thresholdsNs[i] = cfg.thresholdsNs[i];
  }
  // Reset rather than leave: this snapshot is reused across publishes, and the
  // window bounds are producer-stamped after publish returns. Carrying the
  // previous window's timestamps into one that was never stamped would date the
  // window to the wrong interval.
  out.windowStartUnixNs = 0;
  out.windowEndUnixNs = 0;
  if (!valid()) {
    out.numKeys = 0;
    out.catchAllCount = 0;
    out.keys.clear();
    out.values.clear();
    return;
  }
  out.numKeys = staged_;
  out.catchAllCount = keys.catchAllCount();
  // The registry may have grown since the copy was issued; only the slots that
  // were actually transferred are described.
  out.keys = keys.keys();
  out.keys.resize(staged_);
  out.values.assign(values_, values_ + staged_ + 1);
}

cudaError_t collStatsIssueReadWindow(
    const CollStatsDeviceBlockHandle& handle,
    cudaStream_t readerStream,
    const CollStatsReadGating* gating,
    uint64_t currentEpoch,
    cudaEvent_t copyDoneEvent,
    CollStatsPinnedStaging& staging,
    const CollStatsKeyRegistry& keys) {
  if (handle.dev == nullptr || !staging.valid() ||
      staging.capacity() != handle.keyCapacity) {
    return cudaErrorInvalidValue;
  }
  return issueGatedReadout(
      handle,
      readerStream,
      gating,
      currentEpoch,
      keys,
      copyDoneEvent,
      staging,
      /*flipped=*/nullptr);
}

CollStatSnapshot collStatsReadWindow(
    const CollStatsDeviceBlockHandle& handle,
    cudaStream_t readerStream,
    const CollStatsKeyRegistry& keys,
    const CollStatsReadGating* gating) {
  CollStatSnapshot snapshot{};
  if (handle.dev == nullptr) {
    return snapshot;
  }

  // Read just the epoch word to pick the retired bank. Only the reader mutates
  // the epoch, so this host read is race-free; the capacity comes from the
  // handle, so no full-block read is needed.
  //
  // Async on the reader stream rather than a plain cudaMemcpy. A synchronous
  // copy runs on the legacy default stream, which implicitly synchronizes with
  // every blocking stream on the device -- so reading eight bytes of telemetry
  // would wait on the training streams, which is exactly what this whole path
  // promises not to do. The byte offset avoids forming &handle.dev->bank.epoch
  // on the host, the same reason collStatsEnqueuePreReset uses offsets.
  uint64_t currentEpoch = 0;
  const char* epochAddr = reinterpret_cast<const char*>(handle.dev) +
      offsetof(CollStatsDeviceBlock, bank) +
      offsetof(CollStatDoubleBank, epoch);
  if (cudaMemcpyAsync(
          &currentEpoch,
          epochAddr,
          sizeof(currentEpoch),
          cudaMemcpyDeviceToHost,
          readerStream) != cudaSuccess ||
      cudaStreamSynchronize(readerStream) != cudaSuccess) {
    return snapshot;
  }

  // This path synchronizes anyway, so pinning buys nothing here; the staging
  // buffer exists because issueGatedReadout writes through it.
  CollStatsPinnedStaging staging;
  if (!staging.allocate(handle.keyCapacity)) {
    return snapshot;
  }

  bool flipped = false;
  if (issueGatedReadout(
          handle,
          readerStream,
          gating,
          currentEpoch,
          keys,
          /*copyDoneEvent=*/nullptr,
          staging,
          &flipped) != cudaSuccess ||
      cudaStreamSynchronize(readerStream) != cudaSuccess) {
    if (flipped) {
      // The epoch advanced but the retired bank was never copied or zeroed.
      // Left resident, its counts become the next window's starting balance and
      // inflate it. Best-effort: a stream wedged badly enough to fail above
      // will fail here too, and then the counts are unrecoverable either way.
      // Return values are consumed to satisfy HIP's nodiscard on
      // hipMemsetAsync/hipStreamSynchronize; the recovery is best-effort, so
      // there is nothing to do with a failure here.
      [[maybe_unused]] const cudaError_t e0 = cudaMemsetAsync(
          handle.values[currentEpoch & 1u],
          0,
          static_cast<std::size_t>(handle.keyCapacity + 1) *
              sizeof(CollStatValue),
          readerStream);
      [[maybe_unused]] const cudaError_t e1 =
          cudaStreamSynchronize(readerStream);
    }
    // `staging` is freed on the way out of this scope, and the failure may have
    // been the synchronize above with both D2H copies already enqueued -- so
    // drain before the pinned buffer they land in goes away. Best-effort, and
    // deliberately not relying on cudaFreeHost's implicit synchronize, which is
    // an implementation detail rather than a documented barrier.
    [[maybe_unused]] const cudaError_t drained =
        cudaStreamSynchronize(readerStream);
    return CollStatSnapshot{};
  }
  staging.publish(currentEpoch, keys, handle.cfg, snapshot);
  return snapshot;
}

} // namespace meta::comms::collstats
