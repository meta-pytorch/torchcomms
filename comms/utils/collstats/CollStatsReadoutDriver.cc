// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsReadoutDriver.h"
#include <chrono>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <utility>

namespace meta::comms::collstats {

namespace {

uint64_t wallNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// Create into `*out`, leaving it null on failure. CUDA does not promise to
// leave an output parameter untouched when a create fails, and the destructor
// destroys any non-null handle -- so a failed create must not be able to leave
// an indeterminate one behind.
bool createReaderStream(cudaStream_t* out) {
  if (cudaStreamCreateWithFlags(out, cudaStreamNonBlocking) != cudaSuccess) {
    *out = nullptr;
    return false;
  }
  return true;
}

bool createTimingDisabledEvent(cudaEvent_t* out) {
  if (cudaEventCreateWithFlags(out, cudaEventDisableTiming) != cudaSuccess) {
    *out = nullptr;
    return false;
  }
  return true;
}

// Puts the caller's CUDA device back on scope exit, or does nothing when
// constructed with -1. The issue path launches the flip kernel on the reader
// stream, which belongs to the driver's device, so that device has to stay
// current across the whole of issue() and flush() -- not just the synchronize.
class DeviceRestorer {
 public:
  explicit DeviceRestorer(int device) : device_(device) {}
  ~DeviceRestorer() {
    if (device_ >= 0) {
      // Best-effort; the return value is consumed for HIP's nodiscard.
      [[maybe_unused]] const cudaError_t e = cudaSetDevice(device_);
    }
  }
  DeviceRestorer(const DeviceRestorer&) = delete;
  DeviceRestorer& operator=(const DeviceRestorer&) = delete;
  DeviceRestorer(DeviceRestorer&&) = delete;
  DeviceRestorer& operator=(DeviceRestorer&&) = delete;

 private:
  int device_;
};

} // namespace

CollStatsReadoutDriver::CollStatsReadoutDriver(
    const CollStatsDeviceBlockHandle& handle,
    uint32_t cadence,
    Sink sink,
    const CollStatsKeyRegistry& keys)
    : handle_(handle),
      cadence_(std::max<uint32_t>(1u, cadence)),
      sink_(std::move(sink)),
      keys_(&keys) {
  // The registry's id space and the bank's value slots are sized
  // independently, and only agree when the registry's catch-all sits exactly at
  // the bank's key capacity. Mismatched, every readout past that point fails
  // inside collStatsIssueReadWindow and is charged as a dropped window, which
  // reads identically to a CUDA fault -- so refuse at construction instead of
  // going dark mid-run.
  if (handle_.dev == nullptr || keys.catchAllId() != handle_.keyCapacity ||
      cudaGetDevice(&device_) != cudaSuccess || !createReaderStream(&reader_) ||
      !createTimingDisabledEvent(&streamEvent_) ||
      !createTimingDisabledEvent(&flipEvent_) ||
      !createTimingDisabledEvent(&copyDone_) ||
      !pinned_.allocate(handle_.keyCapacity)) {
    disabled_ = true;
  }
  // The first window accumulates from here, not from epoch zero.
  windowOpenNs_ = wallNowNs();
}

CollStatsReadoutDriver::~CollStatsReadoutDriver() {
  // Harvest the last issued window and the collectives accumulated since,
  // before the reader stream and the device bank go away.
  flushFinal();
  // A window can still be in flight here, and pending_ does not tell us:
  // collStatsIssueReadWindow enqueues both D2H copies into pinned_ before its
  // last few steps, so a failure at the memset or the event record returns
  // non-success with those copies already on the stream, and issue() then
  // leaves pending_ false. cudaStreamDestroy does not wait, and pinned_ is
  // freed as soon as this body returns, so a gated sync would let the device
  // DMA into a freed page-locked buffer. Unconditional, because the paths that
  // clear pending_ on failure are exactly the ones that need it.
  if (reader_ != nullptr) {
    [[maybe_unused]] const cudaError_t e = cudaStreamSynchronize(reader_);
  }
  // Return values are consumed to satisfy HIP's nodiscard on the destroy
  // entry points. Teardown is fail-open, so failures are ignored.
  if (copyDone_ != nullptr) {
    [[maybe_unused]] const cudaError_t e = cudaEventDestroy(copyDone_);
  }
  if (flipEvent_ != nullptr) {
    [[maybe_unused]] const cudaError_t e = cudaEventDestroy(flipEvent_);
  }
  if (streamEvent_ != nullptr) {
    [[maybe_unused]] const cudaError_t e = cudaEventDestroy(streamEvent_);
  }
  if (reader_ != nullptr) {
    [[maybe_unused]] const cudaError_t e = cudaStreamDestroy(reader_);
  }
}

void CollStatsReadoutDriver::harvestIfReady() {
  if (!pending_) {
    return;
  }
  const cudaError_t q = cudaEventQuery(copyDone_);
  if (q == cudaSuccess) {
    pinned_.publish(pendingEpoch_, *keys_, handle_.cfg, snapshot_);
    // publish() fills the bank-derived fields; the wall bounds are the
    // producer's and are stamped here.
    snapshot_.windowStartUnixNs = pendingOpenNs_;
    snapshot_.windowEndUnixNs = pendingCloseNs_;
    if (sink_) {
      // The sink is caller-supplied and this runs from the destructor's final
      // flush, where an escaping exception would hit a noexcept boundary and
      // std::terminate -- killing the job over telemetry. Swallow it, count it,
      // and say so once so a silently throwing sink is still diagnosable.
      try {
        sink_(snapshot_);
      } catch (const std::exception& e) {
        if (sinkExceptions_ == 0) {
          fprintf(
              stderr,
              "collstats: readout sink threw (%s); window dropped, "
              "further occurrences counted only\n",
              e.what());
        }
        ++sinkExceptions_;
      } catch (...) {
        if (sinkExceptions_ == 0) {
          fprintf(
              stderr,
              "collstats: readout sink threw a non-std exception; window "
              "dropped, further occurrences counted only\n");
        }
        ++sinkExceptions_;
      }
    }
    ++windowsExported_;
    pending_ = false;
  } else if (q == cudaErrorNotReady) {
    // The previous copy is still in flight; leave it pending and retry the
    // harvest next cycle rather than overwrite an in-flight staging buffer.
    // Deferred, not lost: the window still lands in exported or dropped later,
    // so charging it here would double-count it and stop the two counters from
    // partitioning windows.
    ++harvestRetries_;
  } else {
    disabled_ = true;
    ++windowsDropped_;
  }
}

void CollStatsReadoutDriver::onCollective(cudaStream_t instrumentedStream) {
  if (disabled_ || handle_.dev == nullptr) {
    return;
  }
  // Not reset here: an issue that does not happen must not clear the count, or
  // the collectives it covers become invisible to flushFinal. Left above
  // cadence, the next tick retries instead of waiting a whole cadence again.
  if (++sinceReadout_ < cadence_) {
    return;
  }

  harvestIfReady();
  if (disabled_ || pending_) {
    // Either a real error, or the previous window's copy is not done yet; do
    // not issue a new one until the staging buffer is free.
    return;
  }

  const cudaEvent_t streamEvents[1] = {streamEvent_};
  CollStatsReadGating gating{};
  gating.instrumentedStreams = &instrumentedStream;
  gating.streamEvents = streamEvents;
  gating.numStreams = 1;
  gating.flipEvent = flipEvent_;
  issue(&gating);
}

void CollStatsReadoutDriver::issue(const CollStatsReadGating* gating) {
  const cudaError_t e = collStatsIssueReadWindow(
      handle_, reader_, gating, localEpoch_, copyDone_, pinned_, *keys_);
  if (e == cudaSuccess) {
    const uint64_t now = wallNowNs();
    pendingEpoch_ = localEpoch_;
    pendingOpenNs_ = windowOpenNs_;
    pendingCloseNs_ = now;
    windowOpenNs_ = now;
    ++localEpoch_;
    pending_ = true;
    // The flip happened, so the bank this counts against is now the new one.
    sinceReadout_ = 0;
  } else {
    disabled_ = true;
    ++windowsDropped_;
  }
}

void CollStatsReadoutDriver::flush() {
  if (disabled_ || !pending_) {
    return;
  }
  if (cudaStreamSynchronize(reader_) != cudaSuccess) {
    disabled_ = true;
    ++windowsDropped_;
    return;
  }
  harvestIfReady();
}

void CollStatsReadoutDriver::flushFinal() {
  flush();

  // Nothing has accumulated since the last boundary, or the staging buffer is
  // still occupied by a window flush() could not land: either way there is no
  // extra window to issue.
  if (disabled_ || pending_ || sinceReadout_ == 0 || handle_.dev == nullptr) {
    return;
  }
  // Stands in for the gating an on-boundary issue would do, without needing a
  // handle to the instrumented streams. The sync only covers the calling
  // thread's current device, and teardown can run on a thread that never
  // selected ours, so select the driver's device and put the caller's back —
  // otherwise the wait is vacuous and the ungated issue races live finalizers.
  int callerDev = -1;
  if (cudaGetDevice(&callerDev) != cudaSuccess) {
    disabled_ = true;
    ++windowsDropped_;
    return;
  }
  const bool swapDev = callerDev != device_;
  if (swapDev && cudaSetDevice(device_) != cudaSuccess) {
    disabled_ = true;
    ++windowsDropped_;
    return;
  }
  // Held until after issue() and flush(), not just the synchronize below.
  // issue() launches the flip kernel on reader_, which belongs to device_, and
  // a launch onto a stream of a non-current device fails with
  // cudaErrorInvalidResourceHandle -- which would disable the driver and drop
  // the very window this function exists to export.
  const DeviceRestorer restore(swapDev ? callerDev : -1);

  if (cudaDeviceSynchronize() != cudaSuccess) {
    disabled_ = true;
    ++windowsDropped_;
    return;
  }
  issue(/*gating=*/nullptr);
  flush();
}

} // namespace meta::comms::collstats
