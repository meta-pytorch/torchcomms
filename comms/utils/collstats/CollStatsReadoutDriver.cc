// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsReadoutDriver.h"
#include <chrono>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <thread>
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
  // Then wait for the reader stream regardless of what flushFinal decided. A
  // D2H into pinned_ can still be in flight with pending_ false: issue() can
  // fail *after* both copies are enqueued (at the memset or the event record),
  // and that failure disables the driver, after which flushFinal and flush both
  // early-return. Freeing the DMA destination under a live copy is a
  // use-after-free that cudaFreeHost's implicit synchronize only happens to
  // mask.
  //
  // Not on the error path: once flushOnError has fired the device is wedged by
  // definition and this wait is unbounded. Teardown of an already-failing job
  // must not hang, so there the copy is left to process exit.
  if (reader_ != nullptr && !errored_) {
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

void CollStatsReadoutDriver::harvestIfReadyLocked() {
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
    // so charging it here would double-count it and, because sinceReadout_ is
    // no longer cleared on a skipped issue, fire once per collective for the
    // whole length of a stall.
    ++harvestRetries_;
  } else {
    disabled_ = true;
    ++windowsDropped_;
  }
}

void CollStatsReadoutDriver::onCollective(cudaStream_t instrumentedStream) {
  // Cheap pre-check off the lock: the overwhelming majority of calls are
  // non-readout ticks and must not serialize against anything.
  if (disabled_ || handle_.dev == nullptr) {
    return;
  }
  // Not reset here: an issue that does not happen must not clear the count,
  // or the collectives it covers become invisible to flushFinal. Left above
  // cadence, the next tick retries instead of waiting a whole cadence again.
  if (sinceReadout_.fetch_add(1, std::memory_order_relaxed) + 1 < cadence_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (disabled_) {
    // flushOnError may have disabled the driver since the pre-check.
    return;
  }
  harvestIfReadyLocked();
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
  issueLocked(&gating);
}

void CollStatsReadoutDriver::issueLocked(const CollStatsReadGating* gating) {
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
    sinceReadout_.store(0, std::memory_order_relaxed);
  } else {
    disabled_ = true;
    ++windowsDropped_;
  }
}

void CollStatsReadoutDriver::flush() {
  std::lock_guard<std::mutex> lock(mu_);
  flushLocked();
}

void CollStatsReadoutDriver::flushLocked() {
  if (disabled_ || !pending_) {
    return;
  }
  if (cudaStreamSynchronize(reader_) != cudaSuccess) {
    disabled_ = true;
    ++windowsDropped_;
    return;
  }
  harvestIfReadyLocked();
}

void CollStatsReadoutDriver::flushFinal() {
  std::lock_guard<std::mutex> lock(mu_);

  if (errored_) {
    // Both waits below are unbounded — flushLocked synchronizes the reader
    // stream and the device sync waits on everything — and on this path what
    // they would wait for is whatever wedged the GPU. flushOnError already made
    // a time-boxed attempt at the pending window; take one more non-blocking
    // look and give the rest up rather than hang teardown of a failing job.
    harvestIfReadyLocked();
    return;
  }
  flushLocked();

  // Nothing has accumulated since the last boundary, or the staging buffer is
  // still occupied by a window flushLocked could not land: either way there is
  // no extra window to issue.
  if (disabled_ || pending_ ||
      sinceReadout_.load(std::memory_order_relaxed) == 0 ||
      handle_.dev == nullptr) {
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
  issueLocked(/*gating=*/nullptr);
  flushLocked();
}

void CollStatsReadoutDriver::flushOnError() {
  std::lock_guard<std::mutex> lock(mu_);
  // Set before the early return: the job is failing either way, and teardown
  // uses this to decide it cannot afford to wait on the device.
  errored_ = true;
  if (disabled_ || !pending_) {
    // Nothing in flight: the window is either already exported or was never
    // issued. There is no partial state worth salvaging.
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + kErrorFlushTimeout;
  do {
    const cudaError_t q = cudaEventQuery(copyDone_);
    if (q == cudaSuccess) {
      harvestIfReadyLocked();
      return;
    }
    if (q != cudaErrorNotReady) {
      disabled_ = true;
      ++windowsDropped_;
      return;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);

  // Timed out. Leave the window pending rather than disabling: teardown's
  // flush() gets one more, longer, chance at it — a deferral, not a loss.
  ++harvestRetries_;
}

} // namespace meta::comms::collstats
