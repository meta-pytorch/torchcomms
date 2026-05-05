// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/colltrace/PrecisionClock.h"

#include <cstdlib>
#include <string_view>

#include <folly/logging/xlog.h>

// fbclock is a Meta-internal header. The MCCL/torchcomms OSS build copies
// only fbcode/comms (and ynl) into its build tree, so fbclock.h is absent
// there; gate every reference on __has_include so the OSS build compiles
// and links cleanly against the system_clock fallback path below.
#if __has_include("time/fbclock/fbclock.h")
#include "time/fbclock/fbclock.h"
#define COMMS_HAS_FBCLOCK 1
#else
#define COMMS_HAS_FBCLOCK 0
#endif

namespace meta::comms::colltrace {

namespace {

bool isPtpEnabled() {
  const char* env = std::getenv("NCCL_USE_PTP");
  if (env == nullptr) {
    return true;
  }
  std::string_view v{env};
  return !(
      v == "0" || v == "false" || v == "False" || v == "FALSE" || v == "no" ||
      v == "No" || v == "NO" || v == "n" || v == "N");
}

// Process-global handle to fbclock. Initialized exactly once on first
// access; intentionally leaked so the destructor never races with late
// callers during shutdown (mirrors common/time/PTP.cpp's PTPWrapper).
class PrecisionClockImpl {
 public:
  PrecisionClockImpl() : usingPtp_(false) {
    if (!isPtpEnabled()) {
      XLOG(INFO) << "PrecisionClock: NCCL_USE_PTP disabled, using system_clock";
      return;
    }
#if COMMS_HAS_FBCLOCK
    int err = fbclock_init(&lib_, FBCLOCK_PATH);
    if (err != FBCLOCK_E_NO_ERROR) {
      XLOG(WARN) << "PrecisionClock: fbclock_init failed (path=" << FBCLOCK_PATH
                 << ", err=" << err << "), falling back to system_clock";
      return;
    }
    usingPtp_ = true;
#else
    XLOG(INFO) << "PrecisionClock: built without fbclock, using system_clock";
#endif
  }

  bool usingPtp() const noexcept {
    return usingPtp_;
  }

#if COMMS_HAS_FBCLOCK
  // Reads fbclock without holding a mutex. fbclock_gettime mutates
  // lib_.min_phc_delay (a monotonically-decreasing int64) on every call;
  // this is a benign race accepted by common/time/PTP.cpp as well.
  // The shared-memory read uses fbclock's own seqlock/CRC protocol.
  bool getTruetime(fbclock_truetime* out) noexcept {
    if (!usingPtp_) {
      return false;
    }
    return fbclock_gettime(&lib_, out) == FBCLOCK_E_NO_ERROR;
  }
#endif

  static PrecisionClockImpl& instance() {
    static auto* p = new PrecisionClockImpl();
    return *p;
  }

 private:
#if COMMS_HAS_FBCLOCK
  fbclock_lib lib_{};
#endif
  bool usingPtp_;
};

std::pair<uint64_t, uint64_t> systemClockRangeNs() {
  auto u = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  return {u, u};
}

} // namespace

bool precisionUsingPtp() noexcept {
  return PrecisionClockImpl::instance().usingPtp();
}

std::pair<uint64_t, uint64_t> precisionNowRangeNs() noexcept {
#if COMMS_HAS_FBCLOCK
  fbclock_truetime tt{};
  if (PrecisionClockImpl::instance().getTruetime(&tt)) {
    return {tt.earliest_ns, tt.latest_ns};
  }
#endif
  return systemClockRangeNs();
}

uint64_t precisionNowNs() noexcept {
  auto [earliest, latest] = precisionNowRangeNs();
  return earliest + (latest - earliest) / 2;
}

uint64_t precisionErrorNs() noexcept {
  auto [earliest, latest] = precisionNowRangeNs();
  return latest - earliest;
}

std::chrono::system_clock::time_point precisionNow() noexcept {
  return std::chrono::system_clock::time_point{
      std::chrono::nanoseconds{precisionNowNs()}};
}

} // namespace meta::comms::colltrace
