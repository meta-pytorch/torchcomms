// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GpuClockCalibration.h"

#include <chrono>
#include <cstdint>

#include <gtest/gtest.h>

using meta::comms::colltrace::GlobaltimerCalibration;
using namespace std::chrono_literals;

namespace {

// Build a calibration as if it were captured "right now" so that the
// reconstruction inside toWallClock(uint32_t) — which queries
// std::chrono::system_clock::now() internally — has a known reference.
GlobaltimerCalibration makeCalibrationAtNow(uint64_t deviceNs) {
  GlobaltimerCalibration cal{};
  cal.device_ns = deviceNs;
  cal.host_time = std::chrono::system_clock::now();
  return cal;
}

uint32_t pack(uint64_t gpuNs) {
  return static_cast<uint32_t>(gpuNs >> US_TICK_TIMESTAMP_SHIFT);
}

// Loose equality: timing-dependent because toWallClock(uint32_t) reads
// system_clock::now() to anchor the reconstruction. Anything within
// 100ms of expected is fine — the test exercises the math, not host
// scheduler latency.
constexpr auto kSlack = 100ms;

void expectClose(
    std::chrono::system_clock::time_point actual,
    std::chrono::system_clock::time_point expected) {
  auto diff = actual - expected;
  if (diff < std::chrono::system_clock::duration::zero()) {
    diff = -diff;
  }
  EXPECT_LE(diff, kSlack)
      << "diff = "
      << std::chrono::duration_cast<std::chrono::microseconds>(diff).count()
      << "us";
}

} // namespace

TEST(GpuClockCalibrationTest, ToWallClock64BitIdentity) {
  // The 64-bit overload is pure arithmetic on the calibration point:
  // toWallClock(device_ns) must equal host_time exactly.
  const uint64_t kDeviceNs = 5'000'000'000ULL;
  GlobaltimerCalibration cal{};
  cal.device_ns = kDeviceNs;
  cal.host_time = std::chrono::system_clock::now();

  EXPECT_EQ(cal.toWallClock(kDeviceNs), cal.host_time);
}

TEST(GpuClockCalibrationTest, ToWallClock64BitOffset) {
  // device_ns + 2s should map to host_time + 2s.
  const uint64_t kDeviceNs = 5'000'000'000ULL;
  GlobaltimerCalibration cal{};
  cal.device_ns = kDeviceNs;
  cal.host_time = std::chrono::system_clock::now();

  auto wall =
      cal.toWallClock(static_cast<uint64_t>(kDeviceNs + 2'000'000'000ULL));
  EXPECT_EQ(wall, cal.host_time + 2s);
}

TEST(GpuClockCalibrationTest, ToWallClock32BitAtCalibrationPoint) {
  // Event at exactly the calibration point — packed timestamp is
  // (device_ns >> 10). Reconstruction should yield ≈ host_time.
  const uint64_t kDeviceNs = 5'000'000'000ULL;
  auto cal = makeCalibrationAtNow(kDeviceNs);

  expectClose(cal.toWallClock(pack(kDeviceNs)), cal.host_time);
}

TEST(GpuClockCalibrationTest, ToWallClock32BitPastEvent) {
  // Event 1 second before the calibration point. Reconstruction should
  // yield ≈ host_time - 1s.
  const uint64_t kDeviceNs = 5'000'000'000ULL;
  auto cal = makeCalibrationAtNow(kDeviceNs);

  auto wall = cal.toWallClock(pack(kDeviceNs - 1'000'000'000ULL));
  expectClose(wall, cal.host_time - 1s);
}

TEST(GpuClockCalibrationTest, ToWallClock32BitFutureEvent) {
  // Event 1 second after the calibration point. Reconstruction should
  // yield ≈ host_time + 1s.
  const uint64_t kDeviceNs = 5'000'000'000ULL;
  auto cal = makeCalibrationAtNow(kDeviceNs);

  auto wall = cal.toWallClock(pack(kDeviceNs + 1'000'000'000ULL));
  expectClose(wall, cal.host_time + 1s);
}

TEST(GpuClockCalibrationTest, ToWallClock32BitWrapsCorrectly) {
  // Event near a 32-bit-tick wrap boundary should still reconstruct to
  // the correct full-precision time. We build a calibration where
  // device_ns >> 10 sits very close to UINT32_MAX so that an event a few
  // ms in the past or future crosses the 32-bit wrap.
  //
  // Pick device_ns such that (device_ns >> 10) == 0xFFFF'FFF0 (just
  // below UINT32_MAX). An event at device_ns + 100ms wraps to a small
  // low32 value; the math has to detect that "wrap" really means
  // "100ms in the future", not "~71 minutes in the past".
  const uint64_t kBaselineTicks = 0xFFFF'FFF0ULL;
  const uint64_t kDeviceNs = kBaselineTicks << US_TICK_TIMESTAMP_SHIFT;
  auto cal = makeCalibrationAtNow(kDeviceNs);

  auto wall = cal.toWallClock(pack(kDeviceNs + 100'000'000ULL)); // +100ms
  expectClose(wall, cal.host_time + 100ms);
}

// Past-only bias: events spanning the full ~73 min past lookback should
// reconstruct correctly, including right next to the 32-bit-tick wrap
// boundary. With the old symmetric ±2^31 split each of these would have
// been misread as a small future event a few minutes ahead.
TEST(GpuClockCalibrationTest, ToWallClock32BitNearWrapBoundary) {
  // Device clock past 5h so subtracting >70 minutes stays positive.
  const uint64_t kDeviceNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(5h).count();
  auto cal = makeCalibrationAtNow(kDeviceNs);

  for (auto past : {70min, 71min, 72min}) {
    SCOPED_TRACE("past=" + std::to_string(past.count()) + "min");
    const int64_t pastNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(past).count();
    auto wall =
        cal.toWallClock(pack(kDeviceNs - static_cast<uint64_t>(pastNs)));
    expectClose(wall, cal.host_time - past);
  }
}

// Beyond the ~73 min wrap window, 32 bits cannot disambiguate the event
// from one wrap_window closer to now. Document that a 74-min-past event
// aliases to (74min - 2^42 ns) ≈ 42 sec in the past — a fundamental
// limit of the packed timestamp format. Callers needing events older
// than ~73 min must use the 64-bit overload.
TEST(GpuClockCalibrationTest, ToWallClock32BitWrapsBeyondBoundary) {
  const uint64_t kDeviceNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(5h).count();
  auto cal = makeCalibrationAtNow(kDeviceNs);

  constexpr int64_t k74MinNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(74min).count();
  auto wall =
      cal.toWallClock(pack(kDeviceNs - static_cast<uint64_t>(k74MinNs)));

  constexpr auto kWrapWindow =
      std::chrono::nanoseconds(uint64_t{1} << 42); // 2^32 ticks * 1024 ns
  expectClose(wall, cal.host_time - 74min + kWrapWindow);
}

// Calibration captured long enough ago that (now - host_time) in ticks
// exceeds UINT32_MAX — i.e., the high 32 bits of now_ticks differ from
// the high 32 bits of (device_ns >> 10). The reconstruction must anchor
// against now()'s tick count (not the calibration's) for a recent event
// to come back at ≈ now().
//
// Two hours ≈ 1.7× the ~71-minute 32-bit-tick wrap window, so the
// elapsed device_ticks since calibration will have wrapped past
// UINT32_MAX at least once.
TEST(GpuClockCalibrationTest, ToWallClock32BitOldCalibration) {
  GlobaltimerCalibration cal{};
  cal.device_ns = 1'000'000'000ULL;
  cal.host_time = std::chrono::system_clock::now() - 2h;

  // The "current" device timestamp is device_ns + 2h. Pack it (truncate
  // to low 32 of the 1024ns ticks) and feed it to the reconstruction —
  // the result should be ≈ now(), even though pack() drops bits that
  // distinguish this event from one ~71 minutes earlier.
  constexpr int64_t kTwoHoursNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(2h).count();
  const uint64_t nowGpuNs = cal.device_ns + static_cast<uint64_t>(kTwoHoursNs);

  expectClose(
      cal.toWallClock(pack(nowGpuNs)), std::chrono::system_clock::now());
}

// End-to-end smoke test against the real singleton. Confirms that
// GlobaltimerCalibration::get() boots (kernel launches successfully,
// captures device_ns + host_time) and that the resulting calibration
// passes the same math invariants we tested with synthetic data.
TEST(GpuClockCalibrationTest, SingletonRoundTrip) {
  const auto& cal = GlobaltimerCalibration::get();

  // The 64-bit overload at the calibration point is exact.
  EXPECT_EQ(cal.toWallClock(cal.device_ns), cal.host_time);

  // The 32-bit overload at the calibration point reconstructs to ≈
  // host_time. Slack is wider here because the singleton was captured
  // at process start (potentially many seconds ago) so toWallClock's
  // internal now() has drifted from cal.host_time accordingly — the
  // reconstruction still snaps back to host_time via the signed-delta
  // math, but rounding to ~1024ns ticks contributes a small error.
  expectClose(cal.toWallClock(pack(cal.device_ns)), cal.host_time);
}
