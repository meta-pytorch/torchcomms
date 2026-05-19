// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/trainer/TrainerContext.h"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace {

// Pack iteration (high 64 bits) and timestampUs (low 64 bits) into a single
// 128-bit value so readers always observe a consistent pair. constexpr so
// gIterationPacked below is constant-initialized (no SIOF window).
constexpr __int128 packIteration(int64_t iteration, int64_t timestampUs) {
  return (static_cast<__int128>(static_cast<uint64_t>(iteration)) << 64) |
      static_cast<__int128>(static_cast<uint64_t>(timestampUs));
}

IterationSnapshot unpackIteration(__int128 v) {
  IterationSnapshot snap;
  snap.iteration = static_cast<int64_t>(static_cast<uint64_t>(v >> 64));
  snap.timestampUs = static_cast<int64_t>(static_cast<uint64_t>(v));
  return snap;
}

// std::atomic<__int128> is lock-free on x86_64 (cmpxchg16b, with -mcx16) and
// on aarch64 with LSE2; elsewhere it falls back to libatomic's lock table.
// We link -latomic unconditionally in the v2_29/v2_30 makefiles and the
// github CMakeLists so the fallback always resolves at link time.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::atomic<__int128> gIterationPacked{packIteration(-1, 0)};

} // namespace

void ncclxSetIteration(int64_t iteration) {
  auto timestampUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  gIterationPacked.store(
      packIteration(iteration, timestampUs), std::memory_order_release);
}

int64_t ncclxGetIteration() {
  return ncclxGetIterationSnapshot().iteration;
}

int64_t ncclxGetIterationTimestampUs() {
  return ncclxGetIterationSnapshot().timestampUs;
}

IterationSnapshot ncclxGetIterationSnapshot() {
  return unpackIteration(gIterationPacked.load(std::memory_order_acquire));
}
