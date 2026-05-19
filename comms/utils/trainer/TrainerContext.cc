// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/trainer/TrainerContext.h"

#include <atomic>
#include <chrono>
#include <cstring>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace {

// Pack iteration (high 64 bits) and timestampUs (low 64 bits) into a single
// 128-bit atomic so readers always see a consistent pair.
__int128 packIteration(int64_t iteration, int64_t timestampUs) {
  __int128 v = 0;
  std::memcpy(&v, &timestampUs, 8);
  std::memcpy(reinterpret_cast<char*>(&v) + 8, &iteration, 8);
  return v;
}

IterationSnapshot unpackIteration(__int128 v) {
  IterationSnapshot snap;
  std::memcpy(&snap.timestampUs, &v, 8);
  std::memcpy(&snap.iteration, reinterpret_cast<const char*>(&v) + 8, 8);
  return snap;
}

// SSE2 MOVDQA is single-copy atomic for naturally-aligned 16-byte addresses
// on every production x86_64 CPU, avoiding the __atomic_{store,load}_16
// libatomic libcalls that GCC emits for std::atomic<__int128>.
#if defined(__x86_64__)

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
alignas(16) __int128 gIterationPacked = packIteration(/*iteration=*/-1,
                                                      /*timestampUs=*/0);

void releaseStore128(__int128* dst, __int128 val) {
  __m128i v;
  __builtin_memcpy(&v, &val, sizeof(v));
  std::atomic_thread_fence(std::memory_order_release);
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

__int128 acquireLoad128(const __int128* src) {
  __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
  std::atomic_thread_fence(std::memory_order_acquire);
  __int128 result;
  __builtin_memcpy(&result, &v, sizeof(result));
  return result;
}

#else

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::atomic<__int128> gIterationPacked{
    packIteration(/*iteration=*/-1, /*timestampUs=*/0)};

void releaseStore128(std::atomic<__int128>* dst, __int128 val) {
  dst->store(val, std::memory_order_release);
}

__int128 acquireLoad128(const std::atomic<__int128>* src) {
  return src->load(std::memory_order_acquire);
}

#endif

} // namespace

void ncclxSetIteration(int64_t iteration) {
  auto timestampUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  releaseStore128(&gIterationPacked, packIteration(iteration, timestampUs));
}

int64_t ncclxGetIteration() {
  return ncclxGetIterationSnapshot().iteration;
}

int64_t ncclxGetIterationTimestampUs() {
  return ncclxGetIterationSnapshot().timestampUs;
}

IterationSnapshot ncclxGetIterationSnapshot() {
  return unpackIteration(acquireLoad128(&gIterationPacked));
}
