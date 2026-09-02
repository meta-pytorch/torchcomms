// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdio>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

/*
 * Abort checks for device wait loops.
 *
 * `FT_ABORT_CHECK` is the primitive: it evaluates the handle once and reports
 * whether the caller must stop. `FT_ABORT_BREAK` and `FT_ABORT_RETURN` are the
 * two common shapes on top of it.
 *
 * Under `AbortBehavior::TRAP` all three log the *site* -- file, line and
 * function -- alongside the caller's message, so a log line says where the
 * abort or the timeout was observed rather than only that one happened. Under
 * the default `AbortBehavior::SKIP`, only the call that wins the shared reason
 * CAS logs. Every later observer stays silent, avoiding one printf per thread
 * while preserving the device callsite that first declared the abort.
 *
 * These live in the fault-tolerance module and deliberately depend on nothing
 * from Prims, so CTRAN and MCCL device code can use the same checks.
 */

// True during the CUDA or HIP device pass. HIP parses `__device__` bodies in
// the host pass too, so device-only intrinsics have to be gated.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define FT_IS_DEVICE_COMPILE 1
#else
#define FT_IS_DEVICE_COMPILE 0
#endif

#if FT_IS_DEVICE_COMPILE
#define FT_DEVICE_TRAP() __trap()
#else
#define FT_DEVICE_TRAP() ((void)0)
#endif

namespace comms::fault_tolerance::detail {

/*
 * Shared body behind the macros. `fmt` already carries the caller's message
 * plus the source location; the function name arrives as the last argument and
 * is consumed by the trailing `%s`.
 */
template <typename... Args>
__device__ __forceinline__ bool abortCheckAndLog(
    const AbortDevice& abort,
    const char* fmt,
    const char* firstWriterFmt,
    Args... args) {
#if FT_IS_DEVICE_COMPILE
  bool flippedHere = false;
  const auto result = abort.check(&flippedHere);
  if (result == AbortCheckResult::CONTINUE) {
    return false;
  }
  if (flippedHere) {
    // The transition from NONE happens exactly once per communicator. Print
    // regardless of behavior so the production SKIP path retains the winning
    // device callsite. NOLINTNEXTLINE(facebook-security-vulnerable-printf)
    printf(firstWriterFmt, args...);
  } else if (result == AbortCheckResult::TRAP) {
    // NOLINTNEXTLINE(facebook-security-vulnerable-printf)
    printf(fmt, args...);
  }
  if (result == AbortCheckResult::TRAP) {
    FT_DEVICE_TRAP();
  }
  return true;
#else
  (void)abort;
  (void)fmt;
  (void)firstWriterFmt;
  ((void)args, ...);
  return false;
#endif
}

} // namespace comms::fault_tolerance::detail

#define FT_ABORT_STRINGIFY_(x) #x
#define FT_ABORT_STRINGIFY(x) FT_ABORT_STRINGIFY_(x)

// Source location, baked into the format string at compile time. Only the
// function name needs a runtime argument, so it is passed last.
#define FT_ABORT_SITE_SUFFIX_ \
  " [" __FILE__ ":" FT_ABORT_STRINGIFY(__LINE__) " %s]\n"

/*
 * Returns true when the caller must stop.
 *
 * `fmt` must be a string literal -- the source location is concatenated onto
 * it at compile time. `__func__` is expanded here, at the call site, so it
 * names the function containing the wait.
 *
 * A first-writer check prints the common first-writer marker in either
 * behavior. Other TRAP observations print the legacy CUDA abort message before
 * trapping; other SKIP observations stay silent.
 *
 * Use this where the exit needs more than a bare `break` or `return` -- for
 * example when the decision must be made warp-uniform before a barrier:
 *
 *     const bool stop = FT_ABORT_CHECK(abort, "waiting for peer %d", peer);
 *     if (__any_sync(0xFFFFFFFFU, stop)) {
 *       break;
 *     }
 */
#define FT_ABORT_CHECK(abort, fmt, ...)                        \
  ::comms::fault_tolerance::detail::abortCheckAndLog(          \
      (abort),                                                 \
      "CUDA ABORT ERROR: " fmt FT_ABORT_SITE_SUFFIX_,          \
      FT_ABORT_FIRST_WRITER_DEVICE_ fmt FT_ABORT_SITE_SUFFIX_, \
      ##__VA_ARGS__,                                           \
      __func__)

/*
 * Leaves the enclosing loop when an abort is visible.
 *
 * Deliberately not wrapped in do/while(0): the `break` has to belong to the
 * caller's loop. Only valid inside a loop -- outside one it would bind to the
 * nearest enclosing switch. Use FT_ABORT_RETURN there.
 *
 * The trailing `else (void)0` is what makes the expansion safe in an unbraced
 * `if`. Without it the macro is a naked `if`, so `if (c) FT_ABORT_BREAK(...);
 * else fallback();` binds the caller's `else` to the macro's `if` -- and the
 * misbinding is silent, running `fallback()` exactly when `c` held and nothing
 * had aborted. Consuming the `else` here completes the macro's own `if`, so the
 * caller's `else` binds to the caller's `if`, which is what the caller wrote.
 * Note this is a silent repair, not a diagnostic: the misbinding shape still
 * compiles either way, so `BreakDoesNotCaptureACallerElse` has to catch a
 * regression at runtime rather than at build time.
 */
#define FT_ABORT_BREAK(abort, fmt, ...)            \
  if (FT_ABORT_CHECK(abort, fmt, ##__VA_ARGS__)) { \
    break;                                         \
  } else                                           \
    (void)0

/* Returns `value` from the enclosing function when an abort is visible. */
#define FT_ABORT_RETURN(abort, value, fmt, ...)      \
  do {                                               \
    if (FT_ABORT_CHECK(abort, fmt, ##__VA_ARGS__)) { \
      return value;                                  \
    }                                                \
  } while (0)
