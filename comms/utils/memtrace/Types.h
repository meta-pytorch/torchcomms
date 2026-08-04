// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Shared, device-safe memtrace types (stdlib-only, no folly). Included by both
// the memtrace and logger modules; kept in a standalone header/target to avoid
// a circular dependency between the memory_trace and event_mgr build targets.

#pragma once

#include <string>
#include <utility>

namespace meta::comms::memtrace {

// Callsite metadata for a memtrace record. Baseline NCCL call sites pass a bare
// string, which implicitly constructs a MemCallsite with scope=kNccl, so they
// require no changes. ctran call sites pass {Scope::kCtran, function} to tag
// their allocations. memtrace owns splitting this into the scuba `callsite`
// (function) and `scope` columns and into the per-source in-memory buckets, so
// callers never format it themselves.
struct MemCallsite {
  // Layer that requested a GPU memory allocation: baseline NCCL (kNccl), ctran
  // (kCtran), or mccl (kMccl).
  enum class Scope { kNccl, kCtran, kMccl };

  Scope scope{Scope::kNccl};
  std::string function;

  // NOLINTNEXTLINE(google-explicit-constructor): implicit keeps baseline
  // (bare-string) call sites unchanged.
  MemCallsite(const char* function)
      : function(function != nullptr ? function : "") {}
  // NOLINTNEXTLINE(google-explicit-constructor): implicit keeps baseline
  // (bare-string) call sites unchanged.
  MemCallsite(std::string function) : function(std::move(function)) {}
  MemCallsite(Scope scope, std::string function)
      : scope(scope), function(std::move(function)) {}
};

} // namespace meta::comms::memtrace
