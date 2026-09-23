// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

// The value types crossing the lifecycle-feed boundary. Keep them free of
// colltrace, and of anything carrying state of its own: these cross between
// shared objects, so a type here with a registrar or a reclamation domain
// behind it gets one copy per backend (comms/observatory/README.md).

namespace meta::comms::colltrace {

enum class LifecycleEventType : uint8_t {
  kEnqueue,
  kStart,
  kEnd,
};

// What a collective captured into a CUDA graph is. A replay reports the id the
// capture was given and carries no record of its own, so a caller that did not
// see the capture has no other way to learn what is running.
struct CapturedCollDescription {
  std::string opName;
  std::string algoName;
  std::string dataType;
  std::optional<uint64_t> count;

  bool operator==(const CapturedCollDescription&) const = default;
};

struct LifecycleEventRecord {
  std::optional<uint64_t> replayId;
  uint64_t commId{0};
  uint64_t collId{0};
  std::optional<uint64_t> capturedCollId;
  LifecycleEventType eventType{LifecycleEventType::kEnqueue};
  // The same type as ICollWaitEvent's alias, spelled out: naming that alias
  // here would pull the wait-event interface into observatory.
  std::chrono::system_clock::time_point timestamp{};

  bool operator==(const LifecycleEventRecord&) const = default;
};

} // namespace meta::comms::colltrace
