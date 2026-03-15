// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/CppAttributes.h>
#include <folly/Synchronized.h>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
namespace ncclx {

struct HintKeys {
  static constexpr std::string_view kCollTraceCrashOnAsyncError =
      "ncclx.colltrace.crashOnAsyncError";
  static constexpr std::string_view kCollTraceCrashOnTimeout =
      "ncclx.colltrace.crashOnTimeout";
  static constexpr std::string_view kCollTraceTimeoutMs =
      "ncclx.colltrace.timeoutMs";
  // enable/disable Ctran at communicator creation time
  // NOTE: torch eager init mode is required; otherwise, the hint to
  // communicator mapping may be incorrect
  static constexpr std::string_view kCommUseCtran = "ncclx.comm.useCtran";
  // per-communicator ReduceScatter algorithm override
  // Format: "<redop>:<algo>" e.g., "avg:patavg"
  static constexpr std::string_view kCommAlgoReduceScatter =
      "ncclx.comm.algo_reducescatter";
  // disable local (P2P and SHM) transports at communicator creation time
  static constexpr std::string_view kCommNoLocal = "ncclx.comm.nolocal";
};

constexpr std::array<std::string_view, 6> kHintKeysArray = {
    HintKeys::kCollTraceCrashOnAsyncError,
    HintKeys::kCollTraceCrashOnTimeout,
    HintKeys::kCollTraceTimeoutMs,
    HintKeys::kCommUseCtran,
    HintKeys::kCommAlgoReduceScatter,
    HintKeys::kCommNoLocal};

using GlobalSetHintHook =
    std::function<void(const std::string& key, const std::string& val)>;
using GlobalResetHintHook = std::function<void(const std::string& key)>;

struct GlobalHintEntry {
  GlobalSetHintHook setHook{nullptr};
  GlobalResetHintHook resetHook{nullptr};
};

class GlobalHints {
 public:
  GlobalHints();
  ~GlobalHints() {}

  static std::shared_ptr<GlobalHints> getInstance();

  // Register a hint key.
  // Input:
  //   - key: the hint key
  //   - entry: optional hook functions to be registered with the key.
  // Return:
  //   - true: if the hintkey is registered successfully
  //   - false: if the hintkey is already registered
  bool regHintEntry(std::string key, GlobalHintEntry& entry);
  bool regHintEntry(std::string key);

  // Set the value for a global hint.
  // Input:
  //   - key: the hint key
  //   - val: the hint value
  // Return:
  //   - true: if the hint is set successfully
  //   - false: if the hint key is not available
  bool setHint(std::string key, std::string val);

  // Get the value for a global hint
  // Input:
  //   - key: the hint key
  // Return:
  //   - std::string: the hint value if the hint key is available and set
  //   - std::nullopt: if the hint key is not available or not set
  std::optional<std::string> getHint(const std::string& key);

  // Reset the value for a global hint. If the hint key is available but is not
  // set, it is a no-op.
  // Input:
  //   - key: the hint key
  // Return:
  //   - true: if the hint is reset successfully
  //   - false: if the hint key is not available
  bool resetHint(const std::string& key);

  void testOnlyReset();

 private:
  const GlobalHintEntry* FOLLY_NULLABLE getHintEntry(const std::string& key);

  // Optional hook functions to be registered for each hint key.
  folly::Synchronized<std::unordered_map<std::string, GlobalHintEntry>>
      hintEntries_;
  // key value store for hints
  folly::Synchronized<std::unordered_map<std::string, std::string>> hintVals_{};
};

bool setGlobalHint(const char* key, const char* val);
bool resetGlobalHint(const char* key);
bool resetGlobalHint(const std::string& key);
std::optional<std::string> getGlobalHint(std::string_view key);

template <typename T>
std::optional<T> getTypedGlobalHint(std::string_view key);

// Used for testing only. Do not use in production.
void testOnlyResetGlobalHints();

} // namespace ncclx
