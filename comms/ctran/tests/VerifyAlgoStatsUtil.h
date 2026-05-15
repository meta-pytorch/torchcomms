// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>
#include <string>
#include <vector>
#include "comms/ctran/CtranComm.h"

namespace ctran::test {

// TODO: Migrate to colltrace once CUDA graph colltrace support is fixed.
class VerifyAlgoStatsHelper {
 public:
  ~VerifyAlgoStatsHelper();

  // Enable AlgoStats tracing. Must be called before CtranComm creation.
  // Overrides NCCL_COLLTRACE via both setenv (so the value survives any
  // subsequent ncclCvarInit() re-read) and the in-memory cvar (so the value is
  // visible immediately, regardless of call ordering).
  void enable();

  void verify(
      CtranComm* comm,
      const std::string& collective,
      const std::string& expectedAlgoSubstr) const;

  void verifyNot(
      CtranComm* comm,
      const std::string& collective,
      const std::string& unexpectedAlgoSubstr) const;

  void dump(CtranComm* comm, const std::string& collective) const;

 private:
  bool enabled_{false};
  std::optional<std::string> oldEnvValue_;
  std::vector<std::string> oldColltrace_;
};

} // namespace ctran::test
