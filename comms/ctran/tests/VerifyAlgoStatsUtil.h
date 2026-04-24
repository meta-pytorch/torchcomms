// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include "comms/ctran/CtranComm.h"

namespace ctran::test {

// TODO: Migrate to colltrace once CUDA graph colltrace support is fixed.
class VerifyAlgoStatsHelper {
 public:
  ~VerifyAlgoStatsHelper();

  // Enable AlgoStats tracing. Must be called before CtranComm creation.
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
  std::vector<std::string> oldColltrace_;
};

} // namespace ctran::test
