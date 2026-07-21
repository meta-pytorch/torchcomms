// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/profiler/Profiler.h"

namespace ctran::test {

// Owns a zero-initialized CtranComm-sized buffer for constructing a Profiler
// without pulling in the heavy ctran lib. Safe because the Profiler only reads
// comm_->logMetaData_, a trivial POD struct.
struct FakeProfilerComm {
  // std::vector<char> only guarantees alignof(char), so the reinterpret_cast
  // below is well-defined only while CtranComm is not over-aligned past the
  // default-new alignment the allocator actually provides.
  static_assert(alignof(CtranComm) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);

  std::vector<char> buffer = std::vector<char>(sizeof(CtranComm), 0);

  CtranComm* get() {
    return reinterpret_cast<CtranComm*>(buffer.data());
  }
};

// Runs one collective's worth of Profiler calls in the algo hot-path order:
// init -> BUF_REG / ALGO_CTRL / ALGO_DATA phases -> report.
inline void runOneCollective(Profiler& p, int opCount, int samplingWeight) {
  p.initForEachColl(opCount, samplingWeight);
  p.startEvent(ProfilerEvent::BUF_REG);
  p.endEvent(ProfilerEvent::BUF_REG);
  p.startEvent(ProfilerEvent::ALGO_CTRL);
  p.endEvent(ProfilerEvent::ALGO_CTRL);
  p.startEvent(ProfilerEvent::ALGO_DATA);
  p.endEvent(ProfilerEvent::ALGO_DATA);
  p.reportToScuba();
}

} // namespace ctran::test
