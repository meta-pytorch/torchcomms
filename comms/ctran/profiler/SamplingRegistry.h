// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace ctran {

// Encapsulates the sampling decision for algo profiling.
// Traces operations where sampleCount is a multiple of samplingWeight.
// (1 = every op, N = every Nth op, 0 or negative = never trace)
class SamplingRegistry {
 public:
  explicit SamplingRegistry(int samplingWeight);

  void setSampleCount(int sampleCount);

  bool shouldTrace() const;

 private:
  int samplingWeight_;
  int sampleCount_{-1};
};

} // namespace ctran
