// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/SamplingRegistry.h"

namespace ctran {

SamplingRegistry::SamplingRegistry(int samplingWeight)
    : samplingWeight_(samplingWeight) {}

void SamplingRegistry::setSampleCount(int sampleCount) {
  sampleCount_ = sampleCount;
}

bool SamplingRegistry::shouldTrace() const {
  return samplingWeight_ > 0 && sampleCount_ >= 0 &&
      (sampleCount_ % samplingWeight_) == 0;
}

} // namespace ctran
