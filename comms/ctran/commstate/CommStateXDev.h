// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <stdint.h>

namespace ctran {

#ifdef __CUDA_ARCH__
#define DEVICE_ATTRIBUTE __device__ __forceinline__
#else

#if defined(__HIP_PLATFORM_AMD__)
#define DEVICE_ATTRIBUTE __device__ __forceinline__
#else
#define DEVICE_ATTRIBUTE inline
#endif

#endif

// Define more lightweight struct for device side. Contents copied from
// CommStateX at CtranAlgo init time.
struct alignas(16) CommStateXDev {
  // FIXME: fields should be protected, and shared only with CommStateX;
  // Remove protected attribute due to build issue. Need fix later.
  int rank_;
  int pid_;
  int localRank_;
  int localRanks_;
  int nRanks_;
  int nNodes_;
  uint64_t commHash_;

  DEVICE_ATTRIBUTE int rank() const {
    return rank_;
  }

  DEVICE_ATTRIBUTE int localRank() const {
    return localRank_;
  }

  DEVICE_ATTRIBUTE int localRank(const int rank) const {
    return rank & (localRanks_ - 1);
  }

  DEVICE_ATTRIBUTE int nRanks() const {
    return nRanks_;
  }

  DEVICE_ATTRIBUTE int nLocalRanks() const {
    return localRanks_;
  }

  DEVICE_ATTRIBUTE int nNodes() const {
    return nNodes_;
  }

  DEVICE_ATTRIBUTE int node() const {
    return rank_ / localRanks_;
  }

  DEVICE_ATTRIBUTE int node(const int rank) const {
    return rank / localRanks_;
  }

  DEVICE_ATTRIBUTE int pid() const {
    return pid_;
  }

  DEVICE_ATTRIBUTE uint64_t commHash() const {
    return commHash_;
  }

  DEVICE_ATTRIBUTE int localRankToRank(const int localRank, const int node = -1)
      const {
    auto node_ = node;
    if (node == -1) {
      node_ = this->node();
    }
    return localRanks_ * node_ + localRank;
  }
};
} // namespace ctran
