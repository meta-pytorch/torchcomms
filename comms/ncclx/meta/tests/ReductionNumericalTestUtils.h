// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_bf16.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "comms/testinfra/TestUtils.h"

namespace ncclx::test::numerics {

constexpr size_t kMaxPrintedMismatches = 10;

// Per-datatype allclose-style bounds used by countMismatches:
//   abs(actual - reference) <= atol + rtol * abs(reference)
//
// FP32 uses tight bounds because its NCCL output should stay close to the FP64
// host reference. BF16 uses wider bounds because BF16 reduction paths can
// validly round partial sums at different points. These constants are fixed
// regression-test thresholds, not ULP-derived conformance limits.
constexpr double kFp32Rtol = 1e-5;
constexpr double kFp32Atol = 1e-5;
constexpr double kBfloat16Rtol = 2e-2;
constexpr double kBfloat16Atol = 3e-2;

// Arbitrary fixed seed for reproducible random test inputs.
constexpr uint32_t kReductionInputSeed = 12345;
constexpr const char* kPrintActualOutputEnv =
    "REDUCTION_NUMERICAL_PRINT_ACTUAL";

struct AllCloseTolerance {
  double rtol{0.0};
  double atol{0.0};
};

template <typename T>
void appendRandomInputs(
    std::vector<T>& input,
    size_t count,
    int rank,
    int lane) {
  // The element at local index i is the i-th sample from a deterministic
  // uniform generator seeded by (kReductionInputSeed, rank, lane). AllReduce
  // and Reduce use one lane; ReduceScatter appends one lane per output rank.
  std::seed_seq seed{
      kReductionInputSeed,
      static_cast<uint32_t>(rank),
      static_cast<uint32_t>(lane),
  };
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  input.reserve(input.size() + count);
  for (size_t i = 0; i < count; ++i) {
    input.push_back(
        DataTypeTraits<T>::toDevice(
            static_cast<typename DataTypeTraits<T>::HostT>(
                distribution(generator))));
  }
}

template <typename T>
std::vector<T> gatherInputs(
    const T* deviceInput,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    int numRanks) {
  T* gatheredDevice = nullptr;
  const size_t gatheredCount = count * static_cast<size_t>(numRanks);
  NCCLCHECK_TEST(
      ncclMemAlloc((void**)&gatheredDevice, gatheredCount * sizeof(T)));
  NCCLCHECK_TEST(ncclAllGather(
      deviceInput, gatheredDevice, count, datatype, comm, stream));

  std::vector<T> gathered(gatheredCount);
  CUDACHECK_TEST(cudaMemcpyAsync(
      gathered.data(),
      gatheredDevice,
      gatheredCount * sizeof(T),
      cudaMemcpyDefault,
      stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclMemFree(gatheredDevice));
  return gathered;
}

inline bool shouldPrintActualOutput() {
  return std::getenv(kPrintActualOutputEnv) != nullptr;
}

template <typename T>
void printActualOutputBytes(
    const T* deviceBuffer,
    size_t count,
    cudaStream_t stream,
    int rank,
    const std::string& collectiveName,
    const std::string& caseName) {
  if (!shouldPrintActualOutput()) {
    return;
  }

  std::vector<T> observed(count);
  CUDACHECK_TEST(cudaMemcpyAsync(
      observed.data(),
      deviceBuffer,
      observed.size() * sizeof(T),
      cudaMemcpyDefault,
      stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  const auto* bytes = reinterpret_cast<const unsigned char*>(observed.data());
  const size_t byteCount = observed.size() * sizeof(T);
  std::string hex;
  hex.resize(byteCount * 2);
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (size_t i = 0; i < byteCount; ++i) {
    hex[2 * i] = kHexDigits[bytes[i] >> 4];
    hex[2 * i + 1] = kHexDigits[bytes[i] & 0x0f];
  }

  std::cout << "REDUCTION_NUMERICAL_ACTUAL collective=" << collectiveName
            << " case=" << caseName << " rank=" << rank << " bytes=" << hex
            << std::endl;
}

template <typename T>
size_t countMismatches(
    const T* deviceBuffer,
    const std::vector<double>& reference,
    cudaStream_t stream,
    int rank,
    const std::string& caseName) {
  std::vector<T> observed(reference.size());
  CUDACHECK_TEST(cudaMemcpyAsync(
      observed.data(),
      deviceBuffer,
      observed.size() * sizeof(T),
      cudaMemcpyDefault,
      stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  size_t mismatches = 0;
  const AllCloseTolerance limits = std::is_same_v<T, __nv_bfloat16>
      ? AllCloseTolerance{.rtol = kBfloat16Rtol, .atol = kBfloat16Atol}
      : AllCloseTolerance{.rtol = kFp32Rtol, .atol = kFp32Atol};
  for (size_t i = 0; i < reference.size(); ++i) {
    // observed[i] is the actual NCCL output. Convert only to widen the typed
    // output for host-side comparison and failure diagnostics.
    const double actual =
        static_cast<double>(DataTypeTraits<T>::toHost(observed[i]));
    const double limit = limits.atol + limits.rtol * std::abs(reference[i]);
    const double error = std::abs(actual - reference[i]);
    if (error > limit) {
      if (mismatches < kMaxPrintedMismatches) {
        ADD_FAILURE() << fmt::format(
            "{} rank={} index={} reference={:.17g} actual={:.17g} error={:.6g} tolerance={:.6g}",
            caseName,
            rank,
            i,
            reference[i],
            actual,
            error,
            limit);
      }
      mismatches++;
    }
  }
  return mismatches;
}

} // namespace ncclx::test::numerics
