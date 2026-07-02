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

// Hex-encode the raw bytes of a host vector and emit one structured diagnostic
// line: "<tag> collective=.. case=.. rank=.. bytes=<hex>". Downstream tooling
// decodes the bytes using the element type implied by <tag> (BF16 for actual
// and bf16-hop, FP64 for the reference).
template <typename Elem>
void printNumericalBytes(
    const char* tag,
    const std::vector<Elem>& values,
    int rank,
    const std::string& collectiveName,
    const std::string& caseName) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  const size_t byteCount = values.size() * sizeof(Elem);
  std::string hex;
  hex.resize(byteCount * 2);
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (size_t i = 0; i < byteCount; ++i) {
    hex[2 * i] = kHexDigits[bytes[i] >> 4];
    hex[2 * i + 1] = kHexDigits[bytes[i] & 0x0f];
  }
  std::cout << tag << " collective=" << collectiveName << " case=" << caseName
            << " rank=" << rank << " bytes=" << hex << std::endl;
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

  printNumericalBytes(
      "REDUCTION_NUMERICAL_ACTUAL", observed, rank, collectiveName, caseName);
}

// Emit the host-side FP64 reference values (8 bytes per element) so passing
// runs also record the reference, not just the NCCL actual output.
inline void printReferenceBytes(
    const std::vector<double>& reference,
    int rank,
    const std::string& collectiveName,
    const std::string& caseName) {
  if (!shouldPrintActualOutput()) {
    return;
  }
  printNumericalBytes(
      "REDUCTION_NUMERICAL_REFERENCE",
      reference,
      rank,
      collectiveName,
      caseName);
}

// Accumulate contributions in order, rounding the running sum to T (BF16) after
// each addition. This models NCCL's per-hop BF16 downcast using a simple
// rank-order fold; it intentionally does not reproduce NCCL's exact reduction
// order/topology, so it is a directional diagnostic, not a bitwise predictor.
template <typename T>
T bf16HopReduce(const std::vector<double>& contributions) {
  using HostT = typename DataTypeTraits<T>::HostT;
  double acc = 0.0;
  for (const double value : contributions) {
    acc = static_cast<double>(DataTypeTraits<T>::toHost(
        DataTypeTraits<T>::toDevice(static_cast<HostT>(acc + value))));
  }
  return DataTypeTraits<T>::toDevice(static_cast<HostT>(acc));
}

// Emit the BF16 downcast-per-hop simulated reduction result (2 bytes per
// element).
template <typename T>
void printBf16HopBytes(
    const std::vector<T>& bf16Hop,
    int rank,
    const std::string& collectiveName,
    const std::string& caseName) {
  if (!shouldPrintActualOutput()) {
    return;
  }
  printNumericalBytes(
      "REDUCTION_NUMERICAL_BF16HOP", bf16Hop, rank, collectiveName, caseName);
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
