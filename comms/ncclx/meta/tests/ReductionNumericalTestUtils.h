// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_bf16.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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

// Arbitrary fixed seeds for reproducible random test inputs. Each input
// distribution uses a distinct base seed so its stream is disjoint from the
// others; the Uniform seed is unchanged so Uniform inputs stay bit-stable.
constexpr uint32_t kReductionInputSeed = 12345;
constexpr uint32_t kReductionNormalInputSeed = 67890;
constexpr const char* kPrintActualOutputEnv =
    "REDUCTION_NUMERICAL_PRINT_ACTUAL";

// Uniform and Normal are deterministic random draws; Corner is a fixed set of
// edge-case values (zeros, signed zeros, min normal, subnormal, and an O(1)
// cancellation pair +/-1.0) rather than a random distribution.
enum class InputDistribution { Uniform, Normal, Corner };

inline std::string inputDistributionName(InputDistribution distribution) {
  switch (distribution) {
    case InputDistribution::Normal:
      return "Normal";
    case InputDistribution::Corner:
      return "Corner";
    case InputDistribution::Uniform:
      return "Uniform";
  }
  return "Uniform";
}

struct AllCloseTolerance {
  double rtol{0.0};
  double atol{0.0};
};

template <typename T>
void appendRandomInputs(
    std::vector<T>& input,
    size_t count,
    int rank,
    int lane,
    InputDistribution distribution = InputDistribution::Uniform) {
  // The element at local index i is a deterministic value keyed by (rank,
  // lane). Uniform and Normal draw from a seeded generator (distinct base seed
  // per distribution); Corner uses a fixed edge-case pattern. AllReduce and
  // Reduce use one lane; ReduceScatter appends one lane per output rank.
  using HostT = typename DataTypeTraits<T>::HostT;
  input.reserve(input.size() + count);

  if (distribution == InputDistribution::Corner) {
    // Subnormal/edge-of-range values, rotated per (rank, lane). Focused on the
    // subnormal regime (zeros, signed zeros, smallest and mid subnormals, min
    // normal) plus an O(1) cancellation pair; intentionally avoids large
    // magnitudes so this probes subnormal handling and flush-to-zero rather
    // than cross-magnitude cancellation. FP32-oriented; BF16 subnormal behavior
    // is still under review.
    constexpr double kFloatMinNormal =
        static_cast<double>(std::numeric_limits<float>::min());
    static const double kCornerValues[] = {
        0.0,
        -0.0,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        kFloatMinNormal * 0.5, // a subnormal between denorm_min and min normal
        -kFloatMinNormal * 0.5,
        kFloatMinNormal,
        -kFloatMinNormal,
        1.0,
        -1.0,
    };
    constexpr size_t kNumCorner =
        sizeof(kCornerValues) / sizeof(kCornerValues[0]);
    const size_t offset = static_cast<size_t>(rank + lane);
    for (size_t i = 0; i < count; ++i) {
      input.push_back(
          DataTypeTraits<T>::toDevice(
              static_cast<HostT>(kCornerValues[(i + offset) % kNumCorner])));
    }
    return;
  }

  const uint32_t baseSeed = distribution == InputDistribution::Normal
      ? kReductionNormalInputSeed
      : kReductionInputSeed;
  std::seed_seq seed{
      baseSeed,
      static_cast<uint32_t>(rank),
      static_cast<uint32_t>(lane),
  };
  std::mt19937_64 generator(seed);
  if (distribution == InputDistribution::Normal) {
    std::normal_distribution<double> normal(0.0, 1.0);
    for (size_t i = 0; i < count; ++i) {
      input.push_back(
          DataTypeTraits<T>::toDevice(static_cast<HostT>(normal(generator))));
    }
  } else {
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (size_t i = 0; i < count; ++i) {
      input.push_back(
          DataTypeTraits<T>::toDevice(static_cast<HostT>(uniform(generator))));
    }
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

// Aggregate error metrics comparing the NCCL output against the FP64 reference.
// These complement the elementwise allclose check with whole-vector signals
// that can carry tighter, more meaningful bounds than any single element (for
// example relative L2 error and cosine similarity between the output and the
// reference).
struct AggregateMetrics {
  double relativeL2Error{0.0}; // ||actual - ref||_2 / ||ref||_2
  double cosineSimilarity{1.0}; // dot(actual, ref) / (||actual|| * ||ref||)
};

// Degenerate-case handling (report-only diagnostics; the elementwise allclose
// check remains the pass/fail gate):
//   - relativeL2Error is NaN when the reference is all-zero (relative error is
//     undefined); the elementwise allclose check still bounds |actual - 0|.
//   - cosineSimilarity is 1.0 only when both vectors are all-zero, and 0.0 when
//     exactly one is all-zero, so a zero-vs-nonzero mismatch is not masked as
//     perfect similarity.
template <typename T>
AggregateMetrics computeAggregateMetrics(
    const std::vector<T>& observed,
    const std::vector<double>& reference) {
  // Precondition: observed provides at least one entry per reference element.
  assert(observed.size() >= reference.size());
  double diffSq = 0.0;
  double refSq = 0.0;
  double actualSq = 0.0;
  double dot = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double actual =
        static_cast<double>(DataTypeTraits<T>::toHost(observed[i]));
    const double ref = reference[i];
    const double absError = std::abs(actual - ref);
    diffSq += absError * absError;
    refSq += ref * ref;
    actualSq += actual * actual;
    dot += actual * ref;
  }
  AggregateMetrics metrics;
  // Relative L2 is undefined for an all-zero reference; report NaN rather than
  // an absolute value under the relative_l2 field. The elementwise allclose
  // check still bounds |actual - 0| in that case.
  metrics.relativeL2Error =
      refSq > 0.0 ? std::sqrt(diffSq) / std::sqrt(refSq) : std::nan("");
  const double norms = std::sqrt(actualSq) * std::sqrt(refSq);
  // Only truly-identical all-zero vectors are perfectly similar; a zero-vs-
  // nonzero pair is maximal disagreement, not cosine 1.0.
  metrics.cosineSimilarity =
      norms > 0.0 ? dot / norms : (actualSq == 0.0 && refSq == 0.0 ? 1.0 : 0.0);
  return metrics;
}

inline void printAggregateMetrics(
    const AggregateMetrics& metrics,
    int rank,
    const std::string& caseName) {
  std::cout << "REDUCTION_NUMERICAL_AGGREGATE case=" << caseName
            << " rank=" << rank << " relative_l2_error="
            << fmt::format("{:.6g}", metrics.relativeL2Error)
            << " cosine_similarity="
            << fmt::format("{:.9g}", metrics.cosineSimilarity) << std::endl;
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
  // Aggregate metrics are diagnostics only; they do not affect the elementwise
  // pass/fail gate. Emit them on failure and whenever output diagnostics are
  // on.
  if (mismatches > 0 || shouldPrintActualOutput()) {
    printAggregateMetrics(
        computeAggregateMetrics(observed, reference), rank, caseName);
  }
  return mismatches;
}

} // namespace ncclx::test::numerics
