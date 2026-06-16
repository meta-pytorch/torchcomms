// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_bf16.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "comms/testinfra/TestUtils.h"

namespace ncclx::test::numerics {

constexpr size_t kMaxPrintedMismatches = 10;

struct ExpectedValue {
  double value{0.0};
  double sumAbsInputs{0.0};
  int numInputs{0};
};

inline double bf16Ulp(double value) {
  const double absValue = std::max(std::abs(value), 1e-30);
  int exponent = 0;
  std::frexp(absValue, &exponent);
  return std::ldexp(1.0, exponent - 8);
}

inline double rawReductionInput(int rank, size_t index, int lane) {
  const double sign =
      ((rank + lane + static_cast<int>(index)) % 2 == 0) ? 1.0 : -1.0;
  const double rankTerm = 0.25 * static_cast<double>(rank + 1);
  const double laneTerm = 0.03125 * static_cast<double>((lane % 17) - 8);
  const double indexTerm =
      0.0009765625 * static_cast<double>(static_cast<int>(index % 257) - 128);
  const double jitter =
      0.00013 *
      static_cast<double>(
          (static_cast<int>((index * 17) % 97) + rank * 13 + lane * 7) % 41 -
          20);
  return sign * (1.0 + rankTerm) + laneTerm + indexTerm + jitter;
}

template <typename T>
T makeDeviceInput(int rank, size_t index, int lane) {
  return DataTypeTraits<T>::toDevice(
      static_cast<typename DataTypeTraits<T>::HostT>(
          rawReductionInput(rank, index, lane)));
}

template <typename T>
double referenceInput(int rank, size_t index, int lane) {
  return static_cast<double>(
      DataTypeTraits<T>::toHost(makeDeviceInput<T>(rank, index, lane)));
}

template <typename T>
double observedValue(T value) {
  return static_cast<double>(DataTypeTraits<T>::toHost(value));
}

template <typename T>
double tolerance(const ExpectedValue& expected) {
  const double eps = static_cast<double>(
      std::numeric_limits<typename DataTypeTraits<T>::HostT>::epsilon());
  return std::max(1e-10, expected.sumAbsInputs * eps * 64.0);
}

template <>
inline double tolerance<float>(const ExpectedValue& expected) {
  return std::max(
      1e-6,
      expected.sumAbsInputs *
          static_cast<double>(std::numeric_limits<float>::epsilon()) * 64.0);
}

template <>
inline double tolerance<__nv_bfloat16>(const ExpectedValue& expected) {
  const double outputQuantization = bf16Ulp(expected.value) * 8.0;
  const double accumulationMargin = expected.sumAbsInputs *
      std::pow(2.0, -7.0) *
      static_cast<double>(std::max(expected.numInputs, 1)) * 0.03;
  return std::max(outputQuantization, accumulationMargin);
}

template <typename T>
std::vector<T> makeAllReduceInput(int rank, size_t count) {
  std::vector<T> input(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = makeDeviceInput<T>(rank, i, 0);
  }
  return input;
}

template <typename T>
std::vector<ExpectedValue> allReduceExpected(size_t count, int numRanks) {
  std::vector<ExpectedValue> expected(count);
  for (size_t i = 0; i < count; ++i) {
    for (int rank = 0; rank < numRanks; ++rank) {
      const double value = referenceInput<T>(rank, i, 0);
      expected[i].value += value;
      expected[i].sumAbsInputs += std::abs(value);
      expected[i].numInputs++;
    }
  }
  return expected;
}

template <typename T>
size_t countMismatches(
    const T* deviceBuffer,
    const std::vector<ExpectedValue>& expected,
    cudaStream_t stream,
    int rank,
    const std::string& caseName) {
  std::vector<T> observed(expected.size());
  CUDACHECK_TEST(cudaMemcpyAsync(
      observed.data(),
      deviceBuffer,
      observed.size() * sizeof(T),
      cudaMemcpyDefault,
      stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double actual = observedValue(observed[i]);
    const double limit = tolerance<T>(expected[i]);
    const double error = std::abs(actual - expected[i].value);
    if (error > limit) {
      if (mismatches < kMaxPrintedMismatches) {
        ADD_FAILURE() << fmt::format(
            "{} rank={} index={} expected={:.17g} actual={:.17g} error={:.6g} tolerance={:.6g} sumAbsInputs={:.17g}",
            caseName,
            rank,
            i,
            expected[i].value,
            actual,
            error,
            limit,
            expected[i].sumAbsInputs);
      }
      mismatches++;
    }
  }
  return mismatches;
}

inline std::string countName(size_t count) {
  return fmt::format("Count_{}", count);
}

} // namespace ncclx::test::numerics
