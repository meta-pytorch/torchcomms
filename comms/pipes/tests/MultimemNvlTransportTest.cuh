// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comms/pipes/MultimemNvlTransportDevice.cuh"

namespace comms::pipes::test {

struct MultimemNvlTransportTestResult {
  uint64_t user_add_signal_value{0};
  uint64_t user_set_signal_value{0};
  uint64_t internal_add_signal_value{0};
  uint64_t internal_set_signal_value{0};
  uintptr_t user_signal_addr{0};
  uintptr_t internal_signal_addr{0};
  uint32_t user_signal_count{0};
  uint32_t internal_signal_count{0};
};

void launchMultimemStoreSignalTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks,
    MultimemNvlTransportTestResult* result,
    cudaStream_t stream = nullptr);

void launchStridedMultimemStoreTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks,
    cudaStream_t stream = nullptr);

void launchMultimemRawStoreTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t dst_offset,
    std::size_t bytes,
    int nRanks,
    cudaStream_t stream = nullptr);

} // namespace comms::pipes::test
