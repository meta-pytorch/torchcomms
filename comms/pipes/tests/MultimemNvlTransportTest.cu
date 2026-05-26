// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultimemNvlTransportTest.cuh"

#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

namespace {

constexpr uint64_t kUserSetSignalValue = 0x1234;
constexpr uint64_t kInternalSetSignalValue = 0x5678;

__global__ void multimemStoreSignalKernel(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks,
    MultimemNvlTransportTestResult* result) {
  auto group = make_block_group();

  const std::size_t dst_offset =
      static_cast<std::size_t>(rank) * bytes_per_rank;
  transport.store<4>(group, dst_offset, src, bytes_per_rank);
  transport.signal(group, 0, SignalOp::SIGNAL_ADD, 1);
  transport.signal_internal(group, 0, SignalOp::SIGNAL_ADD, 1);
  if (transport.userLocalSignals.size() > 1) {
    transport.signal(group, 1, SignalOp::SIGNAL_SET, kUserSetSignalValue);
  }
  if (transport.internalLocalSignals.size() > 1) {
    transport.signal_internal(
        group, 1, SignalOp::SIGNAL_SET, kInternalSetSignalValue);
  }
  transport.wait_signal_until(group, 0, CmpOp::CMP_GE, nRanks);
  transport.wait_internal_signal_until(group, 0, CmpOp::CMP_GE, nRanks);
  if (transport.userLocalSignals.size() > 1) {
    transport.wait_signal_until(group, 1, CmpOp::CMP_EQ, kUserSetSignalValue);
  }
  if (transport.internalLocalSignals.size() > 1) {
    transport.wait_internal_signal_until(
        group, 1, CmpOp::CMP_EQ, kInternalSetSignalValue);
  }

  if (group.is_leader()) {
    result->user_add_signal_value = transport.read_signal(0);
    result->internal_add_signal_value = transport.read_internal_signal(0);
    if (transport.userLocalSignals.size() > 1) {
      result->user_set_signal_value = transport.read_signal(1);
    }
    if (transport.internalLocalSignals.size() > 1) {
      result->internal_set_signal_value = transport.read_internal_signal(1);
    }
    result->user_signal_addr =
        reinterpret_cast<uintptr_t>(transport.userLocalSignals.data());
    result->internal_signal_addr =
        reinterpret_cast<uintptr_t>(transport.internalLocalSignals.data());
    result->user_signal_count = transport.userLocalSignals.size();
    result->internal_signal_count = transport.internalLocalSignals.size();
  }
}

__global__ void stridedMultimemStoreKernel(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks) {
  auto group = make_block_group();

  const std::size_t dst_offset =
      static_cast<std::size_t>(rank) * bytes_per_rank;
  auto* dstVec =
      reinterpret_cast<uint4*>(transport.multimem_data_at(dst_offset));
  auto* srcVec = reinterpret_cast<const uint4*>(src);
  detail::strided_multimem_store_aligned<2>(
      group, dstVec, srcVec, bytes_per_rank / sizeof(uint4));
  transport.signal(group, 0, SignalOp::SIGNAL_ADD, 1);
  transport.wait_signal_until(group, 0, CmpOp::CMP_GE, nRanks);
}

__global__ void multimemRawStoreKernel(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t dst_offset,
    std::size_t bytes,
    int nRanks) {
  auto group = make_block_group();
  transport.store<4>(group, dst_offset, src, bytes);
  transport.signal(group, 0, SignalOp::SIGNAL_ADD, 1);
  transport.wait_signal_until(group, 0, CmpOp::CMP_GE, nRanks);
}

} // namespace

void launchMultimemStoreSignalTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks,
    MultimemNvlTransportTestResult* result,
    cudaStream_t stream) {
  multimemStoreSignalKernel<<<1, 256, 0, stream>>>(
      transport, src, bytes_per_rank, rank, nRanks, result);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchStridedMultimemStoreTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t bytes_per_rank,
    int rank,
    int nRanks,
    cudaStream_t stream) {
  stridedMultimemStoreKernel<<<1, 256, 0, stream>>>(
      transport, src, bytes_per_rank, rank, nRanks);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchMultimemRawStoreTest(
    MultimemNvlTransportDevice transport,
    const char* src,
    std::size_t dst_offset,
    std::size_t bytes,
    int nRanks,
    cudaStream_t stream) {
  multimemRawStoreKernel<<<1, 256, 0, stream>>>(
      transport, src, dst_offset, bytes, nRanks);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
