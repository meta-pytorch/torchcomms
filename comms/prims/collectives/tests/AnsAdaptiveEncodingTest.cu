// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Built with `--device-c` + PIPES_ENABLE_ANS_COMPRESSION and device-linked
// against nvcompdx (see `:ans_adaptive_encoding_test_kernels` in BUCK).

#include "comms/prims/collectives/tests/AnsAdaptiveEncodingTest.cuh"

#include <stdexcept>
#include <string>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyOpFeedback.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims::test {
namespace {

constexpr int kNumWarps = 8;
constexpr int kBlockThreads = kNumWarps * 32;
using Comp = AnsCompress<kNumWarps, PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES>;

// Stands in for a transport: replays one scripted backlog per query.
struct ScriptedFeedback {
  const uint64_t* pending;
  uint32_t next;

  __device__ NicSendBacklog nic_send_backlog(ThreadGroup& group) {
    uint64_t value = 0;
    if (group.is_leader()) {
      value = pending[next];
    }
    ++next;
    return NicSendBacklog{
        .pending_bytes = group.broadcast<uint64_t>(value), .valid = true};
  }
};

__device__ __forceinline__ std::size_t
sub_chunk_bytes(std::size_t k, std::size_t nbytes, std::size_t chunkBytes) {
  const std::size_t off = k * chunkBytes;
  return nbytes - off < chunkBytes ? nbytes - off : chunkBytes;
}

__global__ __launch_bounds__(kBlockThreads, 1) void ans_adaptive_send_kernel(
    const char* src,
    char* staging,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride,
    const uint64_t* pending,
    bool feedback,
    std::size_t minPendingBytes,
    uint64_t* sizes) {
  auto group = make_block_group();
  ScriptedFeedback scripted{.pending = pending, .next = 0};
  const Comp::SendArgs<ScriptedFeedback> args{
      .alignedAuxBuf = nullptr,
      .feedback = feedback ? &scripted : nullptr,
      .minPendingBytes = minPendingBytes};
  const std::size_t numChunks = (nbytes + chunkBytes - 1) / chunkBytes;
  for (std::size_t k = 0; k < numChunks; ++k) {
    const std::size_t staged = Comp::send(
        staging + k * stagingStride,
        src + k * chunkBytes,
        sub_chunk_bytes(k, nbytes, chunkBytes),
        group,
        k * chunkBytes,
        args);
    group.sync();
    if (group.is_leader()) {
      sizes[k] = staged;
    }
  }
}

__global__ __launch_bounds__(kBlockThreads, 1) void ans_adaptive_recv_kernel(
    const char* staging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride) {
  auto group = make_block_group();
  const std::size_t numChunks = (nbytes + chunkBytes - 1) / chunkBytes;
  for (std::size_t k = 0; k < numChunks; ++k) {
    (void)Comp::recv(
        dst + k * chunkBytes,
        staging + k * stagingStride,
        sub_chunk_bytes(k, nbytes, chunkBytes),
        group,
        k * chunkBytes);
    group.sync();
  }
}

__global__ __launch_bounds__(kBlockThreads, 1) void ans_adaptive_forward_kernel(
    const char* staging,
    char* fwdStaging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride) {
  auto group = make_block_group();
  const std::size_t numChunks = (nbytes + chunkBytes - 1) / chunkBytes;
  for (std::size_t k = 0; k < numChunks; ++k) {
    (void)Comp::recv_forward(
        dst + k * chunkBytes,
        fwdStaging + k * stagingStride,
        staging + k * stagingStride,
        sub_chunk_bytes(k, nbytes, chunkBytes),
        group,
        k * chunkBytes);
    group.sync();
  }
}

void check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(what) + " launch failed: " + cudaGetErrorString(err));
  }
}

} // namespace

void ans_adaptive_send(
    const char* src,
    char* staging,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride,
    const uint64_t* pending,
    bool feedback,
    std::size_t minPendingBytes,
    uint64_t* sizes) {
  ans_adaptive_send_kernel<<<1, kBlockThreads>>>(
      src,
      staging,
      nbytes,
      chunkBytes,
      stagingStride,
      pending,
      feedback,
      minPendingBytes,
      sizes);
  check_launch("ans_adaptive_send");
}

void ans_adaptive_recv(
    const char* staging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride) {
  ans_adaptive_recv_kernel<<<1, kBlockThreads>>>(
      staging, dst, nbytes, chunkBytes, stagingStride);
  check_launch("ans_adaptive_recv");
}

void ans_adaptive_forward(
    const char* staging,
    char* fwdStaging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride) {
  ans_adaptive_forward_kernel<<<1, kBlockThreads>>>(
      staging, fwdStaging, dst, nbytes, chunkBytes, stagingStride);
  check_launch("ans_adaptive_forward");
}

} // namespace comms::prims::test
