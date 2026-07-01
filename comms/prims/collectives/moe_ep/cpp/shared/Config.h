// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <string>

namespace comms::prims::moe_ep {

/**
 * Config — tuning knobs for the dispatch / combine kernels.
 *
 * The Python `comms.prims.collectives.moe_ep.Config(...)` ctor signature is:
 *
 *   Config(num_sms,
 *          num_max_nvl_chunked_send_tokens,
 *          num_max_nvl_chunked_recv_tokens,
 *          num_max_rdma_chunked_send_tokens=6,
 *          num_max_rdma_chunked_recv_tokens=128)
 *
 * Bound through `PyBindings.cpp`.
 */
struct Config {
  int num_sms{20};
  int num_max_nvl_chunked_send_tokens{6};
  int num_max_nvl_chunked_recv_tokens{256};
  int num_max_rdma_chunked_send_tokens{6};
  int num_max_rdma_chunked_recv_tokens{128};

  Config() = default;
  Config(
      int num_sms,
      int num_max_nvl_chunked_send_tokens,
      int num_max_nvl_chunked_recv_tokens,
      int num_max_rdma_chunked_send_tokens = 6,
      int num_max_rdma_chunked_recv_tokens = 128)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {}

  /** NVLink staging-buffer size hint for `Buffer.__init__` `num_nvl_bytes`.
   *
   *  Mirrors the exact per-rank NVL layout the dispatch kernel slices out of
   *  this region (intranode/kernels/Dispatch.cu): an R×R rank-prefix region,
   *  then per (channel, nvl-rank) 4 ring-metadata ints and
   *  `num_max_nvl_chunked_recv_tokens` slots of payload + src-idx + topk-idx +
   *  topk-weights + scales. `num_channels = num_sms/2`: dropping it (as the
   *  prior hint did) undersizes by ~10-32x and corrupts memory. Worst-case
   *  topk/scales (128) give headroom; the Buffer.cc dispatch/combine host
   *  guards re-check the exact size with the real topk/scales.
   */
  std::size_t getNvlBufferSizeHint(std::size_t hidden_bytes, int num_ranks)
      const noexcept {
    constexpr std::size_t kNumMaxTopK = 128;
    constexpr std::size_t kNumMaxScales = 128;
    // NUM_MAX_NVL_PEERS (== 8) is declared after this struct, so use the
    // literal here (matches the original hint).
    const std::size_t num_nvl_ranks =
        static_cast<std::size_t>(num_ranks < 8 ? num_ranks : 8);
    const std::size_t num_channels = static_cast<std::size_t>(num_sms / 2);
    const std::size_t recv =
        static_cast<std::size_t>(num_max_nvl_chunked_recv_tokens);
    const std::size_t cr = num_channels * num_nvl_ranks;
    std::size_t n = num_nvl_ranks * num_nvl_ranks * sizeof(int);
    n += cr * 4UL * sizeof(int);
    n += cr * recv * hidden_bytes;
    n += cr * recv * sizeof(int);
    n += cr * recv * kNumMaxTopK * sizeof(std::int64_t);
    n += cr * recv * kNumMaxTopK * sizeof(float);
    n += cr * recv * kNumMaxScales * sizeof(float);
    n = ((n + 127UL) / 128UL) * 128UL; // NUM_BUFFER_ALIGNMENT_BYTES
    return n;
  }

  /** RDMA staging-buffer size hint for `Buffer.__init__` `num_rdma_bytes`. */
  std::size_t getRdmaBufferSizeHint() const noexcept {
    constexpr std::size_t kMaxHiddenBytes = 8UL * 1024UL * 2UL;
    return static_cast<std::size_t>(num_max_rdma_chunked_recv_tokens) *
        kMaxHiddenBytes * 8UL +
        (32UL * 1024UL * 1024UL);
  }
};

/**
 * NUM_WORKSPACE_BYTES — persistent scratch used by every kernel for atomic
 * counters and per-expert state.
 */
inline constexpr std::size_t NUM_WORKSPACE_BYTES = 32UL * 1024UL * 1024UL;

/**
 * NUM_MAX_NVL_PEERS — ≤8 NVL peers per node (typical 8-GPU MI300X / H100
 * NVL8 layout).
 */
inline constexpr int NUM_MAX_NVL_PEERS = 8;

/**
 * NUM_BUFFER_ALIGNMENT_BYTES — all buffer offsets are aligned to this.
 */
inline constexpr std::size_t NUM_BUFFER_ALIGNMENT_BYTES = 128;

} // namespace comms::prims::moe_ep
