// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace comms::prims::link_ep {

/**
 * NUM_MAX_NVL_PEERS — ≤8 NVL peers per node (typical 8-GPU MI300X / H100
 * NVL8 layout).
 */
inline constexpr int NUM_MAX_NVL_PEERS = 8;

/**
 * NUM_BUFFER_ALIGNMENT_BYTES — all buffer offsets are aligned to this.
 */
inline constexpr std::size_t NUM_BUFFER_ALIGNMENT_BYTES = 128;

/**
 * Config — tuning knobs for the dispatch / combine kernels.
 *
 * The Python `comms.prims.collectives.link_ep.Config(...)` ctor signature is:
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

  /** RDMA staging-buffer size hint for `Buffer.__init__` `num_rdma_bytes`.
   *
   *  Pure
   *  intranode groups (num_ranks <= NUM_MAX_NVL_PEERS) need no RDMA staging
   *  and return 0. For multi-node groups, sizes the per-channel ×
   *  per-RDMA-rank send+recv stripes (×2). NOTE: the internode (>1 node) path
   *  is not exercised by single-node runs — validate this sizing against the
   *  BNXT/IBGDA transport before relying on it cross-node.
   *
   *  @param hidden_bytes per-token hidden size in bytes
   *  @param num_ranks    total ranks in the expert-parallel group
   *  @return suggested `num_rdma_bytes`, padded to NUM_BUFFER_ALIGNMENT_BYTES
   */
  std::size_t getRdmaBufferSizeHint(std::size_t hidden_bytes, int num_ranks)
      const noexcept {
    if (num_ranks <= NUM_MAX_NVL_PEERS) {
      return 0;
    }
    constexpr int kNumMaxTopK = 128;
    constexpr int kNumMaxScales = 128;
    const std::size_t num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const std::size_t num_channels = static_cast<std::size_t>(num_sms) / 2;

    std::size_t num_bytes = 0;
    num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) *
        2 * sizeof(int);
    num_bytes += num_channels * num_rdma_ranks *
        num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
    num_bytes += num_channels * num_rdma_ranks *
        num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(std::int64_t) *
        2;
    num_bytes += num_channels * num_rdma_ranks *
        num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks *
        num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
    num_bytes = ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES - 1) /
                 NUM_BUFFER_ALIGNMENT_BYTES) *
        NUM_BUFFER_ALIGNMENT_BYTES;
    return num_bytes;
  }
};

/**
 * NUM_WORKSPACE_BYTES — persistent scratch used by every kernel for atomic
 * counters and per-expert state.
 */
inline constexpr std::size_t NUM_WORKSPACE_BYTES = 32UL * 1024UL * 1024UL;

} // namespace comms::prims::link_ep
