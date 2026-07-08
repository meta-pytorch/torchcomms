// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/AllGatherLauncher.h"

#include <cuda_runtime.h>

#include <new>
#include <stdexcept>
#include <string>

#include "comms/prims/collectives/AllGatherDirect.cuh"
#include "comms/prims/core/Checks.h"
#include "comms/prims/core/TimeoutUtils.h"

// Keep the kernel definitions in this translation unit. Package builds link
// this launcher as a standalone shared object, and CUDA kernel symbols are
// hidden across that boundary.
#include "AllGatherDirect.cu"

namespace comms::prims {

namespace {

void validate_direct_nvl(int num_ranks) {
  if (num_ranks < 1 || num_ranks > kDirectNvlMaxRanks) {
    throw std::runtime_error(
        "Unsupported direct NVLink num_ranks=" + std::to_string(num_ranks) +
        " (supported: 1.." + std::to_string(kDirectNvlMaxRanks) + ")");
  }
}

Timeout make_launch_timeout(float timeout_ms) {
  Timeout timeout;
  if (timeout_ms > 0) {
    int device = 0;
    PIPES_CUDA_CHECK(cudaGetDevice(&device));
    timeout = makeTimeout(timeout_ms, device);
  }
  return timeout;
}

} // namespace

void launch_direct_allgather_nvl(const DirectAllgatherNvlLaunchParams& params) {
  validate_direct_nvl(params.num_ranks);

  DirectAllgatherNvlArgs args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.sendcount = params.sendcount;
  args.signaling_data_size = params.signaling_data_size;
  args.sendbuf = params.sendbuf;
  args.recvbuf = params.recvbuf;

  for (int peer = 0; peer < params.num_ranks; ++peer) {
    if (peer == params.my_rank) {
      continue;
    }
    new (&args.peers[peer]) P2pNvlTransportDevice(params.peers[peer]);
  }

  direct_allgather_nvl_kernel<512>
      <<<params.num_blocks, 512, 0, params.stream>>>(
          args, make_launch_timeout(params.timeout_ms));
  PIPES_CUDA_CHECK(cudaGetLastError());
}

void launch_hierarchical_allgather_fused(
    const HierarchicalAllgatherLaunchParams& params) {
  validate_direct_nvl(params.nvl_size);
  if (params.ib_size < 1 ||
      params.ib_size * params.nvl_size != params.num_ranks) {
    throw std::runtime_error("Invalid hierarchical allgather rank geometry");
  }

  HierarchicalAllgatherFusedArgs args{};
  args.ib_rank = params.ib_rank;
  args.ib_size = params.ib_size;
  args.nvl_rank = params.nvl_rank;
  args.nvl_size = params.nvl_size;
  args.sendcount = params.sendcount;
  args.ib_signaling_data_size = params.ib_signaling_data_size;
  args.nvl_signaling_data_size = params.nvl_signaling_data_size;
  args.sendbuf = params.sendbuf;
  args.recvbuf = params.recvbuf;
  args.ib_ring = params.ib_ring;
  for (int peer = 0; peer < params.nvl_size; ++peer) {
    if (peer == params.nvl_rank) {
      continue;
    }
    new (&args.nvl_peers[peer]) P2pNvlTransportDevice(params.nvl_peers[peer]);
  }

  hierarchical_allgather_fused_kernel<512>
      <<<params.ib_num_blocks, 512, 0, params.stream>>>(
          args, make_launch_timeout(params.timeout_ms));
  PIPES_CUDA_CHECK(cudaGetLastError());
}

void launch_hierarchical_allgather_overlap(
    const HierarchicalAllgatherOverlapLaunchParams& params) {
  validate_direct_nvl(params.nvl_size);
  if (params.ib_size < 1 ||
      params.ib_size * params.nvl_size != params.num_ranks) {
    throw std::runtime_error("Invalid hierarchical allgather rank geometry");
  }
  if (params.ib_num_blocks < 1 || params.nvl_num_blocks < 1) {
    throw std::runtime_error(
        "Hierarchical overlap allgather requires positive block counts");
  }
  if (params.chunk_bytes == 0) {
    throw std::runtime_error(
        "Hierarchical overlap allgather requires positive chunk_bytes");
  }
  if (params.ready_counters == nullptr) {
    throw std::runtime_error(
        "Hierarchical overlap allgather requires ready counters");
  }

  HierarchicalAllgatherOverlapArgs args{};
  args.num_ranks = params.num_ranks;
  args.ib_rank = params.ib_rank;
  args.ib_size = params.ib_size;
  args.nvl_rank = params.nvl_rank;
  args.nvl_size = params.nvl_size;
  args.ib_num_blocks = params.ib_num_blocks;
  args.nvl_num_blocks = params.nvl_num_blocks;
  args.sendcount = params.sendcount;
  args.ib_signaling_data_size = params.ib_signaling_data_size;
  args.nvl_signaling_data_size = params.nvl_signaling_data_size;
  args.chunk_bytes = params.chunk_bytes;
  args.ready_sequence = params.ready_sequence;
  args.ready_counters = params.ready_counters;
  args.sendbuf = params.sendbuf;
  args.recvbuf = params.recvbuf;
  args.ib_ring = params.ib_ring;
  args.use_direct = params.use_direct;
  args.use_finer_nvl_handoff = params.use_finer_nvl_handoff;
  // ib_peers is only populated/used on the direct/star path, which the host
  // gates to ib_size <= kHierarchicalAgMaxNodes. Guard the copy so a ring-mode
  // launch with ib_size > kHierarchicalAgMaxNodes does not read/write past the
  // fixed-size ib_peers arrays.
  if (params.use_direct) {
    for (int node = 0; node < params.ib_size; ++node) {
      args.ib_peers[node] = params.ib_peers[node];
    }
  }
  args.trace = params.trace;
  for (int peer = 0; peer < params.nvl_size; ++peer) {
    if (peer == params.nvl_rank) {
      continue;
    }
    new (&args.nvl_peers[peer]) P2pNvlTransportDevice(params.nvl_peers[peer]);
  }

  hierarchical_allgather_overlap_kernel<512>
      <<<params.ib_num_blocks + params.nvl_num_blocks, 512, 0, params.stream>>>(
          args, make_launch_timeout(params.timeout_ms));
  PIPES_CUDA_CHECK(cudaGetLastError());
}

} // namespace comms::prims
