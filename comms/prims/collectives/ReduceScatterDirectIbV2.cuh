// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

inline constexpr int kDirectIbV2DeviceMaxRanks = 256;

// DirectIB v2 reduce-scatter: BF16 in, BF16 on the wire, FP32 out. Stochastic
// rounding moved into the FSDP copy-in (D114734625), so unlike v1 this kernel
// neither quantizes nor carries a seed.
struct DirectReduceScatterIbV2Args {
  int my_rank{0};
  int num_ranks{0};
  // Elements per rank shard -- the output size, not the input size.
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  P2pIbTransportDevice peers[kDirectIbV2DeviceMaxRanks]{};
  const __nv_bfloat16* input{nullptr};
  float* output{nullptr};
  // `input` registered with the multi-peer transport so the NIC reads the send
  // payload directly out of it. Must cover the whole input buffer; offsets are
  // taken relative to `input`.
  IbgdaLocalBuffer input_reg{};
};

void launch_direct_reduce_scatter_ib_v2(
    const DirectReduceScatterIbV2Args& args,
    int num_blocks,
    int block_threads,
    cudaStream_t stream,
    AbortDevice abortDevice);

} // namespace comms::prims
