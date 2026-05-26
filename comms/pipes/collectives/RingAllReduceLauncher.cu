// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/RingAllReduceLauncher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/collectives/RingAllgather.cuh"
#include "comms/pipes/collectives/RingReduceScatter.cuh"

namespace comms::pipes {

namespace {

template <int NumRings>
void launch_impl(const RingAllReduceLaunchParams& params, Timeout timeout) {
  const int my_rank = params.my_rank;
  const int num_ranks = params.num_ranks;

  if (num_ranks == 0) {
    throw std::runtime_error("num_ranks must be > 0");
  }

  const std::size_t chunk_elements = params.count / num_ranks;
  const std::size_t chunk_bytes = chunk_elements * sizeof(float);

  // Phase 1: ReduceScatter
  // Writes reduced shard to output[my_rank * chunk_elements]
  {
    RingReduceScatterArgs<NumRings, float> args{};
    args.my_rank = my_rank;
    args.num_ranks = num_ranks;
    args.chunk_elements = chunk_elements;
    args.signaling_data_size = params.signaling_data_size;
    args.input = params.input;
    args.output = params.output + my_rank * chunk_elements;

    for (int r = 0; r < NumRings; r++) {
      auto& src = params.rings[r];
      args.rings[r] = RingTopology{
          .prev_rank = src.prev_rank,
          .next_rank = src.next_rank,
          .prev = src.prev,
          .next = src.next,
      };
    }

    ring_reduce_scatter_kernel<NumRings, float, SumOp, 16384, 512>
        <<<params.num_blocks, 512, 0, params.stream>>>(args, timeout);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }
  }

  // Phase 2: AllGather
  // Reads reduced shard from output[my_rank * chunk_bytes], gathers into output
  {
    RingAllgatherArgs<NumRings> args{};
    args.my_rank = my_rank;
    args.num_ranks = num_ranks;
    args.sendcount = chunk_bytes;
    args.signaling_data_size = params.signaling_data_size;
    args.sendbuf =
        reinterpret_cast<const char*>(params.output) + my_rank * chunk_bytes;
    args.recvbuf = reinterpret_cast<char*>(params.output);

    for (int r = 0; r < NumRings; r++) {
      auto& src = params.rings[r];
      args.rings[r] = RingTopology{
          .prev_rank = src.prev_rank,
          .next_rank = src.next_rank,
          .prev = src.prev,
          .next = src.next,
      };
    }

    ring_allgather_kernel<NumRings, 512>
        <<<params.num_blocks, 512, 0, params.stream>>>(args, timeout);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }
  }
}

} // namespace

void launch_ring_allreduce(const RingAllReduceLaunchParams& params) {
  Timeout timeout;
  if (params.timeout_ms > 0) {
    int device = 0;
    cudaError_t cudaErr = cudaGetDevice(&device);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error(
          "Failed to get CUDA device: " +
          std::string(cudaGetErrorString(cudaErr)));
    }

    timeout = makeTimeout(params.timeout_ms, device);
  }

  switch (params.num_rings) {
    case 1:
      launch_impl<1>(params, timeout);
      break;
    case 2:
      launch_impl<2>(params, timeout);
      break;
    case 4:
      launch_impl<4>(params, timeout);
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_rings=" + std::to_string(params.num_rings) +
          " (supported: 1, 2, 4)");
  }
}

} // namespace comms::pipes
