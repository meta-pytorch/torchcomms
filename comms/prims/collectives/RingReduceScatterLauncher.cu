// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/RingReduceScatterLauncher.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/collectives/RingReduceScatter.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

template <int NumRings, typename T, typename AccumOp>
void launch_impl(const RingReduceScatterLaunchParams& params, Timeout timeout) {
  RingReduceScatterArgs<NumRings, T> args{
      .my_rank = params.my_rank,
      .num_ranks = params.num_ranks,
      .chunk_elements = params.chunk_elements,
      .signaling_data_size = params.signaling_data_size,
      .input = static_cast<const T*>(params.input),
      .output = static_cast<T*>(params.output),
  };

  for (int r = 0; r < NumRings; r++) {
    auto& src = params.rings[r];
    args.rings[r] = RingTopology{
        .prev_rank = src.prev_rank,
        .next_rank = src.next_rank,
        .prev = src.prev,
        .next = src.next,
    };
  }

  ring_reduce_scatter_kernel<NumRings, T, AccumOp, 16384, 512>
      <<<params.num_blocks, 512, 0, params.stream>>>(args, timeout);
}

template <int NumRings, typename T>
void launch_type(const RingReduceScatterLaunchParams& params, Timeout timeout) {
  switch (params.reduce_op) {
    case RingReduceScatterReduceOp::kSum:
      launch_impl<NumRings, T, SumOp>(params, timeout);
      break;
    case RingReduceScatterReduceOp::kMax:
      launch_impl<NumRings, T, MaxOp>(params, timeout);
      break;
    case RingReduceScatterReduceOp::kMin:
      launch_impl<NumRings, T, MinOp>(params, timeout);
      break;
    default:
      throw std::runtime_error("Unsupported ring reduce-scatter reduce op");
  }
}

template <int NumRings>
void launch_dispatch(
    const RingReduceScatterLaunchParams& params,
    Timeout timeout) {
  switch (params.data_type) {
    case RingReduceScatterDataType::kInt8:
      launch_type<NumRings, int8_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kUint8:
      launch_type<NumRings, uint8_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kInt32:
      launch_type<NumRings, int32_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kUint32:
      launch_type<NumRings, uint32_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kInt64:
      launch_type<NumRings, int64_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kUint64:
      launch_type<NumRings, uint64_t>(params, timeout);
      break;
    case RingReduceScatterDataType::kFloat32:
      launch_type<NumRings, float>(params, timeout);
      break;
    case RingReduceScatterDataType::kFloat64:
      launch_type<NumRings, double>(params, timeout);
      break;
    case RingReduceScatterDataType::kFloat16:
      launch_type<NumRings, __half>(params, timeout);
      break;
    case RingReduceScatterDataType::kBfloat16:
      launch_type<NumRings, __nv_bfloat16>(params, timeout);
      break;
    default:
      throw std::runtime_error("Unsupported ring reduce-scatter data type");
  }
}

} // namespace

void launch_ring_reduce_scatter(const RingReduceScatterLaunchParams& params) {
  Timeout timeout;
  if (params.timeout_ms > 0) {
    int device = 0;
    cudaGetDevice(&device);
    timeout = makeTimeout(params.timeout_ms, device);
  }

  switch (params.num_rings) {
    case 1:
      launch_dispatch<1>(params, timeout);
      break;
    case 2:
      launch_dispatch<2>(params, timeout);
      break;
    case 4:
      launch_dispatch<4>(params, timeout);
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_rings=" + std::to_string(params.num_rings) +
          " (supported: 1, 2, 4)");
  }
}

} // namespace comms::prims
