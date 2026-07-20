// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Explicit instantiations of ring_reduce_scatter_kernel for uint32_t.
// One translation unit per datatype keeps each nvcc compile action small and
// parallel (see RingReduceScatterKernel.cuh).

#include "comms/prims/collectives/RingReduceScatterKernel.cuh"

namespace comms::prims {

INSTANTIATE_RING_REDUCE_SCATTER_FOR_TYPE(uint32_t);

} // namespace comms::prims
