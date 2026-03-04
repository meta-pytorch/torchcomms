// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/CopyKernelBench.cuh"

namespace comms::pipes::benchmark {

struct ChunkPartition {
  std::size_t start_offset;
  std::size_t chunk_bytes;
};

__device__ __forceinline__ ChunkPartition
compute_chunk_partition(std::size_t nBytes, const ThreadGroup& group) {
  const std::size_t bytes_per_group =
      (nBytes + group.total_groups - 1) / group.total_groups;
  const std::size_t start_offset =
      static_cast<std::size_t>(group.group_id) * bytes_per_group;
  const std::size_t end_offset = (start_offset + bytes_per_group < nBytes)
      ? start_offset + bytes_per_group
      : nBytes;
  const std::size_t chunk_bytes =
      (start_offset < nBytes) ? end_offset - start_offset : 0;
  return {start_offset, chunk_bytes};
}

__global__ void copyKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  ChunkPartition chunk = compute_chunk_partition(nBytes, group);

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(
        dst + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
  }
}

__global__ void sequentialCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  ChunkPartition chunk = compute_chunk_partition(nBytes, group);

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(
        dst1 + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
    memcpy_vectorized(
        dst2 + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
  }
}

__global__ void dualDestCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  ChunkPartition chunk = compute_chunk_partition(nBytes, group);

  for (int run = 0; run < nRuns; ++run) {
    std::array<char*, 2> dsts = {
        {dst1 + chunk.start_offset, dst2 + chunk.start_offset}};
    memcpy_vectorized_multi_dest<2>(
        dsts, src + chunk.start_offset, chunk.chunk_bytes, group);
  }
}

__global__ void sequentialTriCopyKernel(
    char* dst1,
    char* dst2,
    char* dst3,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  ChunkPartition chunk = compute_chunk_partition(nBytes, group);

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(
        dst1 + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
    memcpy_vectorized(
        dst2 + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
    memcpy_vectorized(
        dst3 + chunk.start_offset,
        src + chunk.start_offset,
        chunk.chunk_bytes,
        group);
  }
}

__global__ void triDestCopyKernel(
    char* dst1,
    char* dst2,
    char* dst3,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  ChunkPartition chunk = compute_chunk_partition(nBytes, group);

  std::array<char*, 3> dsts = {
      {dst1 + chunk.start_offset,
       dst2 + chunk.start_offset,
       dst3 + chunk.start_offset}};

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized_multi_dest<3>(
        dsts, src + chunk.start_offset, chunk.chunk_bytes, group);
  }
}

} // namespace comms::pipes::benchmark
