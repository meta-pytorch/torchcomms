// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/CopyKernelBench.cuh"

namespace comms::prims::benchmark {

__global__ void copyKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);

  const std::size_t bytes_per_group =
      (nBytes + group.total_groups - 1) / group.total_groups;
  const std::size_t start_offset =
      static_cast<std::size_t>(group.group_id) * bytes_per_group;

  const std::size_t end_offset = (start_offset + bytes_per_group < nBytes)
      ? start_offset + bytes_per_group
      : nBytes;
  const std::size_t chunk_bytes =
      (start_offset < nBytes) ? end_offset - start_offset : 0;

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(
        dst + start_offset, src + start_offset, chunk_bytes, group);
  }
}

__global__ void dualDstCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);

  const std::size_t bytes_per_group =
      (nBytes + group.total_groups - 1) / group.total_groups;
  const std::size_t start_offset =
      static_cast<std::size_t>(group.group_id) * bytes_per_group;

  const std::size_t end_offset = (start_offset + bytes_per_group < nBytes)
      ? start_offset + bytes_per_group
      : nBytes;
  const std::size_t chunk_bytes =
      (start_offset < nBytes) ? end_offset - start_offset : 0;

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(
        dst1 + start_offset,
        dst2 + start_offset,
        src + start_offset,
        chunk_bytes,
        group);
  }
}

// Two-pass copy in a single kernel: read src twice, write to dst1 then dst2
__global__ void twoPassCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);

  const std::size_t bytes_per_group =
      (nBytes + group.total_groups - 1) / group.total_groups;
  const std::size_t start_offset =
      static_cast<std::size_t>(group.group_id) * bytes_per_group;

  const std::size_t end_offset = (start_offset + bytes_per_group < nBytes)
      ? start_offset + bytes_per_group
      : nBytes;
  const std::size_t chunk_bytes =
      (start_offset < nBytes) ? end_offset - start_offset : 0;

  for (int run = 0; run < nRuns; ++run) {
    // Pass 1: src -> local dst1
    memcpy_vectorized(
        dst1 + start_offset, src + start_offset, chunk_bytes, group);
    // Pass 2: src -> remote dst2
    memcpy_vectorized(
        dst2 + start_offset, src + start_offset, chunk_bytes, group);
  }
}

} // namespace comms::prims::benchmark
