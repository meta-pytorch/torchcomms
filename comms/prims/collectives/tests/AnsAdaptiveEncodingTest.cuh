// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

namespace comms::prims::test {

// Single-block `AnsCompress<8>` drivers over sub-chunks of `chunkBytes`; sub-
// chunk k is staged at `staging + k * stagingStride`.

// Encodes `src` sub-chunk by sub-chunk. With `feedback`, sub-chunk k's backlog
// query reports `pending[k]`; with `!feedback` the handle is null. Records
// each staged size in `sizes[k]`.
void ans_adaptive_send(
    const char* src,
    char* staging,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride,
    const uint64_t* pending,
    bool feedback,
    std::size_t minPendingBytes,
    uint64_t* sizes);

// Decodes every staged sub-chunk into `dst`.
void ans_adaptive_recv(
    const char* staging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride);

// recv_forward(): decodes into `dst` and forwards each staged sub-chunk to
// the same offset in `fwdStaging`.
void ans_adaptive_forward(
    const char* staging,
    char* fwdStaging,
    char* dst,
    std::size_t nbytes,
    std::size_t chunkBytes,
    std::size_t stagingStride);

} // namespace comms::prims::test
