// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace comms::prims {
// Forward declaration only: keep the DOCA/CUDA device headers out of host
// translation units (LLIbgdaSendRecvTest.cc) that just pass this as a pointer.
class P2pIbgdaTransportDevice;
} // namespace comms::prims

namespace comms::prims::test {

// Fill `buffer` with the deterministic pattern verifyPattern() checks:
//   buffer[i] = baseValue + (i % 256)
void fillPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize);

// Count bytes in `buffer` that do NOT match fillPattern(baseValue). The
// mismatch count is atomically accumulated into the device int `errorCount`.
void verifyPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int* errorCount,
    int numBlocks,
    int blockSize);

// Launch one Proto=LL send (send=true) or recv (send=false) of `nbytes` PAYLOAD
// bytes over `transport`'s channel via detail::send/recv<..., LL>. Throws
// std::runtime_error on a launch failure. Caller synchronizes the stream.
void launchLLSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize);

// Same as launchLLSendRecv but with Proto=Simple (baseline nccl-"simple"
// put+signal path). Identical geometry to the transport's built-in send/recv.
void launchSimpleSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize);

} // namespace comms::prims::test
