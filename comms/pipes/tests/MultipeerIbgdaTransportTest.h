// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes {
// Forward declaration
class P2pIbgdaTransportDevice;
} // namespace comms::pipes

namespace comms::pipes::test {

/**
 * Test kernel: Send data via put_signal
 *
 * Sender fills local buffer with pattern and uses put_signal to transfer
 * data to remote peer, signaling completion.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param signalVal Signal value to send
 */
void testPutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for signal
 *
 * Receiver waits for the signal value to arrive, indicating data is ready.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param expectedSignal Signal value to wait for
 */
void testWaitSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multiple put_signal operations in sequence
 *
 * Performs multiple put_signal operations, each with a unique signal value.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param bytesPerPut Bytes per put operation
 * @param numPuts Number of put operations
 */
void testMultiplePutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Fill a device buffer with a pattern based on index
 *
 * Each byte is set to (baseValue + (index % 256))
 *
 * @param buffer Device buffer pointer
 * @param nbytes Number of bytes to fill
 * @param baseValue Base value for the pattern
 */
void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize);

/**
 * Verify a device buffer matches expected pattern
 *
 * Returns the count of mismatched bytes.
 *
 * @param buffer Device buffer pointer
 * @param nbytes Number of bytes to verify
 * @param expectedBaseValue Expected base value of pattern
 * @param errorCount Output: number of errors found (device pointer)
 */
void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
