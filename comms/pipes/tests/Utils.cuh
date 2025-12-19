// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace comms::pipes::test {

// Host wrapper functions

/**
 * Fills a device buffer with a specified value.
 *
 * @param deviceBuffer Pointer to the device memory buffer to fill
 * @param value The integer value to fill the buffer with
 * @param numElements The number of elements in the buffer
 */
void fillBuffer(int* deviceBuffer, int value, size_t numElements);

/**
 * Verifies that all elements in a device buffer match the expected value.
 * Counts and stores the number of mismatches in deviceErrorCount.
 *
 * @param deviceBuffer Pointer to the device memory buffer to verify
 * @param expectedValue The expected value that all elements should have
 * @param numElements The number of elements to verify in the buffer
 * @param deviceErrorCount Pointer to device memory where the count of
 *                         mismatched elements will be stored
 */
void verifyBuffer(
    const int* deviceBuffer,
    int expectedValue,
    size_t numElements,
    int* deviceErrorCount);

} // namespace comms::pipes::test
