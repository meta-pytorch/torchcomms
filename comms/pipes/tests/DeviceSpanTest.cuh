// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

// Basic properties tests
void testBasicProperties(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d);

// Element access tests
void testElementAccess(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d);

// Iterator tests
void testIterator(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* sum_d,
    uint32_t* errorCount_d);

// Subspan tests
void testSubspan(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d);

// Factory function tests
void testMakeDeviceSpan(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d);

// Const conversion tests
void testConstConversion(
    uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d);

// Empty span tests
void testEmptySpan(uint32_t* errorCount_d);

// Mutable span write tests
void testMutableSpanWrite(
    uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d);

} // namespace comms::pipes::test
