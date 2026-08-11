// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims::test {

void test_llpacket_geometry(uint32_t* errorCount_d);

void test_llpacket_addressing(void* p8_d, uint32_t* errorCount_d);

} // namespace comms::prims::test
