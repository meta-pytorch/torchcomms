// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test Main
//
// NOTE: TEST_F macros are in DeviceApiTest.cpp (which is compiled with
// TORCHCOMMS_HAS_NCCL_DEVICE_API=1) to ensure consistent type resolution.
// This file only contains the main() entry point.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
