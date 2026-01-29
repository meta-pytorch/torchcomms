// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test Main

#include <gtest/gtest.h>
#include "DeviceApiTest.hpp"
#include "comms/torchcomms/TorchComm.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
