// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvDynamicTest.hpp"

#include <gtest/gtest.h>

#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllToAllvDynamicTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllToAllvDynamicTest::SetUp() {
  // NCCLX alltoallvDynamic requires NCCL_CTRAN_ENABLE=1
  setenv("NCCL_CTRAN_ENABLE", "1", 1);
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();

  // Get the backend and cast to TorchCommNCCLX for NCCLX-specific APIs
  auto backend = torchcomm_->unsafeGetBackend();
  ncclx_comm_ =
      std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(backend);
  ASSERT_NE(ncclx_comm_, nullptr)
      << "Test requires NCCLX backend. Set TEST_BACKEND=ncclx";
}

void AllToAllvDynamicTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}
