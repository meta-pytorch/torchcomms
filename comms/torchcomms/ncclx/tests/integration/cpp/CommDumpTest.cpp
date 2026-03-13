// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CommDumpTest.hpp"
#include <folly/logging/xlog.h>

void CommDumpTest::SetUp() {
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();

  auto backend = torchcomm_->getBackendImpl();
  ncclx_comm_ =
      std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(backend);
  ASSERT_NE(ncclx_comm_, nullptr)
      << "Test requires NCCLX backend. Set TEST_BACKEND=ncclx";
}

void CommDumpTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}
