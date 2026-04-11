// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>

#include "comms/torchcomms/transport/StagedRdmaTransport.h"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace torch::comms;

TEST(StagedRdmaTransportTest, ConstructAndDestroy) {
  // Minimal construction — no IB resources, no EventBase.
  StagedRdmaServerTransport server(0);
  StagedRdmaClientTransport client(0);
  EXPECT_EQ(server.stagingBufSize(), 64 * 1024 * 1024);
  EXPECT_EQ(client.stagingBufSize(), 64 * 1024 * 1024);
}

TEST(StagedRdmaTransportTest, ConstructWithConfig) {
  StagedTransferConfig config;
  config.stagingBufSize = 1024 * 1024;
  config.chunkTimeout = std::chrono::milliseconds{5000};

  StagedRdmaServerTransport server(0, nullptr, config);
  StagedRdmaClientTransport client(0, nullptr, config);
  EXPECT_EQ(server.stagingBufSize(), 1024 * 1024);
  EXPECT_EQ(client.stagingBufSize(), 1024 * 1024);
}

TEST(StagedRdmaTransportTest, ConstructWithEventBase) {
  folly::ScopedEventBaseThread evbThread("test-evb");
  StagedRdmaServerTransport server(0, evbThread.getEventBase());
  StagedRdmaClientTransport client(0, evbThread.getEventBase());
  EXPECT_EQ(server.stagingBufSize(), 64 * 1024 * 1024);
  EXPECT_EQ(client.stagingBufSize(), 64 * 1024 * 1024);
}
