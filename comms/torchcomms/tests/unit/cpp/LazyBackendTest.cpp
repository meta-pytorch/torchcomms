// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <comms/torchcomms/fake/TorchCommFake.hpp>
#include <comms/torchcomms/lazy/LazyBackend.hpp>

namespace torch::comms::test {

namespace {

class LazyTorchCommFake final : public TorchCommFake {
 public:
  std::shared_ptr<LazyTorchCommFake> createPairComm(
      int /* peerRank */,
      const std::string& /* name */) {
    return std::make_shared<LazyTorchCommFake>();
  }

  void setBootstrapStore(c10::intrusive_ptr<c10d::Store> /* store */) {}
};

} // namespace

TEST(LazyBackendTest, WindowOperationsFollowBackendLifecycle) {
  LazyBackend<LazyTorchCommFake> backend;

  EXPECT_FALSE(backend.supportsWindow());
  EXPECT_THROW(backend.new_window(), std::runtime_error);

  backend.init(at::kCPU, "lazy-window-test");
  EXPECT_TRUE(backend.supportsWindow());
  EXPECT_NE(backend.new_window(), nullptr);

  backend.finalize();
  EXPECT_FALSE(backend.supportsWindow());
  EXPECT_THROW(backend.new_window(), std::runtime_error);
}

} // namespace torch::comms::test
