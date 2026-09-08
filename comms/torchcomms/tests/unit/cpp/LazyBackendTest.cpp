// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <mutex>

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

  using TorchCommFake::abort;
  void abort(const AbortInfo& info) override {
    if (!isAbortSupported()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(abortInfoMutex_);
      abortInfo_ = info;
    }
    TorchCommFake::abort();
  }

  std::optional<AbortInfo> getAbortInfo() const override {
    if (!isAborted()) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(abortInfoMutex_);
    return abortInfo_.value_or(AbortInfo{});
  }

 private:
  mutable std::mutex abortInfoMutex_;
  std::optional<AbortInfo> abortInfo_;
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

TEST(LazyBackendTest, PreservesContextualAbortInfo) {
  LazyBackend<LazyTorchCommFake> backend;
  backend.init(at::kCPU, "lazy-abort-test");
  backend.getPrimary()->enableAbort();

  const AbortInfo expected{
      .reason = AbortReason::INTERNAL_ERROR,
      .context = "watchdog detected a stalled operation",
  };
  backend.abort(expected);

  EXPECT_TRUE(backend.isAborted());
  EXPECT_EQ(backend.getAbortInfo(), expected);
}

TEST(LazyBackendTest, TestFakeSynthesizesInfoForPlainAbort) {
  LazyTorchCommFake backend;
  backend.enableAbort();

  backend.abort();

  EXPECT_TRUE(backend.isAborted());
  EXPECT_EQ(backend.getAbortInfo(), AbortInfo{});
}

} // namespace torch::comms::test
