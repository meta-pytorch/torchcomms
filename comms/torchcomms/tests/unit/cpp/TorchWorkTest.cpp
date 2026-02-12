// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <c10/util/intrusive_ptr.h>
#include "comms/torchcomms/TorchWork.hpp"

namespace torch::comms::test {

class TestWork : public TorchWork {
 public:
  explicit TestWork(bool* destroyed) : destroyed_(destroyed) {}
  ~TestWork() override {
    if (destroyed_) {
      *destroyed_ = true;
    }
  }

  void wait() override {}

  // expose for testing
  using TorchWork::setCallback;

 private:
  bool* destroyed_;
};

// validate that a work object with a callback capturing a weak_intrusive_ptr
// back to itself is properly destroyed when the last strong reference is
// dropped. this is the cycle that release_resources() is designed to break...
TEST(TorchWorkTest, WorkDestroyedAfterCallbackWithWeakRef) {
  bool destroyed = false;
  {
    c10::intrusive_ptr<TestWork> work =
        c10::make_intrusive<TestWork>(&destroyed);
    // postHook pattern: callback captures a weak ref to work.
    c10::weak_intrusive_ptr<TestWork> weak_work(work);
    // capture creates the prevent-destruction cycle...
    work->setCallback(
        [weak_work = std::move(weak_work)]() { (void)weak_work; });
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

TEST(TorchWorkTest, WorkDestroyedWithoutCallback) {
  bool destroyed = false;
  {
    c10::intrusive_ptr<TestWork> work =
        c10::make_intrusive<TestWork>(&destroyed);
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

} // namespace torch::comms::test
