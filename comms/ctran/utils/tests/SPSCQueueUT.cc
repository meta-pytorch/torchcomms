// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/utils/SPSCQueue.h"

using ctran::utils::SPSCQueue;
class SPSCQueueTest : public ::testing::Test {
 public:
  int cudaDev;
  SPSCQueueTest() = default;

 protected:
  void SetUp() override {}
};

namespace {
struct TestTask {
  TestTask() = default;
  int x;
  double y;
};
} // namespace

TEST_F(SPSCQueueTest, Basic) {
  auto taskQueue = std::make_unique<SPSCQueue<TestTask>>();
  constexpr int numTasks = 1000;

  std::thread producer([&]() {
    for (int i = 0; i < numTasks; i++) {
      std::unique_ptr<TestTask> t = std::make_unique<TestTask>();
      t->x = i;
      t->y = i * 1.0;
      taskQueue->enqueue(std::move(t));
    }
  });

  std::thread consumer([&]() {
    for (int i = 0; i < numTasks; i++) {
      auto task = taskQueue->dequeue();
      ASSERT_NE(task, nullptr);
      ASSERT_EQ(task->x, i);
      ASSERT_EQ(task->y, i * 1.0);
    }
  });

  producer.join();
  consumer.join();
}
