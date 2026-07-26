// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "comms/torchcomms/BackendWrapper.hpp"

namespace torch::comms::test {

namespace {

class ControllableWork final : public TorchWork {
 public:
  ControllableWork() {
    setStatus(WorkStatus::INPROGRESS);
  }

  void wait() override {
    ++waitCount_;
  }

  bool supportsActivePolling() const override {
    return true;
  }

  WorkStatus pollStatus() override {
    ++pollCount_;
    return status();
  }

  void complete() {
    setStatus(WorkStatus::COMPLETED);
  }

  void finish(WorkStatus status) {
    setStatus(status);
  }

  int waitCount() const {
    return waitCount_.load();
  }

  int pollCount() const {
    return pollCount_.load();
  }

 private:
  std::atomic<int> waitCount_{0};
  std::atomic<int> pollCount_{0};
};

class PollingWork final : public TorchWork {
 public:
  PollingWork() {
    setStatus(WorkStatus::INPROGRESS);
  }

  void wait() override {
    ++waitCount_;
  }

  bool supportsActivePolling() const override {
    return true;
  }

  WorkStatus pollStatus() override {
    if (++pollCount_ == 2) {
      setStatus(WorkStatus::COMPLETED);
    }
    return status();
  }

  int waitCount() const {
    return waitCount_.load();
  }

  int pollCount() const {
    return pollCount_.load();
  }

 private:
  std::atomic<int> waitCount_{0};
  std::atomic<int> pollCount_{0};
};

class NonPollingWork final : public TorchWork {
 public:
  NonPollingWork() {
    setStatus(WorkStatus::INPROGRESS);
  }

  void wait() override {
    ++waitCount_;
  }

  int waitCount() const {
    return waitCount_.load();
  }

 private:
  std::atomic<int> waitCount_{0};
};

class ThrowingWaitWork final : public TorchWork {
 public:
  ThrowingWaitWork() {
    setStatus(WorkStatus::INPROGRESS);
  }

  void wait() override {
    ++waitCount_;
    throw std::runtime_error("wait failed");
  }

  int waitCount() const {
    return waitCount_.load();
  }

 private:
  std::atomic<int> waitCount_{0};
};

class ThrowingPollingWork final : public TorchWork {
 public:
  ThrowingPollingWork() {
    setStatus(WorkStatus::INPROGRESS);
  }

  void wait() override {}

  bool supportsActivePolling() const override {
    return true;
  }

  [[noreturn]] WorkStatus pollStatus() override {
    throw std::runtime_error("poll failed");
  }
};

class TerminalWork final : public TorchWork {
 public:
  explicit TerminalWork(WorkStatus status) {
    setStatus(status);
  }

  void wait() override {
    ++waitCount_;
  }

  int waitCount() const {
    return waitCount_;
  }

  int markCompletedCount() const {
    return markCompletedCount_;
  }

 protected:
  void markCompleted(
      c10::intrusive_ptr<c10::ivalue::Future>,
      std::vector<at::Tensor>) override {
    ++markCompletedCount_;
  }

 private:
  int waitCount_{0};
  int markCompletedCount_{0};
};

at::Tensor makeFakeCudaTensor() {
  static float data = 0.0;
  auto tensorImpl =
      c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
          at::Storage(
              at::Storage::use_byte_size_t(),
              sizeof(data),
              at::DataPtr(&data, at::Device(at::DeviceType::CUDA, 0)),
              nullptr,
              false),
          c10::DispatchKey::CUDA,
          caffe2::TypeMeta::Make<float>());
  tensorImpl->set_sizes_contiguous({1});
  return at::Tensor(std::move(tensorImpl));
}

} // namespace

TEST(BackendWrapperTest, FiniteWaitTimeoutDoesNotPoisonWork) {
  auto nativeWork = c10::make_intrusive<ControllableWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  EXPECT_THROW(work->wait(std::chrono::milliseconds(1)), c10::Error);
  EXPECT_FALSE(work->isCompleted());
  EXPECT_EQ(nativeWork->waitCount(), 0);
  EXPECT_GT(nativeWork->pollCount(), 0);

  nativeWork->complete();

  EXPECT_TRUE(work->wait(std::chrono::milliseconds(100)));
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(nativeWork->waitCount(), 1);
}

TEST(BackendWrapperTest, FiniteWaitActivelyPollsNativeCompletion) {
  auto nativeWork = c10::make_intrusive<PollingWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  EXPECT_TRUE(work->wait(std::chrono::milliseconds(100)));
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(nativeWork->pollCount(), 2);
  EXPECT_EQ(nativeWork->waitCount(), 1);
}

TEST(BackendWrapperTest, IsCompletedActivelyPollsNativeCompletion) {
  auto nativeWork = c10::make_intrusive<PollingWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  EXPECT_FALSE(work->isCompleted());
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(nativeWork->pollCount(), 2);
  EXPECT_EQ(nativeWork->waitCount(), 0);
}

TEST(BackendWrapperTest, WaitFailureTerminalizesWorkAndFutures) {
  auto nativeWork = c10::make_intrusive<ThrowingWaitWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);
  auto future = work->getFuture();
  auto resultFuture = work->getFutureResult();

  EXPECT_THROW(work->wait(kNoTimeout), std::runtime_error);
  try {
    work->wait(kNoTimeout);
    FAIL() << "Expected the latched wait failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "wait failed");
  }

  EXPECT_EQ(nativeWork->status(), TorchWork::WorkStatus::ERROR);
  EXPECT_EQ(nativeWork->waitCount(), 1);
  EXPECT_TRUE(work->isCompleted());
  EXPECT_FALSE(work->isSuccess());
  EXPECT_TRUE(future->completed());
  EXPECT_TRUE(future->hasError());
  ASSERT_TRUE(resultFuture->completed());
  EXPECT_EQ(
      resultFuture->value().toInt(),
      static_cast<std::int64_t>(
          static_cast<std::uint8_t>(c10d::WorkResult::COMM_ERROR)));
  ASSERT_NE(work->exception(), nullptr);
  try {
    std::rethrow_exception(work->exception());
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "wait failed");
  }
}

TEST(BackendWrapperTest, PollFailureTerminalizesWorkAndFutures) {
  auto nativeWork = c10::make_intrusive<ThrowingPollingWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);
  auto future = work->getFuture();
  auto resultFuture = work->getFutureResult();

  EXPECT_TRUE(work->isCompleted());

  EXPECT_EQ(nativeWork->status(), TorchWork::WorkStatus::ERROR);
  EXPECT_FALSE(work->isSuccess());
  EXPECT_TRUE(future->completed());
  EXPECT_TRUE(future->hasError());
  ASSERT_TRUE(resultFuture->completed());
  EXPECT_EQ(
      resultFuture->value().toInt(),
      static_cast<std::int64_t>(
          static_cast<std::uint8_t>(c10d::WorkResult::COMM_ERROR)));
  ASSERT_NE(work->exception(), nullptr);
  try {
    std::rethrow_exception(work->exception());
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "poll failed");
  }
}

TEST(BackendWrapperTest, FiniteWaitRejectsWorkWithoutActivePolling) {
  auto nativeWork = c10::make_intrusive<NonPollingWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  EXPECT_THROW(work->wait(std::chrono::milliseconds(1)), c10::Error);
  EXPECT_FALSE(work->isCompleted());
  EXPECT_EQ(nativeWork->waitCount(), 0);
}

TEST(BackendWrapperTest, FiniteWaitAcceptsTerminalWorkWithoutActivePolling) {
  auto nativeWork =
      c10::make_intrusive<TerminalWork>(TorchWork::WorkStatus::COMPLETED);
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  EXPECT_TRUE(work->wait(std::chrono::milliseconds(1)));
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(work->isSuccess());
}

TEST(BackendWrapperTest, WaitPreservesAlreadyTerminalNativeFailure) {
  const std::vector<std::pair<TorchWork::WorkStatus, const char*>> cases{
      {TorchWork::WorkStatus::ERROR, "TorchComms operation failed"},
      {TorchWork::WorkStatus::TIMEDOUT, "TorchComms operation timed out"}};

  for (const auto& [status, expectedMessage] : cases) {
    auto nativeWork = c10::make_intrusive<TerminalWork>(status);
    auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

    EXPECT_THROW(work->wait(kNoTimeout), std::runtime_error);

    EXPECT_EQ(nativeWork->status(), status);
    EXPECT_EQ(nativeWork->waitCount(), 0);
    EXPECT_TRUE(work->isCompleted());
    EXPECT_FALSE(work->isSuccess());
    ASSERT_NE(work->exception(), nullptr);
    try {
      std::rethrow_exception(work->exception());
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), expectedMessage);
    }
  }
}

TEST(BackendWrapperTest, ExposedFutureCannotCompleteWork) {
  auto nativeWork = c10::make_intrusive<ControllableWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);
  auto exposedFuture = work->getFuture();
  exposedFuture->markCompleted(c10::IValue(std::vector<at::Tensor>{}));

  EXPECT_TRUE(exposedFuture->completed());
  EXPECT_FALSE(work->isCompleted());
  EXPECT_THROW(work->wait(std::chrono::milliseconds(1)), c10::Error);

  nativeWork->complete();

  EXPECT_TRUE(work->wait(std::chrono::milliseconds(100)));
  EXPECT_TRUE(work->isCompleted());
}

TEST(BackendWrapperTest, BlockCurrentStreamUsesNativeWait) {
  auto nativeWork = c10::make_intrusive<ControllableWork>();
  auto work = c10::make_intrusive<WorkWrapper>(nativeWork);

  work->blockCurrentStream();

  EXPECT_EQ(nativeWork->waitCount(), 1);
  EXPECT_FALSE(work->isCompleted());
}

TEST(BackendWrapperTest, GpuFutureReportsAlreadyTerminalFailure) {
  for (const auto status :
       {TorchWork::WorkStatus::ERROR, TorchWork::WorkStatus::TIMEDOUT}) {
    auto nativeWork = c10::make_intrusive<TerminalWork>(status);
    auto work = c10::make_intrusive<WorkWrapper>(
        nativeWork, std::vector<at::Tensor>{makeFakeCudaTensor()});

    EXPECT_TRUE(work->getFuture()->completed());
    EXPECT_TRUE(work->getFuture()->hasError());
    EXPECT_EQ(nativeWork->markCompletedCount(), 0);
  }
}

TEST(BackendWrapperTest, FutureResultTracksDeferredTerminalStatus) {
  const std::vector<std::pair<TorchWork::WorkStatus, c10d::WorkResult>> cases{
      {TorchWork::WorkStatus::COMPLETED, c10d::WorkResult::SUCCESS},
      {TorchWork::WorkStatus::ERROR, c10d::WorkResult::COMM_ERROR},
      {TorchWork::WorkStatus::TIMEDOUT, c10d::WorkResult::TIMEOUT}};

  for (const auto& [status, expected] : cases) {
    auto nativeWork = c10::make_intrusive<ControllableWork>();
    auto work = c10::make_intrusive<WorkWrapper>(nativeWork);
    auto result = work->getFutureResult();
    ASSERT_FALSE(result->completed());

    nativeWork->finish(status);

    ASSERT_TRUE(result->completed());
    EXPECT_EQ(
        result->value().toInt(),
        static_cast<std::int64_t>(static_cast<std::uint8_t>(expected)));
  }
}

TEST(BackendWrapperTest, ThreadWorkCallbackCanWaitAndReleaseLastOwner) {
  auto release = std::make_shared<std::promise<void>>();
  auto gate = release->get_future().share();
  auto nativeWork = TorchWorkThread::create([gate]() { gate.wait(); });
  c10::weak_intrusive_ptr<TorchWorkThread> weakWork(nativeWork);
  auto owner = std::make_shared<c10::intrusive_ptr<WorkWrapper>>(
      c10::make_intrusive<WorkWrapper>(nativeWork));
  auto callback = std::make_shared<std::promise<void>>();
  auto callbackResult = callback->get_future();
  (*owner)->getFuture()->addCallback([owner, callback](c10::ivalue::Future&) {
    try {
      (*owner)->wait(std::chrono::seconds(5));
      owner->reset();
      callback->set_value();
    } catch (...) {
      callback->set_exception(std::current_exception());
    }
  });
  nativeWork.reset();

  release->set_value();
  ASSERT_EQ(
      callbackResult.wait_for(std::chrono::seconds(5)),
      std::future_status::ready);
  EXPECT_NO_THROW(callbackResult.get());

  for (int attempt = 0; attempt < 500 && weakWork.lock(); ++attempt) {
    // Work destruction has no notification channel in this ownership test.
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(weakWork.lock());
}

TEST(BackendWrapperTest, ThreadWorkLegacyConstructorRemainsSupported) {
  auto work = c10::make_intrusive<TorchWorkThread>([]() {});

  EXPECT_NO_THROW(work->wait());
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

} // namespace torch::comms::test
