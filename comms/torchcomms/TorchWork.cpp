// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchWork.hpp"

#include <c10/core/DeviceGuard.h> // @manual=//caffe2:c10

namespace torch::comms {

namespace {

class SelfOwningTorchWorkThread final : public TorchWorkThread {
 public:
  SelfOwningTorchWorkThread() = default;
  SelfOwningTorchWorkThread(const SelfOwningTorchWorkThread&) = delete;
  SelfOwningTorchWorkThread& operator=(const SelfOwningTorchWorkThread&) =
      delete;
  SelfOwningTorchWorkThread(SelfOwningTorchWorkThread&&) = delete;
  SelfOwningTorchWorkThread& operator=(SelfOwningTorchWorkThread&&) = delete;

  ~SelfOwningTorchWorkThread() override {
    if (!thread_.joinable()) {
      return;
    }
    if (thread_.get_id() == std::this_thread::get_id()) {
      thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
    } else {
      thread_.join();
    }
  }

  void start(
      c10::intrusive_ptr<SelfOwningTorchWorkThread> self,
      std::function<void()> fn) {
    std::promise<void> completion;
    future_ = completion.get_future();
    setStatus(WorkStatus::INPROGRESS);
    thread_ = std::thread([self = std::move(self),
                           fn = std::move(fn),
                           completion = std::move(completion)]() mutable {
      WorkStatus terminalStatus = WorkStatus::COMPLETED;
      std::exception_ptr operationError;
      try {
        fn();
      } catch (...) {
        terminalStatus = WorkStatus::ERROR;
        operationError = std::current_exception();
      }

      const bool transitioned = self->tryTransitionStatus(terminalStatus);
      if (operationError) {
        completion.set_exception(operationError);
      } else {
        completion.set_value();
      }
      if (transitioned) {
        try {
          self->dispatchStatusTransition(terminalStatus);
        } catch (const std::exception& error) {
          LOG(ERROR) << "[TC][TorchWorkThread] Completion hook failed: "
                     << error.what();
        } catch (...) {
          LOG(ERROR)
              << "[TC][TorchWorkThread] Completion hook failed with an unknown exception";
        }
      }
    });
  }

 private:
  std::thread thread_;
};

} // namespace

void TorchWork::markCompleted(
    c10::intrusive_ptr<c10::ivalue::Future> future_,
    std::vector<at::Tensor> outputTensors_) {
  TORCH_CHECK(
      outputTensors_.size() > 0, "Atleast one tensor should be present");
  // CUDA: resolve immediately. Future records a CUDA event on the current
  // stream via markCompleted(). Device guard ensures getCurrentStream()
  // returns the correct device's stream.
  const auto device = outputTensors_[0].device();
  c10::OptionalDeviceGuard guard(device);
  future_->markCompleted(c10::IValue(outputTensors_));
}

TorchWorkCompleted::TorchWorkCompleted() {
  setStatus(WorkStatus::COMPLETED);
}

void TorchWorkCompleted::wait() {
  runWaitPreHooks();
  runWaitPostHooks();
}

void TorchWorkCompleted::waitBlocking() {
  // Note: required for TorchComm::reconfigure to work with the dummy backend.
  // No need to checkStatus as the constructor sets the status to COMPLETED.
  return;
}

c10::intrusive_ptr<TorchWorkThread> TorchWorkThread::create(
    std::function<void()> fn) {
  auto work = c10::make_intrusive<SelfOwningTorchWorkThread>();
  work->start(work, std::move(fn));
  return c10::static_intrusive_pointer_cast<TorchWorkThread>(std::move(work));
}

TorchWorkThread::TorchWorkThread(std::function<void()> fn)
    : future_(std::async(std::launch::async, [this, fn = std::move(fn)]() {
        try {
          fn();
          setStatus(WorkStatus::COMPLETED);
        } catch (...) {
          setStatus(WorkStatus::ERROR);
          throw;
        }
      })) {}

void TorchWorkThread::wait() {
  runWaitPreHooks();

  if (!future_.valid()) {
    // already waited on
    runWaitPostHooks();
    return;
  }
  future_.get();
  runWaitPostHooks();
}

} // namespace torch::comms
