#pragma once

#include <functional>
#include <mutex>

namespace torch::comms {

class RemovableHandle {
 public:
  explicit RemovableHandle(std::function<void()>&& callback)
      : callback_(std::move(callback)) {}
  ~RemovableHandle() = default;

  RemovableHandle(const RemovableHandle&) = delete;
  RemovableHandle& operator=(const RemovableHandle&) = delete;
  RemovableHandle(RemovableHandle&&) = delete;
  RemovableHandle& operator=(RemovableHandle&&) = delete;

  void remove() {
    std::call_once(once_, [this]() noexcept {
      callback_();
      callback_ = nullptr;
    });
  }

 private:
  std::once_flag once_;
  std::function<void()> callback_;
};

} // namespace torch::comms
