// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <comms/torchcomms/TorchCommTypes.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>

namespace torch::comms {

namespace detail {

template <typename T>
T getHint(
    const std::unordered_map<std::string, std::string>& hints,
    std::string_view key,
    const T& default_value);

} // namespace detail

template <typename Derived>
struct OptionsBase {
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout{kNoTimeout};

  template <typename T>
  T getHint(std::string_view key, const T& default_value) const {
    return detail::getHint(hints, key, default_value);
  }
};

// Options classes for collective operations
struct SendOptions : OptionsBase<SendOptions> {
  int tag{0};
};

struct RecvOptions : OptionsBase<RecvOptions> {
  int tag{0};
};

struct BatchP2POptions : OptionsBase<BatchP2POptions> {};
struct BroadcastOptions : OptionsBase<BroadcastOptions> {};
struct AllReduceOptions : OptionsBase<AllReduceOptions> {};
struct ReduceOptions : OptionsBase<ReduceOptions> {};
struct AllGatherOptions : OptionsBase<AllGatherOptions> {};
struct AllGatherSingleOptions : OptionsBase<AllGatherSingleOptions> {};
struct ReduceScatterOptions : OptionsBase<ReduceScatterOptions> {};
struct ReduceScatterSingleOptions : OptionsBase<ReduceScatterSingleOptions> {};
struct AllToAllOptions : OptionsBase<AllToAllOptions> {};
struct AllToAllSingleOptions : OptionsBase<AllToAllSingleOptions> {};
struct AllToAllvSingleOptions : OptionsBase<AllToAllvSingleOptions> {};
struct BarrierOptions : OptionsBase<BarrierOptions> {};
struct ScatterOptions : OptionsBase<ScatterOptions> {};
struct GatherOptions : OptionsBase<GatherOptions> {};
struct GatherSingleOptions : OptionsBase<GatherSingleOptions> {};

class CommOptions {
 public:
  bool abort_process_on_timeout_or_error{true};
  std::chrono::milliseconds timeout{kDefaultTimeout};
  bool is_high_priority_stream{false};
  c10::intrusive_ptr<c10d::Store> store{nullptr};
  /**
   * If true, enables reconfigure() for fault tolerance.
   * With reconfigure enabled, the communicator is not initialized until
   * reconfigure() is called. Default is false.
   */
  bool enable_reconfigure{false};
  std::unordered_map<std::string, std::string> hints;

 public:
  CommOptions();

  bool operator==(const CommOptions& other) const;

  // Look up a hint by key and convert to the requested type.
  // Returns default_value if the key is not present.
  template <typename T>
  T getHint(std::string_view key, const T& default_value) const;
};

class PutOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  PutOptions() : timeout(kNoTimeout) {}
};

class SignalOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  SignalOptions() : timeout(kNoTimeout) {}
};

class WaitSignalOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  WaitSignalOptions() : timeout(kNoTimeout) {}
};

struct AllGatherPInitOptions : OptionsBase<AllGatherPInitOptions> {};
struct AllGatherPExecOptions : OptionsBase<AllGatherPExecOptions> {};

} // namespace torch::comms
