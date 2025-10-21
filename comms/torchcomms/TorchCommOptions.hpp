// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <comms/torchcomms/TorchCommTypes.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Work.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <chrono>
#include <string>
#include <unordered_map>

namespace torch {
namespace comms {

// Options classes for collective operations
class SendOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  SendOptions() : timeout(kNoTimeout) {}
};

class RecvOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  RecvOptions() : timeout(kNoTimeout) {}
};

class BatchP2POptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BatchP2POptions() : timeout(kNoTimeout) {}
};

class BroadcastOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BroadcastOptions() : timeout(kNoTimeout) {}
};

class AllReduceOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllReduceOptions() : timeout(kNoTimeout) {}
};

class ReduceOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceOptions() : timeout(kNoTimeout) {}
};

class AllGatherOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherOptions() : timeout(kNoTimeout) {}
};

class AllGatherSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherSingleOptions() : timeout(kNoTimeout) {}
};

class ReduceScatterOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceScatterOptions() : timeout(kNoTimeout) {}
};

class ReduceScatterSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceScatterSingleOptions() : timeout(kNoTimeout) {}
};

class AllToAllOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllOptions() : timeout(kNoTimeout) {}
};

class AllToAllSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllSingleOptions() : timeout(kNoTimeout) {}
};

class AllToAllvSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllvSingleOptions() : timeout(kNoTimeout) {}
};

class BarrierOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BarrierOptions() : timeout(kNoTimeout) {}
};

class ScatterOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ScatterOptions() : timeout(kNoTimeout) {}
};

class GatherOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  GatherOptions() : timeout(kNoTimeout) {}
};

class CommOptions {
 public:
  bool abort_process_on_timeout_or_error{true};
  std::chrono::milliseconds timeout{kDefaultTimeout};
  bool high_priority_stream{false};
  c10::intrusive_ptr<c10d::Store> store{nullptr};
  std::unordered_map<std::string, std::string> hints;

 public:
  CommOptions();

  bool operator==(const CommOptions& other) const;
};

} // namespace comms
} // namespace torch
