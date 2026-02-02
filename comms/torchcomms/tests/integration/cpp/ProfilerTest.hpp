// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <gtest/gtest.h>
#include <json/reader.h>
#include <json/value.h>
#include <torch/csrc/autograd/profiler_kineto.h> // @manual=//caffe2:torch-cpp-cpu
#include <filesystem>
#include <functional>
#include <map>
#include <vector>
#include "comms/torchcomms/TorchComm.hpp"

constexpr int kProfilerTestTensorCount = 4;
constexpr at::ScalarType kProfilerTestTensorDtype = at::kFloat;

// RAII guard for profiler setup/teardown
class ProfilerGuard {
 public:
  ProfilerGuard(
      const std::set<torch::autograd::profiler::ActivityType>& activities = {
          torch::autograd::profiler::ActivityType::CPU,
          torch::autograd::profiler::ActivityType::CUDA}) {
    torch::autograd::profiler::ProfilerConfig cfg{
        torch::autograd::profiler::ProfilerState::KINETO,
        true,
        false,
        false,
        false,
        false};

    torch::autograd::profiler::prepareProfiler(cfg, activities);
    torch::autograd::profiler::enableProfiler(cfg, activities);
  }

  // Disable copy and move semantics
  ProfilerGuard(ProfilerGuard&&) = delete;
  ProfilerGuard& operator=(ProfilerGuard&&) = delete;
  ProfilerGuard(const ProfilerGuard&) = delete;
  ProfilerGuard& operator=(const ProfilerGuard&) = delete;

  void setEnableTracingSaving(const std::string& trace_file) {
    trace_file_ = trace_file;
  }

  ~ProfilerGuard() {
    auto results = torch::autograd::profiler::disableProfiler();
    if (trace_file_.has_value()) {
      results->save(trace_file_.value());
      LOG(INFO) << "Saved profiler results to " << trace_file_.value();
    }
  }

 private:
  std::optional<std::string> trace_file_{std::nullopt};
};

// Validation function type: takes parsed events map and performs assertions
using ProfilerValidationFunc =
    std::function<void(const std::map<std::string, std::vector<Json::Value>>&)>;

// Backend-agnostic profiler test base class
class ProfilerTest : public ::testing::Test {
 public:
  ProfilerTest(
      std::string backend,
      c10::DeviceType device_type,
      ProfilerValidationFunc validation_func)
      : backend_(std::move(backend)),
        device_type_(device_type),
        validation_func_(std::move(validation_func)),
        rank_(0),
        num_ranks_(0) {}

  ~ProfilerTest() override = default;

  // Helper functions
  static Json::Value readTraceFile(const std::filesystem::path& trace_file);

  static void sanityCheckProfilerMeta(
      const Json::Value& json_value,
      std::map<std::string, std::vector<Json::Value>>& events,
      const std::string& pgName);

  c10::intrusive_ptr<torch::comms::TorchWork> runAllCollectiveOperations();

 protected:
  void SetUp() override;
  void TearDown() override;

  std::string backend_;
  c10::DeviceType device_type_;
  ProfilerValidationFunc validation_func_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
};
