// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>
#include <json/reader.h>
#include <json/value.h>
#include <torch/csrc/autograd/profiler_kineto.h> // @manual=//caffe2:torch-cpp-cpu
#include <filesystem>
#include <vector>
#include "comms/torchcomms/TorchComm.hpp"

constexpr int kTensorCount = 4;
constexpr at::ScalarType kTensorDtype = at::kFloat;

class ProfilerGuard {
 public:
  ProfilerGuard() {
    torch::autograd::profiler::ProfilerConfig cfg{
        torch::autograd::profiler::ProfilerState::KINETO,
        true,
        false,
        false,
        false,
        false};
    std::set<torch::autograd::profiler::ActivityType> activities{
        torch::autograd::profiler::ActivityType::CPU,
        torch::autograd::profiler::ActivityType::CUDA};
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

class ProfilerTest : public ::testing::Test {
 public:
  ProfilerTest() : ProfilerTest(c10::DeviceType::CUDA) {}
  explicit ProfilerTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Helper function declarations with parameters
  Json::Value readTraceFile(std::filesystem::path& trace_file);

  void sanityCheckProfilerMeta(
      Json::Value& json_value,
      std::map<std::string, std::vector<Json::Value>>& events);

  c10::intrusive_ptr<torch::comms::TorchWork> runAllCollectiveOperations();

 protected:
  void SetUp() override;

  void TearDown() override;

  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;
};
