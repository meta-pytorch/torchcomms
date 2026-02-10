// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class SplitTest : public ::testing::Test {
 public:
  SplitTest() : SplitTest(c10::DeviceType::CUDA) {}
  explicit SplitTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;

  void testContiguousGroup(
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  void testNonContiguousGroup(
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  void testDuplicateRanks();
  void testRankNotInGroup();
  void testMultiLevel();
  void testMultipleSplitsSameRanks();

  // Helper function declarations
  std::vector<std::vector<int>> createContigGroups(
      int num_ranks,
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  std::vector<std::vector<int>> createNonContigGroups(
      int num_ranks,
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  void verifyNonContigGroups(
      std::shared_ptr<torch::comms::TorchComm>& parent_comm,
      std::shared_ptr<torch::comms::TorchComm>& child_comm,
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  void verifyContigGroups(
      std::shared_ptr<torch::comms::TorchComm>& parent_comm,
      std::shared_ptr<torch::comms::TorchComm>& child_comm,
      int num_groups,
      const std::unordered_set<int>& empty_groups);
  void testCommunication(std::shared_ptr<torch::comms::TorchComm>& comm);
};
