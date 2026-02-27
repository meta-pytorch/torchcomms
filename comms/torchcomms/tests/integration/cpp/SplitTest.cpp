// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "SplitTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> SplitTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void SplitTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_type_ = wrapper_->getDevice().type();
}

void SplitTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Helper function to test communication within a communicator
void SplitTest::testCommunication(
    std::shared_ptr<torch::comms::TorchComm>& comm) {
  // Skip if communicator is null
  if (!comm) {
    return;
  }

  // Get rank and size from the communicator
  int rank = comm->getRank();
  int size = comm->getSize();

  auto options = at::TensorOptions().dtype(at::kFloat).device(device_type_);
  auto input = at::ones({10}, options) * static_cast<float>(rank + 1);

  // For ranks in groups, test all_reduce
  comm->all_reduce(input, torch::comms::ReduceOp::SUM, false);

  // Verify the result using verifyTensorEquality with integer input
  verifyTensorEquality(input.cpu(), size * (size + 1) / 2);
}

// Helper function to create contiguous groups with even distribution
std::vector<std::vector<int>> SplitTest::createContigGroups(
    int num_ranks,
    int num_groups,
    const std::unordered_set<int>& empty_groups) {
  std::vector<std::vector<int>> rank_groups(num_groups);

  // Calculate base size for each group
  int base_size = num_ranks / num_groups;
  // Calculate how many groups get an extra rank
  int remainder = num_ranks % num_groups;

  int rank_index = 0;

  // Create each group
  for (int group = 0; group < num_groups; group++) {
    // Calculate size for this group
    int group_size = base_size;
    // Add extra rank to last group
    if (group == num_groups - 1) {
      group_size += remainder;
    }

    // If this group should be empty, skip the ranks that would have been in it
    if (empty_groups.find(group) != empty_groups.end()) {
      rank_index += group_size; // Skip these ranks
      continue;
    }

    // Add ranks to this group
    for (int i = 0; i < group_size && rank_index < num_ranks; i++) {
      rank_groups[group].push_back(rank_index++);
    }
  }

  return rank_groups;
}

// Helper function to create non-contiguous groups with round-robin distribution
std::vector<std::vector<int>> SplitTest::createNonContigGroups(
    int num_ranks,
    int num_groups,
    const std::unordered_set<int>& empty_groups) {
  std::vector<std::vector<int>> rank_groups(num_groups);

  // Distribute ranks in round-robin manner
  for (int rank = 0; rank < num_ranks; rank++) {
    int group = rank % num_groups;

    // Skip empty groups
    if (empty_groups.find(group) == empty_groups.end()) {
      rank_groups[group].push_back(rank);
    }
  }

  return rank_groups;
}

// Helper function to verify contiguous groups
void SplitTest::verifyContigGroups(
    std::shared_ptr<torch::comms::TorchComm>& parent_comm,
    std::shared_ptr<torch::comms::TorchComm>& child_comm,
    int num_groups,
    const std::unordered_set<int>& empty_groups) {
  int parent_rank = parent_comm->getRank();
  int parent_size = parent_comm->getSize();

  // Use createContigGroups to get the rank groups
  std::vector<std::vector<int>> rank_groups =
      createContigGroups(parent_size, num_groups, empty_groups);

  // Determine which group this rank belongs to
  int my_group = -1;
  int rank_in_group = -1;
  for (int group = 0; group < num_groups; group++) {
    // Check if parent_rank is in this group
    auto it = std::find(
        rank_groups[group].begin(), rank_groups[group].end(), parent_rank);
    if (it != rank_groups[group].end()) {
      my_group = group;
      rank_in_group =
          static_cast<int>(std::distance(rank_groups[group].begin(), it));
      break;
    }
  }

  // Verify child_comm based on whether this rank is in an empty group
  if (my_group == -1) {
    // This rank should not be in any group
    EXPECT_TRUE(child_comm == nullptr)
        << "Rank " << parent_rank << " should not have a child communicator";
  } else {
    // This rank should be in a non-empty group
    ASSERT_TRUE(child_comm != nullptr)
        << "Rank " << parent_rank << " should have a child communicator";

    // Verify child rank
    EXPECT_EQ(child_comm->getRank(), rank_in_group)
        << "Incorrect child rank for parent rank " << parent_rank;
  }
}

// Helper function to verify non-contiguous groups
void SplitTest::verifyNonContigGroups(
    std::shared_ptr<torch::comms::TorchComm>& parent_comm,
    std::shared_ptr<torch::comms::TorchComm>& child_comm,
    int num_groups,
    const std::unordered_set<int>& empty_groups) {
  int parent_rank = parent_comm->getRank();
  int parent_size = parent_comm->getSize();

  // Use createNonContigGroups to get the rank groups
  std::vector<std::vector<int>> rank_groups =
      createNonContigGroups(parent_size, num_groups, empty_groups);

  // Determine which group this rank belongs to
  int my_group = -1;
  int rank_in_group = -1;
  for (int group = 0; group < num_groups; group++) {
    // Check if parent_rank is in this group
    auto it = std::find(
        rank_groups[group].begin(), rank_groups[group].end(), parent_rank);
    if (it != rank_groups[group].end()) {
      my_group = group;
      rank_in_group =
          static_cast<int>(std::distance(rank_groups[group].begin(), it));
      break;
    }
  }

  // Verify child_comm based on whether this rank is in an empty group
  if (my_group == -1) {
    // This rank should not be in any group
    EXPECT_TRUE(child_comm == nullptr)
        << "Rank " << parent_rank << " should not have a child communicator";
  } else {
    // This rank should be in a non-empty group
    ASSERT_TRUE(child_comm != nullptr)
        << "Rank " << parent_rank << " should have a child communicator";

    // Verify child rank
    EXPECT_EQ(child_comm->getRank(), rank_in_group)
        << "Incorrect child rank for parent rank " << parent_rank;
  }
}

void SplitTest::testContiguousGroup(
    int /* num_groups */,
    const std::unordered_set<int>& /* empty_groups */) {
  int split_size = num_ranks_ / 2;
  if (split_size == 0) {
    split_size = 1; // Ensure at least one rank
  }

  bool rank_in_group = (rank_ < split_size);
  std::vector<int> ranks;

  if (rank_in_group) {
    // Only fill ranks if current rank is in the group
    for (int i = 0; i < split_size; i++) {
      ranks.push_back(i);
    }
  }
  // Otherwise, ranks remains empty

  // Call split function
  std::shared_ptr<torch::comms::TorchComm> new_torchcomm =
      torchcomm_->split(ranks, "contiguous_split_comm");

  if (rank_in_group) {
    // Current rank should be in the group and get a communicator
    ASSERT_TRUE(new_torchcomm != nullptr)
        << "Expected communicator but got nullptr for rank " << rank_;

    // Verify rank and size
    EXPECT_EQ(new_torchcomm->getRank(), rank_)
        << "New rank should match position in ranks list";
    EXPECT_EQ(new_torchcomm->getSize(), split_size)
        << "Size should match ranks list size";

    // Test communication within the child communicator
    testCommunication(new_torchcomm);

    // Finalize the communicator before it's destroyed
    new_torchcomm->finalize();
  } else {
    // Current rank should not be in the group and get nullptr
    EXPECT_TRUE(new_torchcomm == nullptr)
        << "Rank " << rank_ << " should not have a child communicator";
  }
}

void SplitTest::testNonContiguousGroup(
    int /* num_groups */,
    const std::unordered_set<int>& /* empty_groups */) {
  bool rank_in_group = (rank_ % 2 == 0); // Even ranks only
  std::vector<int> ranks;

  if (rank_in_group) {
    // Only fill ranks if current rank is in the group (even ranks)
    for (int i = 0; i < num_ranks_; i += 2) {
      ranks.push_back(i);
    }
  }
  // Otherwise, ranks remains empty

  // Call split function
  std::shared_ptr<torch::comms::TorchComm> new_torchcomm =
      torchcomm_->split(ranks, "noncontig_child_comm");

  if (rank_in_group) {
    // Current rank should be in the group and get a communicator
    ASSERT_TRUE(new_torchcomm != nullptr)
        << "Expected communicator but got nullptr for rank " << rank_;

    // Verify rank and size
    int expected_new_rank = rank_ / 2; // Position among even ranks
    int expected_size = (num_ranks_ + 1) / 2; // Number of even ranks
    EXPECT_EQ(new_torchcomm->getRank(), expected_new_rank)
        << "New rank should match position in even ranks";
    EXPECT_EQ(new_torchcomm->getSize(), expected_size)
        << "Size should match number of even ranks";

    // Test communication within the child communicator
    testCommunication(new_torchcomm);

    // Finalize the communicator before it's destroyed
    new_torchcomm->finalize();
  } else {
    // Current rank should not be in the group and get nullptr
    EXPECT_TRUE(new_torchcomm == nullptr)
        << "Rank " << rank_ << " should not have a child communicator";
  }
}

void SplitTest::testDuplicateRanks() {
  // Test that duplicate ranks are properly rejected with a runtime error
  // All ranks pass the same duplicate ranks list to ensure collective
  // validation Include all ranks but duplicate them to trigger validation error
  std::vector<int> ranks_with_duplicates;
  ranks_with_duplicates.reserve(2 * num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    ranks_with_duplicates.push_back(i);
  }
  for (int i = 0; i < num_ranks_; i++) {
    ranks_with_duplicates.push_back(i);
  }

  // Call split function - this should throw a runtime_error due to duplicate
  // ranks All ranks should get the same exception since duplicate detection
  // should happen during the collective validation phase
  EXPECT_THROW(
      torchcomm_->split(ranks_with_duplicates, "duplicate_comm"),
      std::runtime_error)
      << "Expected runtime_error for duplicate ranks on rank " << rank_;
}

void SplitTest::testRankNotInGroup() {
  // Test that a rank not including itself in the group is rejected with a
  // runtime error Each rank passes a list containing all other ranks except
  // itself This should trigger a runtime error since a rank cannot participate
  // in a split operation without including itself in the ranks list
  std::vector<int> ranks_excluding_self;
  for (int r = 0; r < num_ranks_; r++) {
    if (r != rank_) {
      ranks_excluding_self.push_back(r);
    }
  }

  // Call split function - this should throw a runtime_error since current rank
  // is not included in the ranks list it's trying to split with
  EXPECT_THROW(
      torchcomm_->split(ranks_excluding_self, "exclude_self_comm"),
      std::runtime_error)
      << "Expected runtime_error for rank not in group on rank " << rank_;
}

void SplitTest::testMultiLevel() {
  // First level split: Include first half of ranks
  int first_split_size = num_ranks_ / 2;
  if (first_split_size == 0) {
    first_split_size = 1;
  }

  bool rank_in_first_level = (rank_ < first_split_size);
  std::vector<int> first_level_ranks;

  if (rank_in_first_level) {
    // Only fill ranks if current rank is in the first level
    for (int i = 0; i < first_split_size; i++) {
      first_level_ranks.push_back(i);
    }
  }
  // Otherwise, first_level_ranks remains empty

  // Call split function to create first-level child communicator
  std::shared_ptr<torch::comms::TorchComm> first_level_comm =
      torchcomm_->split(first_level_ranks, "first_level_comm");

  // Test communication on parent communicator first (all ranks participate)
  auto options = at::TensorOptions().dtype(at::kFloat).device(device_type_);
  auto parent_input = at::ones({10}, options) * static_cast<float>(rank_ + 1);
  torchcomm_->all_reduce(parent_input, torch::comms::ReduceOp::SUM, false);
  verifyTensorEquality(parent_input.cpu(), num_ranks_ * (num_ranks_ + 1) / 2);

  if (!rank_in_first_level) {
    // Current rank is not in first level, should get nullptr
    EXPECT_TRUE(first_level_comm == nullptr)
        << "Rank " << rank_
        << " should not have a first-level child communicator";
    return;
  }

  // Current rank is in the first level, should have a communicator
  ASSERT_TRUE(first_level_comm != nullptr)
      << "Expected first-level communicator but got nullptr for rank " << rank_;

  // Get rank and size from the first-level communicator
  int first_level_rank = first_level_comm->getRank();
  int first_level_size = first_level_comm->getSize();

  // Second level split: Split first half of first-level communicator
  int second_split_size = first_level_size / 2;
  if (second_split_size == 0) {
    second_split_size = 1;
  }

  bool rank_in_second_level = (first_level_rank < second_split_size);
  std::vector<int> second_level_ranks;

  if (rank_in_second_level) {
    // Only fill ranks if current rank is in the second level
    for (int i = 0; i < second_split_size; i++) {
      second_level_ranks.push_back(i);
    }
  }
  // Otherwise, second_level_ranks remains empty

  // Call split function to create second-level child communicator
  std::shared_ptr<torch::comms::TorchComm> second_level_comm =
      first_level_comm->split(second_level_ranks, "second_level_comm");

  // Test communication on first-level communicator (only first-level ranks
  // participate)
  auto first_level_input =
      at::ones({10}, options) * static_cast<float>(first_level_rank + 1);
  first_level_comm->all_reduce(
      first_level_input, torch::comms::ReduceOp::SUM, false);
  verifyTensorEquality(
      first_level_input.cpu(), first_level_size * (first_level_size + 1) / 2);

  if (!rank_in_second_level) {
    // Current rank is not in second level, test only with first level
    EXPECT_TRUE(second_level_comm == nullptr)
        << "Rank " << first_level_rank
        << " should not have a second-level child communicator";

    first_level_comm->finalize();
    return;
  }

  // Current rank is in the second level, should have a communicator
  ASSERT_TRUE(second_level_comm != nullptr)
      << "Expected second-level communicator but got nullptr for first-level rank "
      << first_level_rank;

  // Test communication on second-level communicator (only second-level ranks
  // participate)
  int second_level_rank = second_level_comm->getRank();
  int second_level_size = second_level_comm->getSize();
  auto second_level_input =
      at::ones({10}, options) * static_cast<float>(second_level_rank + 1);
  second_level_comm->all_reduce(
      second_level_input, torch::comms::ReduceOp::SUM, false);
  verifyTensorEquality(
      second_level_input.cpu(),
      second_level_size * (second_level_size + 1) / 2);

  // Finalize the communicators before they're destroyed
  second_level_comm->finalize();
  first_level_comm->finalize();
}

void SplitTest::testMultipleSplitsSameRanks() {
  // Test splitting multiple times with the same ranks and same name
  // This verifies that the splitCounter_ properly creates unique store prefixes
  // preventing key collisions when the same name and ranks are used
  std::vector<int> all_ranks;
  all_ranks.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    all_ranks.push_back(i);
  }

  // Perform multiple splits with the same ranks AND same name
  const std::string split_name = "same_name_split";
  std::shared_ptr<torch::comms::TorchComm> split_comm_1 =
      torchcomm_->split(all_ranks, split_name);
  std::shared_ptr<torch::comms::TorchComm> split_comm_2 =
      torchcomm_->split(all_ranks, split_name);
  std::shared_ptr<torch::comms::TorchComm> split_comm_3 =
      torchcomm_->split(all_ranks, split_name);

  // All communicators should be valid since all ranks are included
  ASSERT_TRUE(split_comm_1 != nullptr)
      << "Expected first split communicator but got nullptr for rank " << rank_;
  ASSERT_TRUE(split_comm_2 != nullptr)
      << "Expected second split communicator but got nullptr for rank "
      << rank_;
  ASSERT_TRUE(split_comm_3 != nullptr)
      << "Expected third split communicator but got nullptr for rank " << rank_;

  // Verify each communicator has the correct rank and size
  EXPECT_EQ(split_comm_1->getRank(), rank_);
  EXPECT_EQ(split_comm_1->getSize(), num_ranks_);
  EXPECT_EQ(split_comm_2->getRank(), rank_);
  EXPECT_EQ(split_comm_2->getSize(), num_ranks_);
  EXPECT_EQ(split_comm_3->getRank(), rank_);
  EXPECT_EQ(split_comm_3->getSize(), num_ranks_);

  // Test communication on each split communicator independently
  // to verify they are separate and don't interfere with each other
  testCommunication(split_comm_1);
  testCommunication(split_comm_2);
  testCommunication(split_comm_3);

  // Finalize communicators
  split_comm_1->finalize();
  split_comm_2->finalize();
  split_comm_3->finalize();
}

void SplitTest::testGetRanksRoot() {
  // Test getRanks() on the root communicator
  // Should return sequential ranks [0, 1, 2, ..., size-1]
  std::vector<int> ranks = torchcomm_->getRanks();

  // Verify size matches getSize()
  EXPECT_EQ(static_cast<int>(ranks.size()), num_ranks_)
      << "getRanks() should return a vector of size equal to getSize()";

  // Verify ranks are sequential starting from 0
  std::vector<int> expected_ranks;
  expected_ranks.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    expected_ranks.push_back(i);
  }
  EXPECT_EQ(ranks, expected_ranks)
      << "Root communicator getRanks() should return sequential ranks [0, 1, ..., size-1]";
}

void SplitTest::testGetRanksAfterSplit() {
  // Test getRanks() after split returns the correct global ranks from parent
  // Split with non-contiguous ranks (even ranks only)
  bool rank_in_group = (rank_ % 2 == 0);
  std::vector<int> split_ranks;

  if (rank_in_group) {
    for (int i = 0; i < num_ranks_; i += 2) {
      split_ranks.push_back(i);
    }
  }

  std::shared_ptr<torch::comms::TorchComm> child_comm =
      torchcomm_->split(split_ranks, "getRanks_test_comm");

  if (rank_in_group) {
    ASSERT_TRUE(child_comm != nullptr)
        << "Expected child communicator for even rank " << rank_;

    // getRanks() on child should return the original global ranks
    std::vector<int> child_ranks = child_comm->getRanks();

    // Expected: the even ranks from the parent [0, 2, 4, ...]
    std::vector<int> expected_ranks;
    for (int i = 0; i < num_ranks_; i += 2) {
      expected_ranks.push_back(i);
    }

    EXPECT_EQ(child_ranks, expected_ranks)
        << "Child communicator getRanks() should return global ranks from parent";

    // Verify size matches
    EXPECT_EQ(static_cast<int>(child_ranks.size()), child_comm->getSize())
        << "getRanks() size should match getSize()";

    child_comm->finalize();
  } else {
    EXPECT_TRUE(child_comm == nullptr)
        << "Odd rank " << rank_ << " should not have a child communicator";
  }
}

void SplitTest::testGetRanksMultiLevelSplit() {
  // Test getRanks() with multi-level splits
  // First split: take first half of ranks
  // Second split: take first half of the child
  // Verify getRanks() returns correct global ranks at each level

  int first_split_size = num_ranks_ / 2;
  if (first_split_size == 0) {
    first_split_size = 1;
  }

  bool rank_in_first_level = (rank_ < first_split_size);
  std::vector<int> first_level_ranks;

  if (rank_in_first_level) {
    for (int i = 0; i < first_split_size; i++) {
      first_level_ranks.push_back(i);
    }
  }

  std::shared_ptr<torch::comms::TorchComm> first_level_comm =
      torchcomm_->split(first_level_ranks, "first_level_getRanks_comm");

  if (!rank_in_first_level) {
    EXPECT_TRUE(first_level_comm == nullptr)
        << "Rank " << rank_
        << " should not have a first-level child communicator";
    return;
  }

  ASSERT_TRUE(first_level_comm != nullptr)
      << "Expected first-level communicator for rank " << rank_;

  // Verify first level getRanks()
  std::vector<int> first_level_ranks_result = first_level_comm->getRanks();
  EXPECT_EQ(first_level_ranks_result, first_level_ranks)
      << "First level getRanks() should return global ranks [0, 1, ..., "
      << (first_split_size - 1) << "]";

  // Second level split: take first half of first level
  int first_level_rank = first_level_comm->getRank();
  int first_level_size = first_level_comm->getSize();

  int second_split_size = first_level_size / 2;
  if (second_split_size == 0) {
    second_split_size = 1;
  }

  bool rank_in_second_level = (first_level_rank < second_split_size);
  std::vector<int> second_level_ranks;

  if (rank_in_second_level) {
    for (int i = 0; i < second_split_size; i++) {
      second_level_ranks.push_back(i);
    }
  }

  std::shared_ptr<torch::comms::TorchComm> second_level_comm =
      first_level_comm->split(second_level_ranks, "second_level_getRanks_comm");

  if (!rank_in_second_level) {
    EXPECT_TRUE(second_level_comm == nullptr)
        << "First-level rank " << first_level_rank
        << " should not have a second-level child communicator";
    first_level_comm->finalize();
    return;
  }

  ASSERT_TRUE(second_level_comm != nullptr)
      << "Expected second-level communicator for first-level rank "
      << first_level_rank;

  // Verify second level getRanks() - should return global ranks from root
  // The second level contains ranks [0, 1, ..., second_split_size-1] from first
  // level which maps to global ranks [0, 1, ..., second_split_size-1]
  std::vector<int> second_level_ranks_result = second_level_comm->getRanks();

  std::vector<int> expected_global_ranks;
  expected_global_ranks.reserve(second_split_size);
  for (int i = 0; i < second_split_size; i++) {
    // Map through first_level_ranks to get global ranks
    expected_global_ranks.push_back(first_level_ranks[i]);
  }

  EXPECT_EQ(second_level_ranks_result, expected_global_ranks)
      << "Second level getRanks() should return global ranks mapped through parent";

  second_level_comm->finalize();
  first_level_comm->finalize();
}
