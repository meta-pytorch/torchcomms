// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"

// Base class with common helper functions
class MultiCommTestBase {
 protected:
  // Helper function to test communication within a communicator
  void testCommunication(const std::unique_ptr<TorchCommTestWrapper>& wrapper) {
    // Get the communicator from the wrapper
    auto comm = wrapper->getTorchComm();

    // Skip if communicator is null
    if (!comm) {
      return;
    }

    // Get rank and size from the communicator
    int rank = comm->getRank();
    int size = comm->getSize();

    // Get device from wrapper
    int device_index = comm->getRank() % at::cuda::device_count();
    auto device = c10::Device(c10::DeviceType::CUDA, device_index);
    auto options = at::TensorOptions().dtype(at::kFloat).device(device);
    auto input = at::ones({10}, options) * static_cast<float>(rank + 1);

    // For ranks in groups, test all_reduce
    comm->all_reduce(input, torch::comms::ReduceOp::SUM, false);

    // Verify the result using verifyTensorWithValue
    verifyTensorEquality(input.cpu(), size * (size + 1) / 2);
  }

  // Helper function to test simultaneous communication across multiple
  // communicators
  void testSimultaneousCommunication(
      const std::vector<std::unique_ptr<TorchCommTestWrapper>>& wrappers) {
    if (wrappers.empty()) {
      return;
    }

    std::vector<at::Tensor> inputs;
    std::vector<float> expected_values;
    std::vector<int> device_indices;

    // Create input tensors for each communicator
    for (size_t i = 0; i < wrappers.size(); i++) {
      const auto& comm = wrappers[i]->getTorchComm();
      int comm_rank = comm->getRank();
      int comm_size = comm->getSize();

      // Get device from wrapper
      int device_index = comm->getRank() % at::cuda::device_count();
      auto device = c10::Device(c10::DeviceType::CUDA, device_index);
      auto options = at::TensorOptions().dtype(at::kFloat).device(device);
      device_indices.push_back(device_index);

      auto input = at::ones({10}, options) * static_cast<float>(comm_rank + 1);
      inputs.push_back(input);

      // Calculate expected result for this communicator
      float expected = comm_size * (comm_size + 1) / 2;
      expected_values.push_back(expected);
    }

    // Issue all_reduce operations on all communicators simultaneously
    for (size_t i = 0; i < wrappers.size(); i++) {
      wrappers[i]->getTorchComm()->all_reduce(
          inputs[i], torch::comms::ReduceOp::SUM, false);
    }

    // Verify results for all communicators
    for (size_t i = 0; i < wrappers.size(); i++) {
      std::string description =
          "comm_" + std::to_string(i) + " simultaneous all_reduce result";
      verifyTensorEquality(inputs[i].cpu(), expected_values[i], description);
    }
  }
};

class MultiCommTest : public ::testing::Test, public MultiCommTestBase {
 public:
  MultiCommTest() {}

 protected:
  void SetUp() override {}

  void TearDown() override {}

  void destroyStoreAndSyncStream(
      c10::intrusive_ptr<c10d::Store>&& store,
      std::shared_ptr<torch::comms::TorchComm> torchcomm) {
    destroyStore(std::move(store), torchcomm);

    int rank = torchcomm->getRank();
    int device_index = rank % at::cuda::device_count();

    // Synchronize the CUDA stream on the calculated device
    at::cuda::getCurrentCUDAStream(device_index).synchronize();
  }
};

class MultiCommNoStoreTest
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>>,
      public MultiCommTestBase {
 public:
  MultiCommNoStoreTest() {}

 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Test fixtures
TEST_F(MultiCommTest, TwoCommsOneStore) {
  SCOPED_TRACE(::testing::Message() << "Testing with two communicators");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a shared store for both communicators
  auto store = createStore();

  // Create first communicator using the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Create second communicator using the same store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // The wrappers will clean up the communicators and store in their destructors
}

TEST_F(MultiCommTest, ThreeCommsOneStore) {
  SCOPED_TRACE(::testing::Message() << "Testing with three communicators");

  // Create three communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a shared store for all three communicators
  auto store = createStore();

  // Create first communicator using the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Create second communicator using the same store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Create third communicator using the same store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // The wrappers will clean up the communicators and store in their destructors
}

TEST_F(MultiCommTest, MixedOpsOneStore) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing mixed operations across multiple communicators");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a shared store for both communicators
  auto store = createStore();

  // Create first communicator using the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Create second communicator using the same store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Prepare tensors for different operations
  int device_index0 = comms[0]->getRank() % at::cuda::device_count();
  int device_index1 = comms[1]->getRank() % at::cuda::device_count();
  c10::Device device0 = c10::Device(c10::DeviceType::CUDA, device_index0);
  c10::Device device1 = c10::Device(c10::DeviceType::CUDA, device_index1);
  auto options0 = at::TensorOptions().dtype(at::kFloat).device(device0);
  auto options1 = at::TensorOptions().dtype(at::kFloat).device(device1);

  // For all_reduce on first communicator
  auto input1 =
      at::ones({10}, options0) * static_cast<float>(comms[0]->getRank() + 1);
  int expected1 = comms[0]->getSize() * (comms[0]->getSize() + 1) / 2;

  // For broadcast on second communicator
  const int root_rank = 0;
  const int broadcast_value = 42;
  auto input2 = comms[1]->getRank() == root_rank
      ? at::ones({10}, options1) * static_cast<float>(broadcast_value)
      : at::zeros({10}, options1);

  // Issue operations simultaneously
  comms[0]->all_reduce(input1, torch::comms::ReduceOp::SUM, false);
  comms[1]->broadcast(input2, root_rank, false);

  // Verify results
  verifyTensorEquality(input1.cpu(), expected1, "comm_0 all_reduce result");
  verifyTensorEquality(
      input2.cpu(), broadcast_value, "comm_1 broadcast result");

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // The wrappers will clean up the communicators and store in their destructors
}

// Tests with separate stores for each communicator
TEST_F(MultiCommTest, TwoCommsSeparateStores) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing with two communicators with separate stores");

  // Create two communicators with separate stores
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a store
  auto store = createStore();

  // Create first communicator with the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy and recreate the store
  destroyStoreAndSyncStream(std::move(store), comms[0]);
  store = createStore();

  // Create second communicator with the recreated store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy the store after the final communicator is created
  destroyStoreAndSyncStream(std::move(store), comms[1]);

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators in their destructors
}

TEST_F(MultiCommTest, ThreeCommsSeparateStores) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing with three communicators with separate stores");

  // Create three communicators with separate stores
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a store
  auto store = createStore();

  // Create first communicator with the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy and recreate the store
  destroyStoreAndSyncStream(std::move(store), comms[0]);
  store = createStore();

  // Create second communicator with the recreated store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy and recreate the store
  destroyStoreAndSyncStream(std::move(store), comms[1]);
  store = createStore();

  // Create third communicator with the recreated store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy the store after the final communicator is created
  destroyStoreAndSyncStream(std::move(store), comms[2]);

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators in their destructors
}

TEST_F(MultiCommTest, MixedOpsSeparateStores) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing mixed operations across multiple communicators with separate stores");

  // Create two communicators with separate stores
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a store
  auto store = createStore();

  // Create first communicator with the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy and recreate the store
  destroyStoreAndSyncStream(std::move(store), comms[0]);
  store = createStore();

  // Create second communicator with the recreated store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Destroy the store after the final communicator is created
  destroyStoreAndSyncStream(std::move(store), comms[1]);

  // Prepare tensors for different operations
  int device_index0 = comms[0]->getRank() % at::cuda::device_count();
  int device_index1 = comms[1]->getRank() % at::cuda::device_count();
  c10::Device device0 = c10::Device(c10::DeviceType::CUDA, device_index0);
  c10::Device device1 = c10::Device(c10::DeviceType::CUDA, device_index1);
  auto options0 = at::TensorOptions().dtype(at::kFloat).device(device0);
  auto options1 = at::TensorOptions().dtype(at::kFloat).device(device1);

  // For all_reduce on first communicator
  auto input1 =
      at::ones({10}, options0) * static_cast<float>(comms[0]->getRank() + 1);
  int expected1 = comms[0]->getSize() * (comms[0]->getSize() + 1) / 2;

  // For broadcast on second communicator
  const int root_rank = 0;
  const int broadcast_value = 42;
  auto input2 = comms[1]->getRank() == root_rank
      ? at::ones({10}, options1) * static_cast<float>(broadcast_value)
      : at::zeros({10}, options1);

  // Issue operations simultaneously
  comms[0]->all_reduce(input1, torch::comms::ReduceOp::SUM, false);
  comms[1]->broadcast(input2, root_rank, false);

  // Verify results
  verifyTensorEquality(input1.cpu(), expected1, "comm_0 all_reduce result");
  verifyTensorEquality(
      input2.cpu(), broadcast_value, "comm_1 broadcast result");

  // The wrappers will clean up the communicators in their destructors
}

TEST_F(MultiCommNoStoreTest, TwoCommsNoStore) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing with two communicators with no store");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators and store in their
  // destructors
}

TEST_F(MultiCommNoStoreTest, ThreeCommsNoStore) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing with three communicators with no store");

  // Create three communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators and store in their
  // destructors
}

TEST_F(MultiCommNoStoreTest, MixedOpsNoStore) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing mixed operations across multiple communicators with no store");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Prepare tensors for different operations
  // Get device from wrapper
  int device_index0 = comms[0]->getRank() % at::cuda::device_count();
  int device_index1 = comms[1]->getRank() % at::cuda::device_count();
  c10::Device device0 = c10::Device(c10::DeviceType::CUDA, device_index0);
  c10::Device device1 = c10::Device(c10::DeviceType::CUDA, device_index1);
  auto options0 = at::TensorOptions().dtype(at::kFloat).device(device0);
  auto options1 = at::TensorOptions().dtype(at::kFloat).device(device1);

  // For all_reduce on first communicator
  auto input1 =
      at::ones({10}, options0) * static_cast<float>(comms[0]->getRank() + 1);
  int expected1 = comms[0]->getSize() * (comms[0]->getSize() + 1) / 2;

  // For broadcast on second communicator
  const int root_rank = 0;
  const int broadcast_value = 42;
  auto input2 = comms[1]->getRank() == root_rank
      ? at::ones({10}, options1) * static_cast<float>(broadcast_value)
      : at::zeros({10}, options1);

  // Issue operations simultaneously
  comms[0]->all_reduce(input1, torch::comms::ReduceOp::SUM, false);
  comms[1]->broadcast(input2, root_rank, false);

  // Verify results
  verifyTensorEquality(input1.cpu(), expected1, "comm_0 all_reduce result");
  verifyTensorEquality(
      input2.cpu(), broadcast_value, "comm_1 broadcast result");

  // The wrappers will clean up the communicators and store in their
  // destructors
}

TEST_F(MultiCommTest, TwoCommsMixedStore) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing with two communicators with mixed store (one explicit, one nullptr)");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a store for the first communicator
  auto store = createStore();

  // Create first communicator using explicit store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // Create second communicator using no store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators and store in their destructors
}

TEST_F(MultiCommTest, ThreeCommsMixedStore) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing with three communicators with mixed store (two explicit, one nullptr)");

  // Create three communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a shared store for the first two communicators
  auto store = createStore();

  // Create first communicator using the store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  // Create second communicator using the same store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // Create third communicator using no store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Test communication on each communicator individually
  for (size_t i = 0; i < wrappers.size(); i++) {
    testCommunication(wrappers[i]);
  }

  // Test simultaneous communication across all communicators
  testSimultaneousCommunication(wrappers);

  // The wrappers will clean up the communicators and store in their destructors
}

TEST_F(MultiCommTest, MixedOpsMixedStore) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing mixed operations across multiple communicators with mixed store");

  // Create two communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;

  // Create a store for the first communicator
  auto store = createStore();

  // Create first communicator using explicit store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>(store));
  comms.push_back(wrappers.back()->getTorchComm());

  destroyStoreAndSyncStream(std::move(store), comms[0]);

  // Create second communicator using no store
  wrappers.push_back(std::make_unique<TorchCommTestWrapper>());
  comms.push_back(wrappers.back()->getTorchComm());

  // Prepare tensors for different operations
  int device_index0 = comms[0]->getRank() % at::cuda::device_count();
  int device_index1 = comms[1]->getRank() % at::cuda::device_count();
  c10::Device device0 = c10::Device(c10::DeviceType::CUDA, device_index0);
  c10::Device device1 = c10::Device(c10::DeviceType::CUDA, device_index1);
  auto options0 = at::TensorOptions().dtype(at::kFloat).device(device0);
  auto options1 = at::TensorOptions().dtype(at::kFloat).device(device1);

  // For all_reduce on first communicator
  auto input1 =
      at::ones({10}, options0) * static_cast<float>(comms[0]->getRank() + 1);
  int expected1 = comms[0]->getSize() * (comms[0]->getSize() + 1) / 2;

  // For broadcast on second communicator
  const int root_rank = 0;
  const int broadcast_value = 42;
  auto input2 = comms[1]->getRank() == root_rank
      ? at::ones({10}, options1) * static_cast<float>(broadcast_value)
      : at::zeros({10}, options1);

  // Issue operations simultaneously
  comms[0]->all_reduce(input1, torch::comms::ReduceOp::SUM, false);
  comms[1]->broadcast(input2, root_rank, false);

  // Verify results
  verifyTensorEquality(input1.cpu(), expected1, "comm_0 all_reduce result");
  verifyTensorEquality(
      input2.cpu(), broadcast_value, "comm_1 broadcast result");

  // The wrappers will clean up the communicators and store in their destructors
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
