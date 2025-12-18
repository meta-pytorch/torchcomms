// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

namespace torch {
namespace comms {
namespace test {

class TorchCommWindowNCCLXTest : public TorchCommNCCLXTest {};

TEST_F(TorchCommWindowNCCLXTest, windowPutExceedWindowSize) {
  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});
  auto large_input_tensor =
      createTestTensor({20, 10}); // Divisible by comm_size (2)

  // Helper lambda to test that operations throw "exceeds the window size"
  // exception
  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(
                error_msg.find("exceeds the window size") != std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  // CPU window operations after finalize
  auto win = comm->new_window();
  win->tensor_register(tensor);

  testOperation([&]() { win->put(large_input_tensor, 0, 0, false); });

  testOperation([&]() {
    win->getTensor(
        0, {large_input_tensor.sizes()}, large_input_tensor.scalar_type(), 0);
  });

  // Finalize should wait for work to complete
  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommWindowNCCLXTest, windowRegisterWithInvalidTensor) {
  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});

  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(
                error_msg.find("valid tensor is required") !=
                std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  at::Tensor win_buf;

  testOperation([&]() {
    auto win = comm->new_window();
    win->tensor_register(win_buf);
  });

  // Finalize should wait for work to complete
  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(
    TorchCommNCCLXTest,
    WindowOperationsWithoutInitializationThrowException) {
  // Setup CCA expectations - no init calls
  setupCCAExpectations(0, 1, 1);

  auto comm = createMockedTorchComm();

  // Initialize and then finalize the communicator
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});

  // Helper lambda to test that operations throw "not initialized" exception
  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  // test window operations without initialization
  testOperation([&]() { comm->new_window(); });
}

TEST_F(TorchCommWindowNCCLXTest, WindowOperationsAfterFinalizeThrowException) {
  // Setup CCA expectations - init and finalize calls
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  // Initialize and then finalize the communicator
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);
  setupNormalDestruction(*comm);
  comm->finalize();

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});

  // Helper lambda to test that operations throw "not initialized" exception
  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  // Test window operations after finalize
  testOperation([&]() { comm->new_window(); });
}

} // namespace test
} // namespace comms
} // namespace torch
