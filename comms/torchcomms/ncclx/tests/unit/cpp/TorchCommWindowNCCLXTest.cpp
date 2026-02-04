// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

namespace torch::comms::test {

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

  // CPU window operations
  auto win = comm->new_window();
  win->tensor_register(tensor);

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

  testOperation([&]() { win->put(large_input_tensor, 0, 0, false); });

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

// =============================================================================
// Device API Tests for get_device_window()
// =============================================================================
//
// These tests verify the device API integration in TorchCommWindowNCCLX:
//   1. get_device_window() requires tensor_register() first
//   2. get_device_window() returns the same pointer on subsequent calls
//   3. Proper cleanup when window is destroyed
//
// Note: These tests use mocked NCCL/CUDA APIs and don't test actual device
// operations. Integration tests with real GPUs are in DeviceApiTest.

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API

TEST_F(TorchCommWindowNCCLXTest, GetDeviceWindowWithoutTensorRegisterThrows) {
  // Verifies: get_device_window() requires tensor_register() to be called first
  // Code path: get_device_window() with local_comm_initialized_ == false
  // Production value: Prevents cryptic errors when API is misused
  //
  // The device API requires:
  //   1. tensor_register() creates local_comm_ via ncclCommSplit
  //   2. get_device_window() uses local_comm_ for device state creation
  // Without tensor_register(), get_device_window() would fail.

  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Create window WITHOUT registering tensor
  auto win_base = comm->new_window();
  // Cast to derived class to access get_device_window()
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXGin>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // get_device_window() should throw because tensor_register() wasn't called
  EXPECT_THROW(
      {
        try {
          win->get_device_window();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          // Should indicate window is not registered or local comm not
          // initialized
          EXPECT_TRUE(
              error_msg.find("null") != std::string::npos ||
              error_msg.find("not initialized") != std::string::npos ||
              error_msg.find("not registered") != std::string::npos)
              << "Error message should indicate missing prerequisite, got: "
              << error_msg;
          throw;
        }
      },
      std::runtime_error);

  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommWindowNCCLXTest, GetDeviceWindowReturnsConsistentValue) {
  // Verifies: Multiple calls to get_device_window() return the same pointer
  // Code path: get_device_window() when device_window_ is already created
  // Production value: Ensures idempotency - pointer should be cached
  //
  // The device window is allocated once in device memory and cached. This is
  // important because:
  //   1. Multiple kernels may call get_device_window()
  //   2. The returned pointer should be the same across calls
  //   3. The device window struct is in GPU memory (cudaMalloc)

  setupRankAndSize(0, 8);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  auto tensor = createTestTensor({10, 10});
  auto win_base = comm->new_window();
  // Cast to derived class to access get_device_window()
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXGin>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";
  win->tensor_register(tensor);

  // First call creates the device window - returns device pointer
  auto* device_win1 = win->get_device_window();
  EXPECT_NE(device_win1, nullptr) << "Device window pointer should not be null";

  // Second call should return the SAME pointer (cached)
  auto* device_win2 = win->get_device_window();
  EXPECT_EQ(device_win1, device_win2)
      << "get_device_window() should return same pointer on subsequent calls";

  // Third call with different parameters should STILL return same pointer
  // (parameters are only used on first call)
  auto* device_win3 = win->get_device_window(16, 16, 2);
  EXPECT_EQ(device_win1, device_win3)
      << "get_device_window() should ignore parameters on subsequent calls";

  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommWindowNCCLXTest, GetDeviceWindowDefaultParameters) {
  // Verifies: Default parameters create a valid device window
  // Code path: get_device_window() with signal_count/counter_count == -1
  // Production value: Ensures sensible defaults for common use cases
  //
  // When signal_count or counter_count is -1 (default), they should be
  // set to comm_size. This ensures each rank can have at least one
  // signal/counter per peer.

  setupRankAndSize(0, 8); // rank 0 of 8 (use rank 0 for proper mock setup)
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  auto tensor = createTestTensor({10, 10});
  auto win_base = comm->new_window();
  // Cast to derived class to access get_device_window()
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXGin>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";
  win->tensor_register(tensor);

  // Call with default parameters - returns device pointer
  auto* device_win = win->get_device_window();

  // The device window should have been created successfully with defaults
  // Since the pointer points to device memory, we can only verify it's non-null
  EXPECT_NE(device_win, nullptr) << "Device window pointer should not be null";

  EXPECT_NO_THROW(comm->finalize());
}

#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API

} // namespace torch::comms::test
