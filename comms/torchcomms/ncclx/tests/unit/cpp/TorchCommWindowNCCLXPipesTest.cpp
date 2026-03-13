// Copyright (c) Meta Platforms, Inc. and affiliates.
// Unit tests for TorchCommWindowNCCLX with PipesDeviceBackend.
//
// These tests verify Pipes-specific error paths without real hardware:
//   1. register_local_buffer() throws "not yet supported" for Pipes backend
//   2. get_device_window() throws when win_ is null (no tensor_register)
//
// Both tests set TORCHCOMMS_PIPES_DEVICE_API_ENABLE=1 in SetUp() so that
// TorchCommNCCLX::new_window() returns TorchCommWindowNCCLXPipes instead of
// TorchCommWindowNCCLXGin.
//
// Note: These tests use mocked NCCL/CUDA APIs and don't require real hardware.
// Integration tests with real GPUs and ctran are in PipesDeviceApiTest.

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
#if defined(ENABLE_PIPES)

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

namespace torch::comms::test {

class TorchCommWindowNCCLXPipesTest : public TorchCommNCCLXTest {
 protected:
  void SetUp() override {
    TorchCommNCCLXTest::SetUp();
    // Make new_window() return TorchCommWindowNCCLXPipes
    setenv("TORCHCOMMS_PIPES_DEVICE_API_ENABLE", "1", 1);
  }

  void TearDown() override {
    unsetenv("TORCHCOMMS_PIPES_DEVICE_API_ENABLE");
    TorchCommNCCLXTest::TearDown();
  }
};

TEST_F(TorchCommWindowNCCLXPipesTest, RegisterLocalBufferThrowsNotSupported) {
  // Verifies: register_local_buffer() throws immediately for Pipes backend.
  // IBGDA lkey registration requires ctran MR integration (planned follow-up).
  //
  // Code path: register_local_buffer() → Pipes constexpr branch → throw
  // Production value: Clear error when the unsupported Pipes API path is used.

  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // new_window() returns TorchCommWindowNCCLXPipes because
  // TORCHCOMMS_PIPES_DEVICE_API_ENABLE=1 is set in SetUp()
  auto win_base = comm->new_window();
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXPipes>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes";

  auto src_tensor = createTestTensor({5, 5});

  // register_local_buffer should throw immediately for Pipes (no MR support)
  EXPECT_THROW(
      {
        try {
          win->register_local_buffer(src_tensor);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("not yet supported") != std::string::npos)
              << "Error should indicate not yet supported, got: " << error_msg;
          throw;
        }
      },
      std::runtime_error);

  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommWindowNCCLXPipesTest, GetDeviceWindowThrowsIfWinNull) {
  // Verifies: get_device_window() throws a Pipes-specific error when
  // tensor_register() has not been called (win_ remains null).
  //
  // Code path: get_device_window() → Pipes win_ null check → throw
  // Production value: Prevents silent failures when the API is misused.

  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Create Pipes window WITHOUT calling tensor_register → win_ stays null
  auto win_base = comm->new_window();
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXPipes>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes";

  // get_device_window() should throw Pipes-specific error about win_ not init
  EXPECT_THROW(
      {
        try {
          win->get_device_window();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Window not initialized") != std::string::npos ||
              error_msg.find("tensor_register") != std::string::npos)
              << "Error should indicate window not initialized, got: "
              << error_msg;
          throw;
        }
      },
      std::runtime_error);

  EXPECT_NO_THROW(comm->finalize());
}

} // namespace torch::comms::test

#endif // ENABLE_PIPES
#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API
