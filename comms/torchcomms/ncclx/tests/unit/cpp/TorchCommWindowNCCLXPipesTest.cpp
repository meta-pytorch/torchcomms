// Copyright (c) Meta Platforms, Inc. and affiliates.
// Unit tests for TorchCommWindowNCCLX with PipesDeviceBackend.
//
// These tests verify Pipes-specific error paths without real hardware:
//   1. register_local_buffer() throws when device window not initialized
//   2. get_device_window() throws when win_ is null (no tensor_register)
//
// Both tests set NCCL_CTRAN_USE_PIPES=1 in SetUp() so that
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
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
  }

  void TearDown() override {
    unsetenv("NCCL_CTRAN_USE_PIPES");
    TorchCommNCCLXTest::TearDown();
  }
};

TEST_F(
    TorchCommWindowNCCLXPipesTest,
    RegisterLocalBufferThrowsIfDeviceWindowNotInit) {
  // Verifies: register_local_buffer() throws when get_device_window() has not
  // been called first (device_window_ is null).
  //
  // Code path: register_local_buffer() → device_window_ null check → throw
  // Production value: Clear error when the API is called out of order.

  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // new_window() returns TorchCommWindowNCCLXPipes because
  // NCCL_CTRAN_USE_PIPES=1 is set in SetUp()
  auto win_base = comm->new_window();
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXPipes>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes";

  auto src_tensor = createTestTensor({5, 5});

  // register_local_buffer should throw because device window not initialized
  EXPECT_THROW(
      {
        try {
          win->register_local_buffer(src_tensor);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Device window not initialized") !=
              std::string::npos)
              << "Error should indicate device window not initialized, got: "
              << error_msg;
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

TEST_F(TorchCommWindowNCCLXPipesTest, RegisterLocalBufferSuccess) {
  // Verifies: register_local_buffer() succeeds for the Pipes backend and
  // returns a RegisteredBuffer with the correct lkey from
  // winLocalRegisterBuffer, and backend_window == nullptr (Pipes doesn't use
  // backend_window — only GIN does).
  //
  // Also verifies deregister_local_buffer() calls winLocalDeregisterBuffer.

  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  auto win_base = comm->new_window();
  auto win = std::dynamic_pointer_cast<TorchCommWindowNCCLXPipes>(win_base);
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes";

  // tensor_register() — uses default mock for commWindowRegister
  auto dst_tensor = createTestTensor({5, 5});
  EXPECT_NO_THROW(win->tensor_register(dst_tensor));

  // Mock winCreateDeviceWin to succeed — returns a fake device pointer.
  // PipesDeviceBackend::create_device_window() calls this, then malloc+memcpy.
  void* fake_pipes_dev_win = reinterpret_cast<void*>(0xBEEF);
  EXPECT_CALL(*nccl_mock_, winCreateDeviceWin(_, _, _, _, _))
      .WillOnce(
          DoAll(SetArgPointee<4>(fake_pipes_dev_win), Return(ncclSuccess)));

  // Mock cuda malloc/memcpy for creating TorchCommDeviceWindow in device mem
  void* fake_dev_window_ptr = reinterpret_cast<void*>(0xDEAD);
  EXPECT_CALL(*cuda_mock_, malloc(_, _))
      .WillOnce(
          DoAll(SetArgPointee<0>(fake_dev_window_ptr), Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, memcpy(_, _, _, cudaMemcpyHostToDevice))
      .WillOnce(Return(cudaSuccess));

  // get_device_window() — creates the Pipes device window
  auto* dev_win = win->get_device_window();
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  // Mock winLocalRegisterBuffer to return a specific lkey
  const uint32_t expected_lkey = 0x12345678;
  EXPECT_CALL(*nccl_mock_, winLocalRegisterBuffer(_, _, _, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_lkey), Return(ncclSuccess)));

  auto src_tensor = createTestTensor({5, 5});
  auto buf = win->register_local_buffer(src_tensor);

  // Pipes backend: lkey is set, backend_window is nullptr
  EXPECT_EQ(buf.lkey, expected_lkey);
  EXPECT_EQ(buf.backend_window, nullptr)
      << "Pipes backend should not set backend_window";
  EXPECT_NE(buf.base_ptr, nullptr);
  EXPECT_GT(buf.size, 0u);

  // Deregister and verify winLocalDeregisterBuffer is called
  EXPECT_CALL(*nccl_mock_, winLocalDeregisterBuffer(_, buf.base_ptr))
      .WillOnce(Return(ncclSuccess));
  win->deregister_local_buffer(buf);

  EXPECT_EQ(buf.base_ptr, nullptr)
      << "base_ptr should be cleared after deregister";
  EXPECT_EQ(buf.lkey, 0u) << "lkey should be cleared after deregister";

  // Cleanup mocks for destruction — use ON_CALL since finalize() also
  // triggers other free/destroy calls (barrier buffer, etc.)
  ON_CALL(*nccl_mock_, winDestroyDeviceWin(_))
      .WillByDefault(Return(ncclSuccess));

  EXPECT_NO_THROW(comm->finalize());
}

} // namespace torch::comms::test

#endif // ENABLE_PIPES
#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API
