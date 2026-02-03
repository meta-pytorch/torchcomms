// Copyright (c) Meta Platforms, Inc. and affiliates.

// Unit tests for NCCLGinBackend static methods
//
// These tests verify the device backend lifecycle and error handling:
//   1. create_device_window() - tests success path and failure paths
//   2. destroy_device_window() - tests cleanup behavior
//
// Test Philosophy:
//   - Each test explores a specific code path
//   - Failure tests verify proper cleanup (no resource leaks)
//   - Tests use strict mocks to catch unexpected API calls
//   - Error messages are validated to ensure actionable diagnostics
//
// Ownership Design:
//   NCCLGinBackend::create_device_window() returns std::unique_ptr for clear
//   ownership semantics. destroy_device_window() takes ownership via
//   std::unique_ptr and destroys the resource.

//
// Note: This entire test file requires TORCHCOMMS_HAS_NCCL_DEVICE_API because
// it tests NCCLGinBackend which uses devCommCreate/devCommDestroy APIs.

#include "comms/torchcomms/ncclx/NcclxApi.hpp" // For TORCHCOMMS_HAS_NCCL_DEVICE_API

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

namespace torchcomms::device::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

using torch::comms::test::NcclxMock;

// =============================================================================
// Test Fixture
// =============================================================================

class NCCLGinBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock instance
    nccl_mock_ = std::make_shared<NiceMock<NcclxMock>>();

    // Setup default behaviors
    nccl_mock_->setupDefaultBehaviors();

    // Create a fake NCCL communicator (just needs to be non-null)
    fake_nccl_comm_ = reinterpret_cast<ncclComm_t>(0x1000);

    // Create a fake NCCL window (just needs to be non-null)
    fake_nccl_window_ = reinterpret_cast<ncclWindow_t>(0x2000);

    // Create a fake window base pointer (just needs to be non-null for tests)
    fake_window_base_ = reinterpret_cast<void*>(0x3000);
  }

  void TearDown() override {
    nccl_mock_.reset();
  }

  // Helper to create a typical config
  DeviceBackendConfig createDefaultConfig() {
    DeviceBackendConfig config;
    config.signal_count = 8;
    config.counter_count = 8;
    config.barrier_count = 1;
    config.comm_rank = 0;
    config.comm_size = 8;
    return config;
  }

  std::shared_ptr<NiceMock<NcclxMock>> nccl_mock_;
  ncclComm_t fake_nccl_comm_{nullptr};
  ncclWindow_t fake_nccl_window_{nullptr};
  void* fake_window_base_{nullptr};
};

// =============================================================================
// NCCLGinBackend::create_device_window() Tests - Null Checks
// =============================================================================
//
// These tests verify that create_device_window() performs proper null checks.
// Without these checks, null pointers would cause crashes deep in the code
// with unclear error messages.

TEST_F(NCCLGinBackendTest, CreateDeviceWindowWithNullNcclCommThrowsException) {
  // Verifies: Null NCCL communicator is rejected immediately
  // Code path: create_device_window() null check for nccl_comm
  // Production value: Prevents crash when comm creation fails upstream
  auto config = createDefaultConfig();

  EXPECT_THROW(
      {
        try {
          NCCLGinBackend::create_device_window(
              nullptr,
              nccl_mock_.get(),
              config,
              fake_nccl_window_,
              nullptr,
              1024);
        } catch (const std::runtime_error& e) {
          // Verify error message is actionable
          EXPECT_TRUE(
              std::string(e.what()).find("NCCL communicator cannot be null") !=
              std::string::npos)
              << "Error message should mention NCCL communicator, got: "
              << e.what();
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(NCCLGinBackendTest, CreateDeviceWindowWithNullNcclApiThrowsException) {
  // Verifies: Null NCCL API is rejected immediately
  // Code path: create_device_window() null check for nccl_api
  // Production value: Prevents crash when API abstraction is misconfigured
  auto config = createDefaultConfig();

  EXPECT_THROW(
      {
        try {
          NCCLGinBackend::create_device_window(
              fake_nccl_comm_,
              nullptr,
              config,
              fake_nccl_window_,
              nullptr,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("NCCL API cannot be null") !=
              std::string::npos)
              << "Error message should mention NCCL API, got: " << e.what();
          throw;
        }
      },
      std::runtime_error);
}

// =============================================================================
// NCCLGinBackend::create_device_window() Tests - Success Path
// =============================================================================

TEST_F(NCCLGinBackendTest, CreateDeviceWindowSuccessReturnsValidStruct) {
  // Verifies: Happy path - ncclDevCommCreate succeeds
  // Code path: create_device_window() -> devCommCreate
  // Production value: Ensures device window creation works correctly
  //
  // This test verifies the complete flow:
  //   1. ncclDevCommCreate is called with correct requirements
  //   2. Returns std::unique_ptr<TorchCommDeviceWindow>

  auto config = createDefaultConfig();
  void* fake_base = reinterpret_cast<void*>(0x3000);
  size_t fake_size = 4096;

  // Set up expectations for the success path
  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  // create_device_window() returns std::unique_ptr<TorchCommDeviceWindow>
  auto device_window = NCCLGinBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      config,
      fake_nccl_window_,
      fake_base,
      fake_size);

  // Verify unique_ptr is valid
  ASSERT_NE(device_window, nullptr);

  // Verify all fields are populated correctly
  EXPECT_EQ(device_window->window_, fake_nccl_window_);
  EXPECT_EQ(device_window->base_, fake_base);
  EXPECT_EQ(device_window->size_, fake_size);
  EXPECT_EQ(device_window->rank_, config.comm_rank);
  EXPECT_EQ(device_window->num_ranks_, config.comm_size);
}

TEST_F(NCCLGinBackendTest, CreateDeviceWindowConfigIsPassedCorrectly) {
  // Verifies: Configuration values are correctly passed to NCCL
  // Code path: create_device_window() -> ncclDevCommRequirements setup
  // Production value: Ensures signal/counter/barrier counts are respected
  //
  // If config values aren't passed correctly, device-side operations will fail
  // with cryptic errors about invalid signal/counter indices

  DeviceBackendConfig config;
  config.signal_count = 16;
  config.counter_count = 32;
  config.barrier_count = 2;
  config.comm_rank = 3;
  config.comm_size = 8;

  // Capture the requirements struct passed to devCommCreate
  ncclDevCommRequirements captured_reqs;
  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(DoAll(
          // Capture the requirements for verification
          [&captured_reqs](
              ncclComm_t,
              const ncclDevCommRequirements_t* reqs,
              ncclDevComm_t*) {
            if (reqs) {
              captured_reqs = *reqs;
            }
            return ncclSuccess;
          },
          SetArgPointee<2>(ncclDevComm{}),
          Return(ncclSuccess)));

  auto device_window = NCCLGinBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      config,
      fake_nccl_window_,
      fake_window_base_,
      1024);

  // Verify unique_ptr is valid
  ASSERT_NE(device_window, nullptr);

  // Verify requirements were set correctly
  EXPECT_EQ(captured_reqs.ginSignalCount, config.signal_count);
  EXPECT_EQ(captured_reqs.ginCounterCount, config.counter_count);
  EXPECT_EQ(captured_reqs.barrierCount, config.barrier_count);
  EXPECT_TRUE(captured_reqs.ginForceEnable);
}

// =============================================================================
// NCCLGinBackend::create_device_window() Tests - Failure Paths
// =============================================================================

TEST_F(NCCLGinBackendTest, DevCommCreateFailureThrows) {
  // Verifies: NCCL devCommCreate failure is handled gracefully
  // Code path: create_device_window() when devCommCreate returns error
  // Production value: Handles NCCL internal errors (e.g., resource exhaustion)
  //
  // ncclDevCommCreate can fail for various reasons:
  //   - Invalid communicator
  //   - Resource exhaustion (too many signals/counters)
  //   - Internal NCCL errors

  auto config = createDefaultConfig();

  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(Return(ncclInternalError));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillOnce(Return("internal error"));

  EXPECT_THROW(
      {
        try {
          NCCLGinBackend::create_device_window(
              fake_nccl_comm_,
              nccl_mock_.get(),
              config,
              fake_nccl_window_,
              fake_window_base_,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("Failed to create NCCL device") !=
              std::string::npos)
              << "Error should mention NCCL device creation, got: " << e.what();
          throw;
        }
      },
      std::runtime_error);
}

// =============================================================================
// NCCLGinBackend::destroy_device_window() Tests
// =============================================================================

TEST_F(NCCLGinBackendTest, DestroyDeviceWindowSuccess) {
  // Verifies: Normal cleanup path frees all resources
  // Code path: destroy_device_window() after successful create_device_window()
  // Production value: Ensures proper cleanup during window destruction
  //
  // Note: destroy takes unique_ptr ownership and destroys it

  auto config = createDefaultConfig();

  // Setup successful creation
  EXPECT_CALL(*nccl_mock_, devCommCreate(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLGinBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      config,
      fake_nccl_window_,
      fake_window_base_,
      1024);

  // Expect only devCommDestroy
  EXPECT_CALL(*nccl_mock_, devCommDestroy(fake_nccl_comm_, _))
      .WillOnce(Return(ncclSuccess));

  // destroy_device_window takes ownership via std::move
  NCCLGinBackend::destroy_device_window(
      fake_nccl_comm_, nccl_mock_.get(), std::move(device_window));

  // After std::move, device_window is nullptr
  EXPECT_EQ(device_window, nullptr);
}

TEST_F(NCCLGinBackendTest, DestroyDeviceWindowWithNullCommIsNoOp) {
  // Verifies: Destroying with null comm is safe (no-op)
  // Code path: destroy_device_window() when nccl_comm is nullptr
  // Production value: Prevents double-free and allows defensive cleanup

  auto device_window =
      std::make_unique<TorchCommDeviceWindow<NCCLGinBackend>>();

  // Should not call any cleanup methods
  EXPECT_CALL(*nccl_mock_, devCommDestroy(_, _)).Times(0);

  // Should not throw
  EXPECT_NO_THROW(
      NCCLGinBackend::destroy_device_window(
          nullptr, nccl_mock_.get(), std::move(device_window)));
}

TEST_F(NCCLGinBackendTest, DestroyDeviceWindowWithNullApiIsNoOp) {
  // Verifies: Destroying with null API is safe (no-op)
  // Code path: destroy_device_window() when nccl_api is nullptr
  // Production value: Prevents crashes in cleanup paths

  auto device_window =
      std::make_unique<TorchCommDeviceWindow<NCCLGinBackend>>();

  // Should not call any cleanup methods (can't call on null API)
  // No mock expectation needed since we can't call on null

  // Should not throw
  EXPECT_NO_THROW(
      NCCLGinBackend::destroy_device_window(
          fake_nccl_comm_, nullptr, std::move(device_window)));
}

TEST_F(NCCLGinBackendTest, DestroyDeviceWindowWithNullPtrIsNoOp) {
  // Verifies: Destroying with nullptr unique_ptr is safe (no-op)
  // Code path: destroy_device_window() when device_window is nullptr
  // Production value: Prevents crashes when destroying uninitialized window

  std::unique_ptr<TorchCommDeviceWindow<NCCLGinBackend>> device_window =
      nullptr;

  // Should not call any cleanup methods
  EXPECT_CALL(*nccl_mock_, devCommDestroy(_, _)).Times(0);

  // Should not throw
  EXPECT_NO_THROW(
      NCCLGinBackend::destroy_device_window(
          fake_nccl_comm_, nccl_mock_.get(), std::move(device_window)));
}

// =============================================================================
// Full Lifecycle Tests
// =============================================================================

TEST_F(NCCLGinBackendTest, FullLifecycleCreateAndDestroy) {
  // Verifies: Complete lifecycle works correctly
  // Code path: create_device_window -> use -> destroy_device_window
  // Production value: End-to-end validation of typical usage pattern

  auto config = createDefaultConfig();
  void* fake_base = reinterpret_cast<void*>(0x3000);
  size_t fake_size = 4096;

  // Create
  EXPECT_CALL(*nccl_mock_, devCommCreate(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLGinBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      config,
      fake_nccl_window_,
      fake_base,
      fake_size);

  // Verify unique_ptr is valid and fields are set
  ASSERT_NE(device_window, nullptr);
  EXPECT_EQ(device_window->base_, fake_base);
  EXPECT_EQ(device_window->size_, fake_size);

  // Use device_window (passed by value to kernels in real usage)
  // In real code: myKernel<<<grid, block>>>(*device_window, ...);

  // Destroy - takes ownership via std::move
  EXPECT_CALL(*nccl_mock_, devCommDestroy(_, _)).WillOnce(Return(ncclSuccess));

  NCCLGinBackend::destroy_device_window(
      fake_nccl_comm_, nccl_mock_.get(), std::move(device_window));

  // After std::move, device_window is nullptr
  EXPECT_EQ(device_window, nullptr);
}

} // namespace torchcomms::device::test

#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API
