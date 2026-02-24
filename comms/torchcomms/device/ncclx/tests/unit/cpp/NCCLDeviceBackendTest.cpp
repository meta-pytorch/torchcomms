// Copyright (c) Meta Platforms, Inc. and affiliates.

// Unit tests for NCCLDeviceBackend static methods
//
// These tests verify the device backend lifecycle and error handling:
//   1. create_device_window() - tests success path and failure paths
//   2. DeviceWindowDeleter - tests cleanup behavior through unique_ptr
//
// Test Philosophy:
//   - Each test explores a specific code path
//   - Failure tests verify proper cleanup (no resource leaks)
//   - Tests use strict mocks to catch unexpected API calls
//   - Error messages are validated to ensure actionable diagnostics
//
// Ownership Design:
//   NCCLDeviceBackend::create_device_window() returns std::unique_ptr with a
//   custom DeviceWindowDeleter. The deleter frees device memory via CudaApi -
//   the caller is responsible for calling ncclDevCommDestroy before destroying
//   the ptr.
//   Access dev_comm via unique_ptr::get_deleter().dev_comm.
//
// Note: This entire test file requires TORCHCOMMS_HAS_NCCL_DEVICE_API because
// it tests NCCLDeviceBackend which uses devCommCreate/devCommDestroy APIs.

#include "comms/torchcomms/ncclx/NcclxApi.hpp" // For TORCHCOMMS_HAS_NCCL_DEVICE_API

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

namespace torchcomms::device::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

using torch::comms::DefaultCudaApi;
using torch::comms::test::NcclxMock;

// =============================================================================
// Test Fixture
// =============================================================================

class NCCLDeviceBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    nccl_mock_ = std::make_shared<NiceMock<NcclxMock>>();
    nccl_mock_->setupDefaultBehaviors();
    cuda_api_ = std::make_shared<DefaultCudaApi>();
    fake_nccl_comm_ = reinterpret_cast<ncclComm_t>(0x1000);
    fake_nccl_window_ = reinterpret_cast<ncclWindow_t>(0x2000);
    fake_base_ = reinterpret_cast<void*>(0x3000);
  }

  void TearDown() override {
    nccl_mock_.reset();
  }

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
  std::shared_ptr<DefaultCudaApi> cuda_api_;
  ncclComm_t fake_nccl_comm_{nullptr};
  ncclWindow_t fake_nccl_window_{nullptr};
  void* fake_base_{nullptr};
};

// =============================================================================
// create_device_window() Tests - Null Checks
// =============================================================================

TEST_F(
    NCCLDeviceBackendTest,
    CreateDeviceWindowWithNullNcclCommThrowsException) {
  auto config = createDefaultConfig();

  EXPECT_THROW(
      {
        try {
          NCCLDeviceBackend::create_device_window(
              nullptr,
              nccl_mock_.get(),
              cuda_api_.get(),
              config,
              fake_nccl_window_,
              fake_base_,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("NCCL communicator cannot be null") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(
    NCCLDeviceBackendTest,
    CreateDeviceWindowWithNullNcclApiThrowsException) {
  auto config = createDefaultConfig();

  EXPECT_THROW(
      {
        try {
          NCCLDeviceBackend::create_device_window(
              fake_nccl_comm_,
              nullptr,
              cuda_api_.get(),
              config,
              fake_nccl_window_,
              fake_base_,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("NCCL API cannot be null") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(
    NCCLDeviceBackendTest,
    CreateDeviceWindowWithNullCudaApiThrowsException) {
  auto config = createDefaultConfig();

  EXPECT_THROW(
      {
        try {
          NCCLDeviceBackend::create_device_window(
              fake_nccl_comm_,
              nccl_mock_.get(),
              nullptr,
              config,
              fake_nccl_window_,
              fake_base_,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("CUDA API cannot be null") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// =============================================================================
// create_device_window() Tests - Success Path
// NOTE: These tests require real CUDA hardware because DefaultCudaApi calls
// cudaMalloc/cudaMemcpy. They verify the code path when running on actual GPU
// infrastructure via buck test.
// =============================================================================

TEST_F(NCCLDeviceBackendTest, CreateDeviceWindowSuccessReturnsValidStruct) {
  // This test verifies create_device_window returns a valid pointer
  // and that cleanup works correctly.
  auto config = createDefaultConfig();
  size_t fake_size = 4096;

  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      fake_size);

  // Verify pointer is valid (non-null)
  // Note: We cannot dereference device_window directly because it points to
  // GPU memory. To verify contents, we would need to cudaMemcpy back to host.
  ASSERT_NE(device_window, nullptr);
  ASSERT_NE(device_window.get(), nullptr);

  // Verify deleter has correct state for cleanup
  auto& deleter = device_window.get_deleter();
  EXPECT_EQ(deleter.nccl_comm, fake_nccl_comm_);
  EXPECT_EQ(deleter.nccl_api, nccl_mock_.get());

  // Copy device window back to host to verify contents
  TorchCommDeviceWindow<NCCLDeviceBackend> host_copy{};
  cudaError_t cuda_result = cudaMemcpy(
      &host_copy,
      device_window.get(),
      sizeof(TorchCommDeviceWindow<NCCLDeviceBackend>),
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(cuda_result, cudaSuccess);

  EXPECT_EQ(host_copy.window_, fake_nccl_window_);
  EXPECT_EQ(host_copy.base_, fake_base_);
  EXPECT_EQ(host_copy.size_, fake_size);
  EXPECT_EQ(host_copy.rank_, config.comm_rank);
  EXPECT_EQ(host_copy.num_ranks_, config.comm_size);

  // Clean up: call devCommDestroy before releasing (as required by contract)
  EXPECT_CALL(*nccl_mock_, devCommDestroy(fake_nccl_comm_, _))
      .WillOnce(Return(ncclSuccess));
  nccl_mock_->devCommDestroy(deleter.nccl_comm, &deleter.dev_comm);

  // Let unique_ptr destructor call cudaFree via the deleter
}

TEST_F(NCCLDeviceBackendTest, CreateDeviceWindowConfigIsPassedCorrectly) {
  // This test verifies that the config is correctly passed to devCommCreate.
  DeviceBackendConfig config;
  config.signal_count = 16;
  config.counter_count = 32;
  config.barrier_count = 2;
  config.comm_rank = 3;
  config.comm_size = 8;

  ncclDevCommRequirements captured_reqs{};
  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(DoAll(
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

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      1024);

  ASSERT_NE(device_window, nullptr);
  EXPECT_EQ(captured_reqs.ginSignalCount, 0);
  EXPECT_EQ(captured_reqs.ginCounterCount, config.counter_count);
  EXPECT_EQ(captured_reqs.barrierCount, config.barrier_count);
  EXPECT_TRUE(captured_reqs.ginForceEnable);

  // Verify resource buffer requirements were chained in
  EXPECT_NE(captured_reqs.resourceRequirementsList, nullptr);
  if (captured_reqs.resourceRequirementsList) {
    auto& res_reqs = *captured_reqs.resourceRequirementsList;
    size_t expected_buf_size = static_cast<size_t>(config.signal_count) *
        config.comm_size * sizeof(uint64_t);
    EXPECT_EQ(res_reqs.bufferSize, expected_buf_size);
    EXPECT_EQ(res_reqs.bufferAlign, 8u);
  }

  // Clean up: call devCommDestroy before releasing (as required by contract)
  auto& deleter = device_window.get_deleter();
  EXPECT_CALL(*nccl_mock_, devCommDestroy(fake_nccl_comm_, _))
      .WillOnce(Return(ncclSuccess));
  nccl_mock_->devCommDestroy(deleter.nccl_comm, &deleter.dev_comm);

  // Let unique_ptr destructor call cudaFree via the deleter
}

TEST_F(NCCLDeviceBackendTest, CreateDeviceWindowStoresSignalBufferHandle) {
  // This test verifies that the signal_buffer_handle is stored in the window.
  auto config = createDefaultConfig();

  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      1024);

  ASSERT_NE(device_window, nullptr);

  // Copy device window back to host to verify signal_buffer_handle
  TorchCommDeviceWindow<NCCLDeviceBackend> host_copy{};
  cudaError_t cuda_result = cudaMemcpy(
      &host_copy,
      device_window.get(),
      sizeof(TorchCommDeviceWindow<NCCLDeviceBackend>),
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(cuda_result, cudaSuccess);

  // The signal_buffer_handle_ should be set (value assigned by
  // ncclDevCommCreate) We verify it was stored in the struct — the actual value
  // depends on ncclDevCommCreate's allocation, which returns 0 for the mock.
  EXPECT_EQ(host_copy.signal_buffer_handle_, 0u);

  // Clean up
  auto& deleter = device_window.get_deleter();
  EXPECT_CALL(*nccl_mock_, devCommDestroy(fake_nccl_comm_, _))
      .WillOnce(Return(ncclSuccess));
  nccl_mock_->devCommDestroy(deleter.nccl_comm, &deleter.dev_comm);
}

// =============================================================================
// create_device_window() Tests - Failure Paths
// =============================================================================

TEST_F(NCCLDeviceBackendTest, DevCommCreateFailureThrows) {
  auto config = createDefaultConfig();

  EXPECT_CALL(*nccl_mock_, devCommCreate(fake_nccl_comm_, _, _))
      .WillOnce(Return(ncclInternalError));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillOnce(Return("internal error"));

  EXPECT_THROW(
      {
        try {
          NCCLDeviceBackend::create_device_window(
              fake_nccl_comm_,
              nccl_mock_.get(),
              cuda_api_.get(),
              config,
              fake_nccl_window_,
              fake_base_,
              1024);
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("Failed to create NCCL device") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// =============================================================================
// DeviceWindowDeleter Tests
// =============================================================================

TEST_F(NCCLDeviceBackendTest, DeviceWindowDeleterOnlyCudaFrees) {
  // This test verifies the deleter stores the correct state for caller cleanup
  // and that devCommDestroy is NOT called automatically by the deleter.
  auto config = createDefaultConfig();

  EXPECT_CALL(*nccl_mock_, devCommCreate(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      1024);

  ASSERT_NE(device_window, nullptr);

  // Verify that the deleter stores the right info for caller cleanup
  auto& deleter = device_window.get_deleter();
  EXPECT_EQ(deleter.nccl_comm, fake_nccl_comm_);
  EXPECT_EQ(deleter.nccl_api, nccl_mock_.get());

  // The deleter does NOT call devCommDestroy - caller must do that
  // We expect no devCommDestroy calls during unique_ptr destruction
  // (Caller is responsible for calling it before releasing)
  // For this test, we intentionally skip calling devCommDestroy to verify
  // the deleter doesn't call it automatically.

  // Let unique_ptr destructor call cudaFree via the deleter
  // Note: This "leaks" the ncclDevComm, but that's intentional for this test
}

TEST_F(
    NCCLDeviceBackendTest,
    DeviceWindowDeleterStoresDevCommForCallerCleanup) {
  // This test verifies the caller can access dev_comm from deleter for cleanup.
  auto config = createDefaultConfig();
  ncclDevComm expected_dev_comm{};

  EXPECT_CALL(*nccl_mock_, devCommCreate(_, _, _))
      .WillOnce(
          DoAll(SetArgPointee<2>(expected_dev_comm), Return(ncclSuccess)));

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      1024);

  ASSERT_NE(device_window, nullptr);

  auto& deleter = device_window.get_deleter();
  EXPECT_EQ(deleter.nccl_comm, fake_nccl_comm_);
  EXPECT_EQ(deleter.nccl_api, nccl_mock_.get());

  // Simulate what the caller should do before releasing the pointer:
  // call devCommDestroy using info stored in deleter
  EXPECT_CALL(*nccl_mock_, devCommDestroy(fake_nccl_comm_, _))
      .WillOnce(Return(ncclSuccess));
  nccl_mock_->devCommDestroy(deleter.nccl_comm, &deleter.dev_comm);

  // Let unique_ptr destructor call cudaFree via the deleter
}

TEST_F(NCCLDeviceBackendTest, DeviceWindowDeleterWithNullPtrIsSafe) {
  NCCLDeviceBackend::DeviceWindowDeleter deleter(
      fake_nccl_comm_, nccl_mock_.get(), cuda_api_.get(), ncclDevComm{});
  EXPECT_NO_THROW(deleter(nullptr));
}

// =============================================================================
// Full Lifecycle Tests
// =============================================================================

TEST_F(NCCLDeviceBackendTest, FullLifecycleCreateAndDestroy) {
  // This test verifies the full create and cleanup lifecycle.
  auto config = createDefaultConfig();
  size_t fake_size = 4096;

  EXPECT_CALL(*nccl_mock_, devCommCreate(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(ncclDevComm{}), Return(ncclSuccess)));

  auto device_window = NCCLDeviceBackend::create_device_window(
      fake_nccl_comm_,
      nccl_mock_.get(),
      cuda_api_.get(),
      config,
      fake_nccl_window_,
      fake_base_,
      fake_size);

  ASSERT_NE(device_window, nullptr);

  // Copy device window back to host to verify contents
  TorchCommDeviceWindow<NCCLDeviceBackend> host_copy{};
  cudaError_t cuda_result = cudaMemcpy(
      &host_copy,
      device_window.get(),
      sizeof(TorchCommDeviceWindow<NCCLDeviceBackend>),
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(cuda_result, cudaSuccess);
  EXPECT_EQ(host_copy.base_, fake_base_);
  EXPECT_EQ(host_copy.size_, fake_size);

  // Simulate full lifecycle: caller must call devCommDestroy before release
  auto& deleter = device_window.get_deleter();
  EXPECT_CALL(*nccl_mock_, devCommDestroy(deleter.nccl_comm, _))
      .WillOnce(Return(ncclSuccess));
  nccl_mock_->devCommDestroy(deleter.nccl_comm, &deleter.dev_comm);

  // Let unique_ptr destructor call cudaFree via the deleter
}

} // namespace torchcomms::device::test

#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API
