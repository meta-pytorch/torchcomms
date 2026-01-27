// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test - NCCL GIN Backend

#include "DeviceApiTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

std::unique_ptr<TorchCommTestWrapper> DeviceApiTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void DeviceApiTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping Device API tests (RUN_DEVICE_API_TEST not set)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();
}

void DeviceApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool DeviceApiTest::checkIfSkip() {
  // Check RUN_DEVICE_API_TEST env var
  const char* device_api_env = getenv("RUN_DEVICE_API_TEST");
  if (!device_api_env) {
    return true; // skip if not set
  }
  std::string val(device_api_env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val != "1" && val != "true") {
    return true; // skip if not enabled
  }
  return false;
}

at::Tensor DeviceApiTest::createTestTensor(
    int64_t count,
    at::ScalarType dtype) {
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  return at::ones({count}, options) * (rank_ + 1);
}

std::string DeviceApiTest::getDtypeName(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "float32";
    case at::kDouble:
      return "float64";
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    case at::kInt:
      return "int32";
    case at::kLong:
      return "int64";
    default:
      return "unknown";
  }
}

// Test device window creation and basic properties
void DeviceApiTest::testDeviceWindowCreation(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window Creation with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Allocate window buffer
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win = dynamic_cast<torch::comms::TorchCommWindowNCCLX*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLX";

  // Get device window
  auto* dev_win = win->get_device_window();
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testLocalBufferRegistration(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Local Buffer Registration with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Allocate window buffer
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win = dynamic_cast<torch::comms::TorchCommWindowNCCLX*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLX";

  // Create and register local source buffer
  at::Tensor src_tensor = createTestTensor(count, dtype);
  auto src_buf = win->register_local_buffer(src_tensor);

  // Verify buffer properties
  ASSERT_NE(src_buf.base_ptr, nullptr) << "Buffer base_ptr should not be null";
  ASSERT_GT(src_buf.size, 0) << "Buffer size should be positive";
  ASSERT_NE(src_buf.backend_window, nullptr)
      << "Buffer backend_window should not be null";

  // Cleanup
  win->deregister_local_buffer(src_buf);
  base_win->tensor_deregister();
  base_win.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testDeviceWindowWithSignals(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window with Signals count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Allocate window buffer
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win = dynamic_cast<torch::comms::TorchCommWindowNCCLX*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLX";

  // Get device window with signals
  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testDeviceWindowWithCounters(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window with Counters count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Allocate window buffer
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win = dynamic_cast<torch::comms::TorchCommWindowNCCLX*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLX";

  // Get device window with signals and counters
  int signal_count = num_ranks_;
  int counter_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, counter_count, 1);
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();

  torchcomm_->barrier(false);
}

// GTest test cases
TEST_F(DeviceApiTest, DeviceWindowCreationFloat) {
  testDeviceWindowCreation(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowCreationHalf) {
  testDeviceWindowCreation(1024, at::kHalf);
}

TEST_F(DeviceApiTest, LocalBufferRegistrationFloat) {
  testLocalBufferRegistration(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowWithSignalsFloat) {
  testDeviceWindowWithSignals(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowWithCountersFloat) {
  testDeviceWindowWithCounters(1024, at::kFloat);
}
