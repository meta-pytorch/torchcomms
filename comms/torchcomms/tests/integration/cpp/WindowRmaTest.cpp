// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "WindowRmaTest.hpp"

#include <gtest/gtest.h>
#include <ifaddrs.h>
#include <algorithm>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> WindowRmaTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void WindowRmaTest::SetUp() {
  // NCCLX Window RMA requires NCCL_CTRAN_ENABLE=1
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();
}

void WindowRmaTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

bool WindowRmaTest::checkIfSkip() {
  // Check 1: beth0 NIC exists (OSS env requirements)
  struct ifaddrs* ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    return true; // skip if can't get interface info
  }
  bool nic_found = false;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_name &&
        std::string(ifa->ifa_name).find("beth0") != std::string::npos) {
      nic_found = true;
      break;
    }
  }
  freeifaddrs(ifaddr);
  if (!nic_found) {
    return true; // skip if NIC not found
  }

  // Check 2: NCCL_CTRAN_ENABLE is "1" or "true"
  const char* ctran_env = getenv("NCCL_CTRAN_ENABLE");
  if (!ctran_env) {
    return true; // skip if env not set
  }
  std::string val(ctran_env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val != "1" && val != "true") {
    return true; // skip if not enabled
  }

  // Check 3: TEST_BACKEND is "ncclx"
  const char* backend_env = getenv("TEST_BACKEND");
  if (!backend_env || std::string(backend_env) != "ncclx") {
    return true; // skip if backend is not ncclx
  }

  // All conditions met, don't skip
  return false;
}

// Test function for basic window allocation & put
void WindowRmaTest::testWindowPutBasic(
    int count,
    at::ScalarType dtype,
    bool async_op,
    bool signal,
    bool async_signal) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Window Put with count=" << count
                           << " and dtype=" << getDtypeName(dtype));
  // Create tensor with different values based on rank
  at::Tensor input_tensor = createWindowRmaTensor(rank_, count, dtype);
  auto current_stream = at::cuda::getCurrentCUDAStream(device_index_);

  // Call window_allocate
  auto win = torchcomm_->window_allocate(
      input_tensor.numel() * input_tensor.element_size() * num_ranks_);
  // call barrier to ensure all ranks have allocated the window
  torchcomm_->barrier(false);

  auto dst_rank = (rank_ + 1) % num_ranks_;
  auto src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Put the tensor to the Window of the next rank using the current stream if
  // async_op is false, otherwise use the internal op_stream
  auto work = win->put(input_tensor, dst_rank, dst_rank * count, async_op);
  if (async_op) {
    // register async op to current stream if async_op is True
    work->wait();
  }

  if (signal) {
    // call signal on current stream to notify remote rank that the put is
    // complete
    auto signal_work = win->signal(dst_rank, async_signal);
    auto wait_signal_work = win->waitSignal(src_rank, async_signal);
    if (async_signal) {
      // register async Signal op to current stream since it is launched on
      // the internal op_stream
      signal_work->wait();
      // register async WaitSignal op to current stream since it is launched on
      // the internal wait_stream
      wait_signal_work->wait();
    }
    // synchronize current stream to ensure signal is complete
    current_stream.synchronize();
  } else {
    // call barrier on current stream to ensure put is complete
    torchcomm_->barrier(false);
  }

  auto result_tensor = win->getTensor(
      rank_, {input_tensor.sizes()}, input_tensor.scalar_type(), rank_ * count);

  // Verify results
  verifyWindowRmaResults(result_tensor.cpu(), src_rank);

  win.reset();
}

void WindowRmaTest::testWindowCpuPut(
    int count,
    at::ScalarType dtype,
    bool async_op,
    bool signal,
    bool async_signal) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Window Put with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create tensor with different values based on rank
  std::vector<int64_t> tensor_size = {count};
  auto current_stream = at::cuda::getCurrentCUDAStream(device_index_);

  auto dst_rank = (rank_ + 1) % num_ranks_;
  auto src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  auto input_tensor = at::full(
      tensor_size,
      rank_ + 100,
      at::TensorOptions().dtype(dtype).device(at::kCUDA));

  auto target_tensor = at::full(
      tensor_size,
      src_rank + 100,
      at::TensorOptions().dtype(dtype).device(at::kCPU));

  torchcomm_->barrier(false);

  // Call window_allocate
  auto win = torchcomm_->window_allocate(
      input_tensor.numel() * input_tensor.element_size() * num_ranks_, true);
  // call barrier to ensure all ranks have allocated the window
  torchcomm_->barrier(false);

  auto work = win->put(input_tensor, dst_rank, dst_rank * count, async_op);
  if (async_op) {
    // register async op to current stream if async_op is True
    work->wait();
  }

  if (signal) {
    // call signal on current stream to notify remote rank that the put is
    // complete
    auto signal_work = win->signal(dst_rank, async_signal);
    auto wait_signal_work = win->waitSignal(src_rank, async_signal);
    if (async_signal) {
      // register async Signal op to current stream since it is launched on
      // the internal op_stream
      signal_work->wait();
      // register async WaitSignal op to current stream since it is launched on
      // the internal wait_stream
      wait_signal_work->wait();
    }
    // synchronize current stream to ensure signal is complete
    current_stream.synchronize();
  } else {
    // call barrier on current stream to ensure put is complete
    torchcomm_->barrier(false);
    // [TODO]: wait for barrier to accomplish since we are checking on CPU
    // buffer,
    // we should add some feature in TorchCommWindow to help aovid this if CPU
    // buffer is used.
    current_stream.synchronize();
  }

  auto output_tensor = win->getTensor(
      rank_, {tensor_size}, input_tensor.scalar_type(), rank_ * count);

  CHECK(output_tensor.device().type() == at::kCPU);

  torchcomm_->barrier(false);

  // Verify results
  verifyTensorEquality(output_tensor, target_tensor, "WindowRmaTest");
  win.reset();
}

// Helper function to create tensor for broadcast
at::Tensor WindowRmaTest::createWindowRmaTensor(
    int value,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor tensor;

  // Initialize tensor based on dtype
  if (dtype == at::kFloat) {
    tensor = at::ones({count}, options) * static_cast<float>(value);
  } else if (dtype == at::kInt) {
    tensor = at::ones({count}, options) * static_cast<int>(value);
  } else if (dtype == at::kChar) {
    tensor = at::ones({count}, options) * static_cast<signed char>(value);
  }

  return tensor;
}

// Helper function to verify results
void WindowRmaTest::verifyWindowRmaResults(
    const at::Tensor& tensor,
    int value) {
  // Use verifyTensorEquality to compare tensor with expected tensor
  std::string description = "Window RMA with value " + std::to_string(value);
  verifyTensorEquality(tensor, value, description);
}

TEST_P(WindowRmaTest, WindowPutBasic) {
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping NCCLX-CTRAN-only window tests";
  }
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  bool async_op = std::get<2>(GetParam());
  bool signal = std::get<3>(GetParam());
  bool async_signal = std::get<4>(GetParam());
  testWindowPutBasic(count, dtype, async_op, signal, async_signal);
}

TEST_P(WindowRmaTest, WindowCpuBuf) {
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping NCCLX-CTRAN-only window tests";
  }
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  bool async_op = std::get<2>(GetParam());
  bool signal = std::get<3>(GetParam());
  bool async_signal = std::get<4>(GetParam());
  testWindowCpuPut(count, dtype, async_op, signal, async_signal);
}

INSTANTIATE_TEST_SUITE_P(
    WindowRmaTestParams,
    WindowRmaTest,
    ::testing::Combine(
        ::testing::Values(4, 1024, 1024 * 1024),
        ::testing::Values(at::kFloat, at::kInt, at::kChar),
        ::testing::Values(true, false),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),

    [](const ::testing::TestParamInfo<
        std::tuple<int, at::ScalarType, bool, bool, bool>>& info) {
      int count = std::get<0>(info.param);
      at::ScalarType dtype = std::get<1>(info.param);
      std::string async_op = std::get<2>(info.param) ? "asyncOp" : "syncOp";
      std::string signal = std::get<3>(info.param) ? "signal" : "noSignal";
      std::string async_signal =
          std::get<4>(info.param) ? "asyncSignal" : "syncSignal";
      return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype) +
          "_" + async_op + "_" + signal + "_" + async_signal;
    });

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
