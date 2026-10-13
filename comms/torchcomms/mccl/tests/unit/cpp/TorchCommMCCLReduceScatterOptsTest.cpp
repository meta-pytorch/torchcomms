// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "comms/mccl/tests/MockMcclComm.h"
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/mccl/tests/unit/cpp/MockCudaApi.h"

using ::testing::NiceMock;

namespace torch::comms::test {

class GetReduceScatterOptsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_comm_ = std::make_unique<::mccl::testing::MockMcclComm>();
  }

  std::unique_ptr<::mccl::testing::MockMcclComm> mock_comm_;
};

TEST_F(GetReduceScatterOptsTest, MismatchedDtypesThrows) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), /*rank=*/0, /*size=*/4);
  mccl->cuda_api_ = std::make_shared<NiceMock<MockCudaApi>>();

  auto inputTensor = at::zeros(
      {4096}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputTensor = at::zeros(
      {1024}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  ReduceScatterSingleOptions options;
  EXPECT_THROW(
      mccl->getReduceScatterOpts(
          outputTensor, inputTensor, 0, nullptr, ReduceOp::SUM, options),
      std::runtime_error);
}

TEST_F(GetReduceScatterOptsTest, WrongSizeRatioThrows) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), /*rank=*/0, /*size=*/4);
  mccl->cuda_api_ = std::make_shared<NiceMock<MockCudaApi>>();

  // input has 2048 elements, output has 1024; ratio is 2, but commSize is 4
  auto inputTensor = at::zeros(
      {2048}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputTensor = at::zeros(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  ReduceScatterSingleOptions options;
  EXPECT_THROW(
      mccl->getReduceScatterOpts(
          outputTensor, inputTensor, 0, nullptr, ReduceOp::SUM, options),
      std::runtime_error);
}

TEST_F(GetReduceScatterOptsTest, CpuTensorThrows) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), /*rank=*/0, /*size=*/4);
  mccl->cuda_api_ = std::make_shared<NiceMock<MockCudaApi>>();

  auto inputTensor = at::zeros({4096}, at::TensorOptions().dtype(at::kFloat));
  auto outputTensor = at::zeros({1024}, at::TensorOptions().dtype(at::kFloat));

  ReduceScatterSingleOptions options;
  EXPECT_ANY_THROW(mccl->getReduceScatterOpts(
      outputTensor, inputTensor, 0, nullptr, ReduceOp::SUM, options));
}

TEST_F(GetReduceScatterOptsTest, ValidInputsProducesCorrectOpts) {
  auto mccl = std::make_shared<TorchCommMCCL>(
      std::move(mock_comm_), /*rank=*/0, /*size=*/4);
  mccl->cuda_api_ = std::make_shared<NiceMock<MockCudaApi>>();

  auto inputTensor = at::zeros(
      {4096}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputTensor = at::zeros(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  ReduceScatterSingleOptions options;
  options.timeout = std::chrono::milliseconds(5000);
  options.hints = {{"key1", "val1"}};

  auto opts = mccl->getReduceScatterOpts(
      outputTensor, inputTensor, 0, nullptr, ReduceOp::SUM, options);

  EXPECT_EQ(opts.dataSend, inputTensor.data_ptr());
  EXPECT_EQ(opts.dataRecv, outputTensor.data_ptr());
  EXPECT_EQ(opts.numElements, 1024);
  EXPECT_EQ(opts.dataType, commDataType_t::commFloat32);
  EXPECT_EQ(opts.reduceOpType, commRedOp_t::commSum);
  EXPECT_EQ(opts.timeout, std::chrono::milliseconds(5000));
  EXPECT_EQ(opts.stream, nullptr);
  EXPECT_EQ(opts.kvPairs, options.hints);
}

} // namespace torch::comms::test
