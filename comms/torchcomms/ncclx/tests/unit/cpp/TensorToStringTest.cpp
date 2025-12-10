// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class TensorToStringTest : public ::testing::Test {};

TEST_F(TensorToStringTest, EmptyTensor) {
  auto tensor = at::empty({0}, at::kInt);
  EXPECT_EQ(tensorToString(tensor), "[]");
}

TEST_F(TensorToStringTest, ScalarTensor) {
  auto tensor = at::scalar_tensor(42, at::kInt);
  EXPECT_EQ(tensorToString(tensor), "42");
}

TEST_F(TensorToStringTest, OneDimensional) {
  auto tensor = at::arange(1, 6, at::kInt);
  EXPECT_EQ(tensorToString(tensor), "[1, 2, 3, 4, 5]");
}

TEST_F(TensorToStringTest, TwoDimensional) {
  auto tensor = at::arange(1, 7, at::kInt).reshape({2, 3});
  EXPECT_EQ(tensorToString(tensor), "[[1, 2, 3], [4, 5, 6]]");
}

TEST_F(TensorToStringTest, ThreeDimensional) {
  auto tensor = at::arange(24, at::kInt).reshape({2, 3, 4});
  std::string expected =
      "[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], "
      "[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]";
  EXPECT_EQ(tensorToString(tensor), expected);
}

TEST_F(TensorToStringTest, FloatTensor) {
  auto tensor = at::arange(3, at::kFloat) + 1.5;
  EXPECT_EQ(tensorToString(tensor), "[1.5, 2.5, 3.5]");
}

TEST_F(TensorToStringTest, SingleElement) {
  auto tensor = at::ones({1}, at::kInt) * 42;
  EXPECT_EQ(tensorToString(tensor), "[42]");
}

TEST_F(TensorToStringTest, SingleElement2D) {
  auto tensor = at::ones({1, 1}, at::kInt) * 42;
  EXPECT_EQ(tensorToString(tensor), "[[42]]");
}

TEST_F(TensorToStringTest, BoolTensor) {
  auto options = torch::TensorOptions().dtype(torch::kBool);
  torch::Tensor tensor = torch::tensor({{true, false}, {false, true}}, options);

  // bool tensor is printed as item<double>
  EXPECT_EQ(tensorToString(tensor), "[[1, 0], [0, 1]]");
}

TEST_F(TensorToStringTest, DoubleTensor) {
  auto tensor = at::arange(3, at::kDouble) + 1.5;
  EXPECT_EQ(tensorToString(tensor), "[1.5, 2.5, 3.5]");
}

TEST_F(TensorToStringTest, FloatCudaTensor) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat).device(at::kCUDA);
  auto tensor = at::arange(3, options) + at::scalar_tensor(1.5f, options);
  EXPECT_EQ(tensorToString(tensor), "[1.5, 2.5, 3.5]");
}

TEST_F(TensorToStringTest, BFloat16CudaTensor) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(at::kCUDA);
  auto tensor = at::arange(3, options) + at::scalar_tensor(1.5f, options);
  // Values should be approximately [1.5, 2.5, 3.5] in string form
  EXPECT_EQ(tensorToString(tensor), "[1.5, 2.5, 3.5]");
}

TEST_F(TensorToStringTest, HalfCudaTensor) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  auto options = torch::TensorOptions().dtype(torch::kHalf).device(at::kCUDA);
  auto tensor = at::arange(3, options) + at::scalar_tensor(1.5f, options);
  EXPECT_EQ(tensorToString(tensor), "[1.5, 2.5, 3.5]");
}

TEST_F(TensorToStringTest, ComplexFloatTensor) {
  auto options = torch::TensorOptions().dtype(torch::kComplexFloat);
  // Create 1D complex tensor: [1+2j, 3+4j, -5-6j]
  auto real = at::tensor({1.0f, 3.0f, -5.0f}, at::kFloat);
  auto imag = at::tensor({2.0f, 4.0f, -6.0f}, at::kFloat);
  auto tensor = at::complex(real, imag).to(options.dtype());
  EXPECT_EQ(tensorToString(tensor), "[(1+2j), (3+4j), (-5-6j)]");
}

TEST_F(TensorToStringTest, ComplexScalarTensor) {
  // Single complex scalar: (7-3j)
  auto z = at::complex(
      at::scalar_tensor(7.0, at::kDouble),
      at::scalar_tensor(-3.0, at::kDouble));
  EXPECT_EQ(tensorToString(z), "(7-3j)");
}

TEST_F(TensorToStringTest, ComplexEmptyTensor) {
  auto tensor = at::empty({0}, at::kComplexFloat);
  EXPECT_EQ(tensorToString(tensor), "[]");
}

TEST_F(TensorToStringTest, ComplexCudaTensorFloat) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  auto options =
      torch::TensorOptions().dtype(torch::kComplexFloat).device(at::kCUDA);

  auto real = at::arange(
      3,
      torch::TensorOptions().dtype(torch::kFloat).device(at::kCUDA)); // [0,1,2]
  auto imag = at::ones_like(real) * 0.5f; // [0.5,0.5,0.5]
  auto tensor = at::complex(real, imag).to(options.dtype());
  EXPECT_EQ(tensorToString(tensor), "[(0+0.5j), (1+0.5j), (2+0.5j)]");
}

TEST_F(TensorToStringTest, ComplexCudaTensorDouble) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  auto options =
      torch::TensorOptions().dtype(torch::kComplexDouble).device(at::kCUDA);
  auto elemOptions =
      torch::TensorOptions().dtype(torch::kDouble).device(at::kCUDA);

  auto real = at::tensor({-1.0, 0.0, 2.0}, elemOptions);
  auto imag = at::tensor({2.0, -3.0, 0.0}, elemOptions);
  auto tensor = at::complex(real, imag).to(options.dtype());
  EXPECT_EQ(tensorToString(tensor), "[(-1+2j), (0-3j), (2+0j)]");
}
