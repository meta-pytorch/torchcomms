// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"
#include "torch/csrc/distributed/c10d/TCPStore.hpp" // @manual=//caffe2:torch-cpp-cpu

#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"

using namespace torch::comms;

void configureManualRankSize() {
  auto ranksize_query_method_env =
      std::getenv("TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD");

  std::string ranksize_query_method =
      ranksize_query_method_env ? ranksize_query_method_env : "";

  if (ranksize_query_method == "manual") {
    // Get rank and size from environment variables
    auto [rank, size] = getRankAndSize();
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);
    // unsetenv("TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD");

    TC_LOG(INFO) << "Using manual rank and size: " << rank << ", " << size;
  }
}

std::unordered_map<std::string, std::string> getHintsFromEnv() {
  std::unordered_map<std::string, std::string> hints;
  auto fast_init_mode_env = std::getenv("TEST_FAST_INIT_MODE");
  if (fast_init_mode_env) {
    hints["fastInitMode"] = std::string(fast_init_mode_env);
  }
  return hints;
}

std::string getDtypeName(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "Float";
    case at::kInt:
      return "Int";
    case at::kHalf:
      return "Half";
    case at::kChar:
      return "SignedChar";
    case at::kBFloat16:
      return "BFloat16";
    case at::kDouble:
      return "Double";
    default:
      return "Unknown";
  }
}

std::string getOpName(torch::comms::ReduceOp op) {
  switch (op) {
    case torch::comms::ReduceOp::RedOpType::SUM:
      return "Sum";
    case torch::comms::ReduceOp::RedOpType::PRODUCT:
      return "Product";
    case torch::comms::ReduceOp::RedOpType::MIN:
      return "Min";
    case torch::comms::ReduceOp::RedOpType::MAX:
      return "Max";
    case torch::comms::ReduceOp::RedOpType::BAND:
      return "BAnd";
    case torch::comms::ReduceOp::RedOpType::BOR:
      return "BOr";
    case torch::comms::ReduceOp::RedOpType::BXOR:
      return "BXor";
    case torch::comms::ReduceOp::RedOpType::PREMUL_SUM:
      return "PremulSum";
    case torch::comms::ReduceOp::RedOpType::AVG:
      return "Avg";
    default:
      return "Unknown";
  }
}

std::tuple<int, int> getRankAndSize() {
  // Try OpenMPI environment variables first
  const char* ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
  const char* ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");

  if (ompi_rank && ompi_size) {
    return {std::stoi(ompi_rank), std::stoi(ompi_size)};
  }

  // Try TorchComms environment variables
  const char* rank = std::getenv("TORCHCOMM_RANK");
  const char* size = std::getenv("TORCHCOMM_SIZE");

  if (rank && size) {
    return {std::stoi(rank), std::stoi(size)};
  }

  // Try SLURM environment variables first
  const char* slurm_rank = std::getenv("SLURM_PROCID");
  const char* slurm_size = std::getenv("SLURM_NTASKS");

  if (slurm_rank && slurm_size) {
    return {std::stoi(slurm_rank), std::stoi(slurm_size)};
  }

  // Try PMI environment variables
  const char* pmi_rank = std::getenv("PMI_RANK");
  const char* pmi_size = std::getenv("PMI_SIZE");

  if (pmi_rank && pmi_size) {
    return {std::stoi(pmi_rank), std::stoi(pmi_size)};
  }

  // Try torchrun environment variables
  const char* torchrun_rank = std::getenv("RANK");
  const char* torchrun_size = std::getenv("WORLD_SIZE");

  if (torchrun_rank && torchrun_size) {
    return {std::stoi(torchrun_rank), std::stoi(torchrun_size)};
  }

  const auto [rank, size] = query_pals_ranksize();
  if (rank > -1 && size > 0) {
    return {rank, size};
  }

  throw std::runtime_error(
      "Could not determine rank or world size from environment variables.");
}

c10::intrusive_ptr<c10d::Store> createStore() {
  configureManualRankSize();

  static int next_store_id = 0;
  next_store_id += 1;

  return StoreManager::get().getStore(
      "comms_test",
      fmt::format("store_{}", next_store_id),
      std::chrono::milliseconds(60000));
}

void destroyStore(
    c10::intrusive_ptr<c10d::Store>&& store,
    std::shared_ptr<torch::comms::TorchComm> torchcomm) {
  // Move the store to a local variable that will be destroyed when this
  // function exits
  auto local_store = std::move(store);

  // Reset the store to delete its reference
  local_store.reset();

  // Call barrier on torchcomm to ensure all processes have deleted their store
  // objects
  torchcomm->barrier(false);
}

void verifyTensorEquality(
    const at::Tensor& output,
    const at::Tensor& expected,
    const std::string& description) {
  // Skip verification if tensor is empty
  if (output.numel() == 0) {
    return;
  }

  // Check that output tensor is on GPU
  EXPECT_TRUE(output.device().is_cpu())
      << "Output tensor should be on CPU for " << description;

  // Check that expected tensor is on CPU
  EXPECT_TRUE(expected.device().is_cpu())
      << "Expected tensor should be on CPU for " << description;

  // Check that tensors have the same shape
  EXPECT_TRUE(output.sizes().equals(expected.sizes()))
      << "Tensor shapes don't match for " << description;

  // Check that tensors have the same dtype
  EXPECT_EQ(output.scalar_type(), expected.scalar_type())
      << "Tensor dtypes don't match for " << description;

  // Copy output tensor to CPU
  at::Tensor output_cpu = output.cpu();

  // Different verification based on dtype
  at::ScalarType dtype = output_cpu.scalar_type();

  if (dtype == at::kFloat) {
    // For float tensors, check if they are close enough
    at::Tensor diff = at::abs(output_cpu - expected);
    bool all_close = diff.max().item<float>() < 1e-5;
    EXPECT_TRUE(all_close) << "Tensors are not close enough for "
                           << description;

    // If not all close, print individual differences for debugging
    if (!all_close) {
      // Find indices where difference is significant
      at::Tensor significant_diff = diff > 1e-5;
      at::Tensor indices = significant_diff.nonzero();

      // Print up to 10 differences
      int num_diffs = std::min(10, static_cast<int>(indices.size(0)));
      for (int i = 0; i < num_diffs; i++) {
        at::Tensor idx = indices[i];
        int flat_idx = idx.item<int>();
        EXPECT_NEAR(
            output_cpu.flatten()[flat_idx].item<float>(),
            expected.flatten()[flat_idx].item<float>(),
            1e-5)
            << "Difference at index " << flat_idx;
      }
    }
  } else {
    // For integer types, check exact equality
    bool equal = at::all(output_cpu.eq(expected)).item<bool>();
    EXPECT_TRUE(equal) << "Tensors are not equal for " << description;

    // If not equal, print individual differences for debugging
    if (!equal) {
      // Find indices where values differ
      at::Tensor diff_indices = (output_cpu != expected).nonzero();

      // Print up to 10 differences
      int num_diffs = std::min(10, static_cast<int>(diff_indices.size(0)));
      for (int i = 0; i < num_diffs; i++) {
        at::Tensor idx = diff_indices[i];
        int flat_idx = idx.item<int>();

        if (dtype == at::kInt) {
          EXPECT_EQ(
              output_cpu.flatten()[flat_idx].item<int>(),
              expected.flatten()[flat_idx].item<int>())
              << "Difference at index " << flat_idx;
        } else if (dtype == at::kChar) {
          EXPECT_EQ(
              output_cpu.flatten()[flat_idx].item<signed char>(),
              expected.flatten()[flat_idx].item<signed char>())
              << "Difference at index " << flat_idx;
        }
      }
    }
  }
}

void verifyTensorEquality(
    const at::Tensor& output,
    const double expected_value,
    const std::string& description) {
  // Skip verification if tensor is empty
  if (output.numel() == 0) {
    return;
  }

  // Create expected tensor with the same size and dtype as output, filled with
  // expected_value
  at::Tensor expected = at::full_like(output.cpu(), expected_value);

  // Call the original verifyTensorEquality function
  verifyTensorEquality(output, expected, description);
}

std::string tensorToString(const at::Tensor& tensor) {
  std::ostringstream oss;
  if (tensor.numel() == 0) {
    oss << "[]";
    return oss.str();
  }

  // Check if tensor is complex type
  const auto isComplex = tensor.is_complex();
  // Convert tensor to appropriate type for item() extraction
  // Types that don't support direct item<double>() need conversion
  at::Tensor cpu_tensor = tensor.cpu();
  at::Tensor flat;

  if (isComplex) {
    // For complex types, convert to ComplexDouble for consistent access
    flat = cpu_tensor.to(at::kComplexDouble).flatten();
  } else {
    const auto unsafeToDouble = cpu_tensor.scalar_type() == at::kHalf ||
        cpu_tensor.scalar_type() == at::kBFloat16 ||
        cpu_tensor.scalar_type() == at::kBool;
    flat = unsafeToDouble ? cpu_tensor.to(at::kDouble).flatten()
                          : cpu_tensor.flatten();
  }

  // Helper lambda to format a single element
  auto formatElement = [&oss, &flat, isComplex](int64_t i) {
    if (isComplex) {
      auto val = flat[i].item<c10::complex<double>>();
      oss << "(" << val.real() << (val.imag() >= 0 ? "+" : "") << val.imag()
          << "j)";
    } else {
      oss << flat[i].item<double>();
    }
  };

  if (tensor.dim() == 0) {
    // Scalar tensor - print its value
    formatElement(0);
    return oss.str();
  }

  int64_t numel = flat.numel();
  int64_t ndim = tensor.dim();

  // Calculate stride for each dimension (product of sizes from dim d to end)
  std::vector<int64_t> strides(ndim);
  strides[ndim - 1] = tensor.size(ndim - 1);
  for (int64_t d = ndim - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * tensor.size(d);
  }

  for (int64_t i = 0; i < numel; ++i) {
    // Opening brackets (from outermost to innermost)
    for (int64_t d = 0; d < ndim; ++d) {
      if (i % strides[d] == 0) {
        oss << "[";
      }
    }

    formatElement(i);

    // Closing brackets (from innermost to outermost)
    for (int64_t d = ndim - 1; d >= 0; --d) {
      if ((i + 1) % strides[d] == 0) {
        oss << "]";
      }
    }

    if (i < numel - 1) {
      oss << ", ";
    }
  }

  return oss.str();
}

TorchCommTestWrapper::TorchCommTestWrapper(
    c10::intrusive_ptr<c10d::Store> store) {
  configureManualRankSize();

  c10::Device device = getDevice();

  // Get backend from TEST_BACKEND environment variable, throw if not set
  const char* test_backend_env = std::getenv("TEST_BACKEND");
  if (!test_backend_env) {
    throw std::runtime_error("TEST_BACKEND environment variable is not set");
  }
  std::string backend = test_backend_env;

  static int next_comm_id = 0;
  next_comm_id += 1;

  torch::comms::CommOptions options;
  options.store = store;
  options.hints = getHintsFromEnv();
  torchcomm_ = torch::comms::new_comm(
      backend, device, fmt::format("comms_test_{}", next_comm_id), options);

  // Release our reference to the store object
  if (options.store != nullptr) {
    options.store.reset();
  }
}
