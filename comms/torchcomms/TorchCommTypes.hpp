// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Work.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <chrono>
#include <variant>

namespace torch {
namespace comms {

// Forward declaration
class TorchComm;
class TorchWork;

using PreMulSumFactorT = std::variant<at::Tensor, double>;

class ReduceOp {
 public:
  // ReduceOp enum for reduction operations
  enum class RedOpType {
    SUM = 0,
    PRODUCT,
    MIN,
    MAX,
    BAND,
    BOR,
    BXOR,
    PREMUL_SUM,
    AVG,
  };

  /* implicit */ ReduceOp(RedOpType type) : type_(type), factor_(std::nullopt) {
    TORCH_INTERNAL_ASSERT(
        type != RedOpType::PREMUL_SUM, "PREMUL_SUM needs a factor");
  }

  static ReduceOp make_nccl_premul_sum(const PreMulSumFactorT& factor) {
    return ReduceOp(RedOpType::PREMUL_SUM, factor);
  }

  // The const static ReduceOp objects are for python bindings, for torchcomms
  // internal *static* function, it is better to use the RedOpType enum directly
  // to avoid static initialization order fiasco.
  // @lint-ignore-every CLANGTIDY NonPodStaticDeclaration
  static const ReduceOp SUM;
  static const ReduceOp PRODUCT;
  static const ReduceOp MIN;
  static const ReduceOp MAX;
  static const ReduceOp BAND;
  static const ReduceOp BOR;
  static const ReduceOp BXOR;
  static const ReduceOp AVG;

  ReduceOp(const ReduceOp& other) = default;
  ReduceOp& operator=(const ReduceOp& other) = delete;

  ReduceOp(ReduceOp&& other) noexcept = default;
  ReduceOp& operator=(ReduceOp&& other) noexcept = delete;
  ~ReduceOp() = default;

  operator RedOpType() const {
    return type_;
  }

  RedOpType type() const {
    return type_;
  }

  const std::optional<const PreMulSumFactorT>& factor() const {
    return factor_;
  }

 private:
  ReduceOp() = default;
  ReduceOp(RedOpType type, const PreMulSumFactorT& factor)
      : type_(type), factor_(factor) {}

  RedOpType type_{RedOpType::SUM};
  std::optional<const PreMulSumFactorT> factor_{std::nullopt};
};

// CmpOp enum for comparison operations in Signal
enum class SignalCmpOp {
  EQ = 0,
  GE,
  LE,
};

// Default timeout for collective operations.  It can be overridden during
// TorchComm creation or during each collective operation.
constexpr std::chrono::milliseconds kDefaultTimeout = std::chrono::seconds(600);
constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds(0);

} // namespace comms
} // namespace torch
