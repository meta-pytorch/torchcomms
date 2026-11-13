// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdexcept>

namespace torch::comms {

// Raised in Python as AssertionError instead of RuntimeError.
class AssertionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

} // namespace torch::comms
