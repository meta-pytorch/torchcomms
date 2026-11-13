// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <exception>

#include <pybind11/pybind11.h>

#include "comms/torchcomms/AssertionError.hpp"

namespace torch::comms {

// pybind11 exception translator that raises AssertionError in Python.
inline void translateAssertionError(std::exception_ptr p) {
  try {
    if (p) {
      std::rethrow_exception(p);
    }
  } catch (const AssertionError& e) {
    PyErr_SetString(PyExc_AssertionError, e.what());
  }
}

} // namespace torch::comms
