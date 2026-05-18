// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/nccl/TorchCommNCCL.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_nccl, m, py::mod_gil_not_used()) {
  m.doc() = "NCCL specific python bindings for TorchComm";

  py::class_<TorchCommNCCL, TorchCommBackend, std::shared_ptr<TorchCommNCCL>>(
      m, "TorchCommNCCL");
}
