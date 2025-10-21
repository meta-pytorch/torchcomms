// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/gloo/TorchCommGloo.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_gloo, m) {
  m.doc() = "Gloo specific python bindings for TorchComm";

  py::class_<TorchCommGloo, std::shared_ptr<TorchCommGloo>>(m, "TorchCommGloo");
}
