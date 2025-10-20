// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/rccl/TorchCommRCCL.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_rccl, m) {
  m.doc() = "RCCL specific python bindings for TorchComm";

  py::class_<TorchCommRCCL, std::shared_ptr<TorchCommRCCL>>(m, "TorchCommRCCL");
}
