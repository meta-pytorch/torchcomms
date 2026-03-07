// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/StoreManager.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_store_manager, m) {
  m.def(
      "_create_tcp_store",
      [](std::chrono::milliseconds timeout) { return createTCPStore(timeout); },
      R"(
      Return a new TCPStore from MASTER_ADDR / MASTER_PORT env vars.
      )",
      py::arg("timeout") = std::chrono::milliseconds(60000),
      py::call_guard<py::gil_scoped_release>());
}
