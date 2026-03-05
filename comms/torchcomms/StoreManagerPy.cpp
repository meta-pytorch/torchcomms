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
      "_get_store",
      [](const std::string& backend_name,
         const std::string& name,
         std::chrono::milliseconds timeout) {
        return StoreManager::get().getStore(backend_name, name, timeout);
      },
      R"(
      Return a new store object that's unique to the given backend and
      communicator name.
      )",
      py::arg("backend_name"),
      py::arg("name"),
      py::arg("timeout") = std::chrono::milliseconds(60000),
      py::call_guard<py::gil_scoped_release>());
}
