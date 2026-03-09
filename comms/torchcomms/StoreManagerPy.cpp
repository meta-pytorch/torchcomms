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
      "_create_prefix_store",
      [](const std::string& prefix, std::chrono::milliseconds timeout) {
        return createPrefixStore(prefix, timeout);
      },
      R"(
      Return a new PrefixStore wrapping a TCPStore from MASTER_ADDR / MASTER_PORT env vars.
      )",
      py::arg("prefix"),
      py::arg("timeout") = std::chrono::milliseconds(60000),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "_dup_prefix_store",
      [](const std::string& prefix,
         const c10::intrusive_ptr<c10d::Store>& bootstrapStore,
         std::chrono::milliseconds timeout) {
        return dupPrefixStore(prefix, bootstrapStore, timeout);
      },
      R"(
      Return a new PrefixStore wrapping an independent TCPStore,
      using bootstrapStore only to exchange connection info.
      )",
      py::arg("prefix"),
      py::arg("bootstrap_store"),
      py::arg("timeout") = std::chrono::milliseconds(60000),
      py::call_guard<py::gil_scoped_release>());
}
