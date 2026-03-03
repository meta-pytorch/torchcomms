// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/util/intrusive_ptr.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchComm.hpp"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(_internal, m) {
  intrusive_ptr_class_<BackendWrapper, c10d::Backend>(m, "_BackendWrapper")
      .def(
          py::init<std::shared_ptr<TorchComm>>(),
          "Create BackendWrapper around a TorchCommBackend",
          py::arg("comm"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_comm",
          &BackendWrapper::getComm,
          "Get the underlying TorchComm instance",
          py::call_guard<py::gil_scoped_release>())
      .def("name", &BackendWrapper::getBackendName)
      .def_property_readonly(
          "options",
          &BackendWrapper::getOptions,
          R"(Return the options used to create the torchComm under the hood.)",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_verify_work_timeout",
          &BackendWrapper::verifyWorkTimeoutForTest,
          R"(
Verify that a work object has the expected timeout.
Used for testing timeout propagation.

Args:
    work: The work object to verify.
    timeout: The expected timeout.

Returns:
    bool: True if the work object has the expected timeout, False otherwise.
          )",
          py::arg("work"),
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_default_timeout",
          &BackendWrapper::setTimeout,
          R"(
Set the default timeout for this backend.

Args:
    timeout: The timeout value to set.
          )",
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>());
  intrusive_ptr_class_<WorkWrapper, c10d::Work>(m, "WorkWrapper");
  // Register the backend Options
  intrusive_ptr_class_<BackendWrapper::Options, c10d::Backend::Options>(
      m, "_BackendWrapperOptions");

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
