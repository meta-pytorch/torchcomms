// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/transport/RdmaTransport.h"

namespace py = pybind11;
using namespace torch::comms;

namespace {
folly::ScopedEventBaseThread& getScopedEventBaseThread() {
  // This intentionally creates and leaks a global event base thread to be used
  // for all Transports on first use.
  static folly::ScopedEventBaseThread scopedEventBaseThread{"torchcomms_evb"};
  return scopedEventBaseThread;
}
} // namespace

PYBIND11_MODULE(_comms_ncclx, m) {
  m.doc() = "NCCLX specific python bindings for TorchComm";

  py::class_<TorchCommNCCLX, std::shared_ptr<TorchCommNCCLX>>(
      m, "TorchCommNCCLX");

  py::class_<RdmaTransport, std::shared_ptr<RdmaTransport>>(m, "RdmaTransport")
      // initialize a new RDMATransport using a custom init fn
      .def(py::init([](at::Device device) {
        TORCH_INTERNAL_ASSERT(device.is_cuda());
        int cuda_device = device.index();
        return std::make_shared<RdmaTransport>(
            cuda_device, getScopedEventBaseThread().getEventBase());
      }))
      .def_static("supported", &RdmaTransport::supported)
      .def("bind", [](RdmaTransport& self) { return py::bytes(self.bind()); })
      .def(
          "connect",
          [](RdmaTransport& self, const py::bytes& peerUrl) {
            std::string peerUrlStr = peerUrl.cast<std::string>();
            return static_cast<int>(self.connect(peerUrlStr));
          })
      .def("connected", &RdmaTransport::connected);
}
