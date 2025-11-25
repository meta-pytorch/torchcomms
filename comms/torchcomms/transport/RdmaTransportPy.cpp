// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/transport/RdmaTransport.h"

using namespace torch::comms;

namespace {
folly::ScopedEventBaseThread& getScopedEventBaseThread() {
  // This intentionally creates and leaks a global event base thread to be used
  // for all Transports on first use.
  static folly::ScopedEventBaseThread scopedEventBaseThread{"torchcomms_evb"};
  return scopedEventBaseThread;
}
} // namespace

PYBIND11_MODULE(_transport, m) {
  m.doc() = "RdmaTransport python bindings for TorchComm";

  py::class_<RdmaRemoteBuffer, std::shared_ptr<RdmaRemoteBuffer>>(
      m, "RdmaRemoteBuffer")
      .def(
          py::pickle(
              [](const RdmaRemoteBuffer& buffer) { // __getstate__
                return py::make_tuple(
                    reinterpret_cast<uintptr_t>(buffer.ptr), buffer.accessKey);
              },
              [](const py::tuple& t) { // __setstate__
                if (t.size() != 2) {
                  throw std::runtime_error(
                      "Invalid state for RdmaRemoteBuffer");
                }
                return RdmaRemoteBuffer{
                    // NOLINTNEXTLINE(performance-no-int-to-ptr)
                    reinterpret_cast<void*>(t[0].cast<uintptr_t>()),
                    t[1].cast<std::string>()};
              }));

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
      .def("connected", &RdmaTransport::connected)
      .def(
          "write",
          [](RdmaTransport& self,
             const RdmaMemory::View& localBuffer,
             const RdmaRemoteBuffer& remoteBuffer) {
            return static_cast<int>(
                self.write(localBuffer, remoteBuffer, false).get());
          })
      .def(
          "read",
          [](RdmaTransport& self,
             RdmaMemory::MutableView& localBuffer,
             const RdmaRemoteBuffer& remoteBuffer) {
            return static_cast<int>(self.read(localBuffer, remoteBuffer).get());
          });

  py::class_<RdmaMemory::View, std::shared_ptr<RdmaMemory::View>>(
      m, "RdmaMemoryView")
      .def("size", &RdmaMemory::View::size);

  py::class_<RdmaMemory::MutableView, std::shared_ptr<RdmaMemory::MutableView>>(
      m, "RdmaMemoryMutableView");

  py::class_<RdmaMemory, std::shared_ptr<RdmaMemory>>(m, "RdmaMemory")
      .def(py::init([](const at::Tensor& tensor) {
        TORCH_CHECK(
            tensor.is_contiguous(),
            "RdmaMemory currently requires a contiguous tensor");
        // If CPU memory is passed, use device 0 for NIC discovery
        const auto device = tensor.get_device() < 0 ? 0 : tensor.get_device();
        return std::make_shared<RdmaMemory>(
            tensor.data_ptr(), tensor.nbytes(), device);
      }))
      .def(
          "to_view",
          [](RdmaMemory& self) {
            return self.createView(size_t(0), self.length());
          })
      .def(
          "to_mutable_view",
          [](RdmaMemory& self) {
            return self.createMutableView(size_t(0), self.length());
          })
      .def("to_remote_buffer", [](RdmaMemory& self) {
        return RdmaRemoteBuffer{
            const_cast<void*>(self.data()), self.remoteKey()};
      });
}
