// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <optional>

#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/transport/RdmaTransport.h"
#include "comms/torchcomms/transport/RdmaTransportCCA.hpp"

using namespace torch::comms;

namespace {
folly::ScopedEventBaseThread& getScopedEventBaseThread() {
  // This intentionally creates and leaks a global event base thread to be used
  // for all Transports on first use.
  static folly::ScopedEventBaseThread scopedEventBaseThread{"torchcomms_evb"};
  return scopedEventBaseThread;
}

py::tuple getRdmaRemoteBufferState(const RdmaRemoteBuffer& buffer) {
  return py::make_tuple(
      reinterpret_cast<uintptr_t>(buffer.ptr), buffer.len, buffer.accessKey);
}

RdmaRemoteBuffer rdmaRemoteBufferFromState(
    uintptr_t ptr,
    size_t len,
    const std::string& accessKey) {
  return RdmaRemoteBuffer{
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(ptr),
      len,
      accessKey};
}

py::tuple getRdmaRemoteBufferReduceTuple(const py::object& self) {
  return py::make_tuple(
      py::type::of(self),
      getRdmaRemoteBufferState(self.cast<const RdmaRemoteBuffer&>()));
}
} // namespace

PYBIND11_MODULE(_transport, m, py::mod_gil_not_used()) {
  m.doc() = "RdmaTransport python bindings for TorchComm";

  py::class_<RdmaRemoteBuffer>(m, "RdmaRemoteBuffer")
      .def(
          py::init([](uintptr_t ptr, size_t len, const std::string& accessKey) {
            return rdmaRemoteBufferFromState(ptr, len, accessKey);
          }),
          py::arg("ptr"),
          py::arg("len"),
          py::arg("access_key"))
      .def("__getstate__", &getRdmaRemoteBufferState)
      .def("__reduce__", &getRdmaRemoteBufferReduceTuple)
      .def("__reduce_ex__", [](const py::object& self, int /* protocol */) {
        // Older pickle protocols reconstruct via constructor args rather
        // than pybind11's __newobj__ helper.
        return getRdmaRemoteBufferReduceTuple(self);
      });

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
      m, "RdmaMemoryMutableView")
      .def("size", &RdmaMemory::MutableView::size);

  py::class_<RdmaMemory, std::shared_ptr<RdmaMemory>>(m, "RdmaMemory")
      .def(
          py::init([](const at::Tensor& tensor) {
            TORCH_CHECK(
                tensor.is_contiguous(),
                "RdmaMemory currently requires a contiguous tensor");
            // If CPU memory is passed, use device 0 for NIC discovery
            const auto device =
                tensor.get_device() < 0 ? 0 : tensor.get_device();
            return std::make_shared<RdmaMemory>(
                tensor.data_ptr(), tensor.nbytes(), device);
          }),
          py::arg("tensor"))
      .def(
          "to_view",
          [](RdmaMemory& self,
             std::optional<size_t> offset,
             std::optional<size_t> length) {
            size_t off = offset.value_or(0);
            size_t len = length.value_or(self.length() - off);
            return self.createView(off, len);
          },
          py::arg("offset") = py::none(),
          py::arg("length") = py::none())
      .def(
          "to_mutable_view",
          [](RdmaMemory& self,
             std::optional<size_t> offset,
             std::optional<size_t> length) {
            size_t off = offset.value_or(0);
            size_t len = length.value_or(self.length() - off);
            return self.createMutableView(off, len);
          },
          py::arg("offset") = py::none(),
          py::arg("length") = py::none())
      .def(
          "to_remote_buffer",
          [](RdmaMemory& self) {
            return RdmaRemoteBuffer{
                const_cast<void*>(self.data()),
                self.length(),
                self.remoteKey()};
          })
      .def("reused_registration", &RdmaMemory::reusedRegistration);

  m.def(
      "attach_rdma_memory_hook",
      [](const py::capsule& reg, const py::capsule& dereg) {
        torch::comms::attachRdmaMemoryHook(
            reinterpret_cast<torch::comms::RdmaRegFn>(reg.get_pointer()),
            reinterpret_cast<torch::comms::RdmaRegFn>(dereg.get_pointer()));
      },
      py::arg("reg"),
      py::arg("dereg"),
      "Install the CUDA caching-allocator hook that forwards segment events to "
      "the given reg/dereg callbacks (capsules). Idempotent.");
}
