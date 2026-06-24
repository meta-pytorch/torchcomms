// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/pybind11.h>

#include "comms/torchcomms/transport/RdmaTransportCCA.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_transport_cca_hook, m) {
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
