// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/hooks/chash/ChashHook.hpp"

namespace py = pybind11;
using namespace torch::comms;

void init_chash_hook_bindings(py::module_& m) {
  py::class_<ChashHook, std::shared_ptr<ChashHook>>(
      m,
      "chash",
      R"(
chash computes and logs hashes of communication buffers before and
after each collective operation, enabling silent data corruption
detection.

Example:
    >>> from torchcomms.hooks import chash
    >>> hasher = chash(output="/tmp/chash.log")
    >>> hasher.register_with_comm(comm)
    >>> # ... run collectives ...
    >>> comm.finalize()
      )")
      .def(
          py::init<std::string, size_t, int>(),
          R"(
          Create a chash logger.

          Args:
              output: File path for log output.
              ring_size: Number of entries in the per-comm hash ring buffer.
              num_blocks: Number of CUDA blocks for the hash kernel.
          )",
          py::arg("output"),
          py::arg("ring_size") = 1024,
          py::arg("num_blocks") = 8)
      .def(
          "register_with_comm",
          &ChashHook::registerWithComm,
          R"(
          Register this hook with a TorchComm communicator.

          Args:
              comm: The communicator to register with.
          )",
          py::arg("comm"),
          py::call_guard<py::gil_scoped_release>());
}
