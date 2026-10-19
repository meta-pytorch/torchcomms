// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "comms/utils/colltrace/v2/CollTraceV2.h"

namespace py = pybind11;

namespace {

using meta::comms::colltracev2::COLLTRACE_INVALID_COLLID;
using meta::comms::colltracev2::CollTraceV2;
using meta::comms::colltracev2::Event;
using meta::comms::colltracev2::EventKind;

int64_t timestampNs(const Event& e) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             e.ts.time_since_epoch())
      .count();
}

} // namespace

PYBIND11_MODULE(colltrace, m) {
  m.doc() =
      "colltrace v2: per-communicator collective start/end events, keyed by "
      "the communicator's commDesc (the torchcomms/tclite comm name).";

  m.attr("INVALID_COLLID") = py::int_(COLLTRACE_INVALID_COLLID);

  py::enum_<EventKind>(m, "EventKind")
      .value("COLL_START", EventKind::kCollStartEvent)
      .value("COLL_END", EventKind::kCollEndEvent);

  py::class_<Event>(m, "Event")
      .def_readonly("coll_id", &Event::collId)
      .def_readonly("kind", &Event::kind)
      .def_property_readonly(
          "timestamp_ns", &timestampNs, "Wall-clock time in ns since epoch.")
      .def("__repr__", [](const Event& e) {
        return "Event(coll_id=" + std::to_string(e.collId) + ", kind=" +
            (e.kind == EventKind::kCollStartEvent ? "COLL_START" : "COLL_END") +
            ", timestamp_ns=" + std::to_string(timestampNs(e)) + ")";
      });

  m.def(
      "enabled",
      &meta::comms::colltracev2::enabled,
      "True when NCCL_COLLTRACE_VERSION == 2 and NCCL_COLLTRACE holds "
      "'all' or 'collevents'.");

  m.def(
      "get_latest_coll_id",
      [](std::string_view commDesc) {
        return CollTraceV2::instance().getLatestCollId(commDesc);
      },
      py::arg("comm_desc"),
      "Id of the most recent collective issued on comm_desc, or "
      "INVALID_COLLID.");

  m.def(
      "get_num_kernels",
      [](uint64_t collId) {
        return CollTraceV2::instance().getNumKernels(collId);
      },
      py::arg("coll_id"),
      "Kernels launched so far for coll_id; -1 if unknown or aged out.");

  m.def(
      "get_coll_events",
      [](std::string_view commDesc) {
        return CollTraceV2::instance().getCollEvents(commDesc);
      },
      py::arg("comm_desc"),
      py::call_guard<py::gil_scoped_release>(),
      "Drain the ring and return (and forget) every buffered event for "
      "comm_desc.");

  // Producer-side hooks. The backends call these themselves; they are exposed
  // for backends without a C++ integration and for tests.
  m.def(
      "register_comm",
      [](std::string_view commDesc) {
        return CollTraceV2::instance().registerComm(commDesc);
      },
      py::arg("comm_desc"));
  m.def(
      "deregister_comm",
      [](std::string_view commDesc) {
        CollTraceV2::instance().deregisterComm(commDesc);
      },
      py::arg("comm_desc"));
  m.def(
      "new_coll_id",
      [](std::string_view commDesc) {
        return CollTraceV2::instance().newCollId(commDesc);
      },
      py::arg("comm_desc"));
}
