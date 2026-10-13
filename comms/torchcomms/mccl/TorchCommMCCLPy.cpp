// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/util/intrusive_ptr.h>
#include <folly/CppAttributes.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <chrono>
#include <utility>

#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

namespace {

py::list lifecycleEventsToPython(
    const std::vector<::mccl::LifecycleEvent>& events) {
  py::list result;
  for (const auto& event : events) {
    const char* eventType = nullptr;
    switch (event.eventType) {
      case ::mccl::LifecycleEventType::Enqueue:
        eventType = "enqueue";
        break;
      case ::mccl::LifecycleEventType::Start:
        eventType = "start";
        break;
      case ::mccl::LifecycleEventType::End:
        eventType = "end";
        break;
    }
    if (eventType == nullptr) {
      throw py::value_error("unknown MCCL lifecycle event type");
    }
    const py::object replayId =
        event.replayId.has_value() ? py::cast(*event.replayId) : py::none();
    const auto timestamp =
        std::chrono::duration<double>(event.timestamp.time_since_epoch())
            .count();
    result.append(
        py::make_tuple(
            replayId,
            event.commId,
            event.collId,
            event.executionCollId,
            eventType,
            timestamp));
  }
  return result;
}

py::list drainLifecycleEventsToPython(TorchCommMCCL& comm) {
  std::vector<::mccl::LifecycleEvent> events;
  {
    py::gil_scoped_release release;
    events = comm.drainLifecycleEvents();
  }
  return lifecycleEventsToPython(events);
}

// Helper to extract TorchCommMCCL* from a Python backend object.
// Returns nullptr if extraction fails or the backend is not MCCL.
//
// NOTE: We cannot use py::isinstance<TorchCommMCCL> here because on ARM64 with
// RTLD_LOCAL, the pybind11 type registrations for TorchCommMCCL and
// TorchCommBackend are in different libraries (libtorchcomms.so vs
// _comms_mccl.so). Due to typeinfo pointer comparison failures, isinstance()
// would always return false for objects returned by torchcomms APIs.
// String-based type checking is a workaround for this cross-library RTTI issue.
//
// WARNING: This function accesses pybind11 private implementation details
// (py::detail::instance). This is fragile and may break with pybind11 updates.
// We intentionally accept this risk because:
// 1. We cannot modify torchcomms to fix the RTLD_LOCAL loading
// 2. Some MCCL-specific APIs (certain CPU collectives) are not yet in the
// public TC API
// 3. py::cast<> fails due to the same RTTI mismatch that necessitates patching
TorchCommMCCL* FOLLY_NULLABLE extractMcclBackend(py::object backend) {
  if (!backend || backend.is_none()) {
    return nullptr;
  }

  PyTypeObject* py_type = Py_TYPE(backend.ptr());
  const char* type_name = py_type->tp_name;
  if (!type_name) {
    return nullptr;
  }

  std::string_view name_view(type_name);
  if (name_view.find("TorchCommBackend") == std::string_view::npos &&
      name_view.find("TorchCommMCCL") == std::string_view::npos) {
    return nullptr;
  }

  // Access pybind11 internal instance structure to get the C++ object pointer.
  // NOLINTNEXTLINE(facebook-hte-ReliesOnPyBind11PrivateDetails)
  auto* inst = reinterpret_cast<py::detail::instance*>(backend.ptr());
  if (!inst->simple_layout || !inst->simple_holder_constructed) {
    return nullptr;
  }

  // NOLINTNEXTLINE(facebook-hte-ReliesOnPyBind11PrivateDetails)
  void** vh = inst->simple_value_holder;
  auto& holder = reinterpret_cast<std::shared_ptr<TorchCommBackend>&>(vh[1]);
  return dynamic_cast<TorchCommMCCL*>(holder.get());
}

// Patch the MCCL-specific TorchCommBackend methods (setTimeout)
// onto the base backend class so they resolve correctly when cross-library RTTI
// resolution fails under RTLD_LOCAL.
void patchMcclMethods(py::module_& m) {
  try {
    // --- TorchCommBackend method helpers (MCCL-specific methods) ---

    m.def(
        "_set_timeout_direct",
        [](py::object self, std::chrono::milliseconds duration) {
          TorchCommMCCL* mccl = extractMcclBackend(std::move(self));
          if (mccl) {
            mccl->setTimeout(duration);
          }
        },
        py::arg("self"),
        py::arg("duration"));

    m.def(
        "_get_and_clear_collective_stats_direct",
        [](py::object self) {
          TorchCommMCCL* mccl = extractMcclBackend(std::move(self));
          if (mccl) {
            return mccl->getAndClearCollectiveStats();
          }
          return std::unordered_map<std::string, ::mccl::CollectiveStat>{};
        },
        py::arg("self"));

    m.def(
        "_colltrace_get_comm_id_direct",
        [](py::object self) -> py::object {
          TorchCommMCCL* mccl = extractMcclBackend(std::move(self));
          if (mccl == nullptr) {
            return py::none();
          }
          std::optional<uint64_t> commId;
          {
            // The getter takes MCCL's communicator lock; holding the GIL
            // across it would stall every other Python thread.
            py::gil_scoped_release release;
            commId = mccl->getLifecycleCommId();
          }
          return commId.has_value() ? py::cast(*commId) : py::none();
        },
        py::arg("self"));

    m.def(
        "_colltrace_get_latest_coll_id_direct",
        [](py::object self) -> py::object {
          TorchCommMCCL* mccl = extractMcclBackend(std::move(self));
          if (mccl == nullptr) {
            return py::none();
          }
          std::optional<uint64_t> collId;
          {
            py::gil_scoped_release release;
            collId = mccl->getLatestLifecycleCollectiveId();
          }
          return collId.has_value() ? py::cast(*collId) : py::none();
        },
        py::arg("self"));

    m.def(
        "_colltrace_get_unread_events_direct",
        [](py::object self) {
          TorchCommMCCL* mccl = extractMcclBackend(std::move(self));
          return mccl == nullptr ? py::list{}
                                 : drainLifecycleEventsToPython(*mccl);
        },
        py::arg("self"));

    // Use Python to patch methods on TorchWork and TorchCommBackend.
    // We're still inside PYBIND11_MODULE, so the module is not yet in
    // sys.modules. We manually register it first so the patching code can
    // find it.
    py::module_ sys = py::module_::import("sys");
    py::dict modules = sys.attr("modules");

    // Register under a known key that we control
    modules["_comms_mccl_patching_temp"] = m;

    const char* patch_code = R"(
import sys
import torchcomms._comms as _comms

# Get the MCCL module - we registered it under a known key
_mccl = sys.modules['_comms_mccl_patching_temp']

# --- Add MCCL-specific methods to TorchCommBackend ---
# These methods have no public TorchComm equivalent yet, so we add them to the
# base backend class so callers can reach them when the backend is MCCL (the
# backend object is base-typed on ARM64/GB200 due to cross-library RTTI).

def _backend_set_timeout(self, duration):
    return _mccl._set_timeout_direct(self, duration)

_comms.TorchCommBackend.setTimeout = _backend_set_timeout

def _backend_get_and_clear_collective_stats(self):
    return _mccl._get_and_clear_collective_stats_direct(self)

_comms.TorchCommBackend.getAndClearCollectiveStats = _backend_get_and_clear_collective_stats

def _backend_colltrace_get_comm_id(self):
    return _mccl._colltrace_get_comm_id_direct(self)

_comms.TorchCommBackend.colltrace_get_comm_id = _backend_colltrace_get_comm_id

def _backend_colltrace_get_latest_coll_id(self):
    return _mccl._colltrace_get_latest_coll_id_direct(self)

_comms.TorchCommBackend.colltrace_get_latest_coll_id = _backend_colltrace_get_latest_coll_id

def _backend_colltrace_get_unread_events(self):
    return _mccl._colltrace_get_unread_events_direct(self)

_comms.TorchCommBackend.colltrace_get_unread_events = _backend_colltrace_get_unread_events

# Clean up the temporary module registration
del sys.modules['_comms_mccl_patching_temp']
)";

    if (PyRun_SimpleString(patch_code) != 0) {
      throw std::runtime_error("Failed to execute Python patching code");
    }
  } catch (const std::exception& e) {
    TC_LOG(WARNING) << "Failed to patch MCCL methods: " << e.what();
  }
}

} // namespace

PYBIND11_MODULE(_comms_mccl, m) {
  m.doc() = "MCCL specific python bindings for TorchComm";

  py::module_::import("torchcomms._comms");

  py::class_<BroadcastOptions>(m, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("hints", &BroadcastOptions::hints)
      .def_readwrite("timeout", &BroadcastOptions::timeout);

  py::class_<::mccl::CollectiveStat>(m, "CollectiveStat")
      .def_readonly("count", &::mccl::CollectiveStat::count)
      .def_readonly("total_us", &::mccl::CollectiveStat::total_us)
      .def_readonly("min_us", &::mccl::CollectiveStat::min_us)
      .def_readonly("max_us", &::mccl::CollectiveStat::max_us)
      .def_readonly("num_blocks", &::mccl::CollectiveStat::num_blocks)
      .def_readonly("block_size", &::mccl::CollectiveStat::block_size)
      .def_readonly("blocks_per_sm", &::mccl::CollectiveStat::blocks_per_sm)
      .def_readonly("total_sm_us", &::mccl::CollectiveStat::total_sm_us)
      .def_readonly("p50_us", &::mccl::CollectiveStat::p50_us)
      .def_readonly("p90_us", &::mccl::CollectiveStat::p90_us)
      .def_readonly("p99_us", &::mccl::CollectiveStat::p99_us)
      .def_readonly("queue_p99_us", &::mccl::CollectiveStat::queue_p99_us);

  py::class_<TorchCommMCCL, TorchCommBackend, std::shared_ptr<TorchCommMCCL>>(
      m, "TorchCommMCCL")
      .def("setTimeout", &TorchCommMCCL::setTimeout, py::arg("duration"))
      .def(
          "getAndClearCollectiveStats",
          &TorchCommMCCL::getAndClearCollectiveStats)
      .def(
          "colltrace_get_comm_id",
          [](TorchCommMCCL& self) -> py::object {
            std::optional<uint64_t> commId;
            {
              py::gil_scoped_release release;
              commId = self.getLifecycleCommId();
            }
            return commId.has_value() ? py::cast(*commId) : py::none();
          })
      .def(
          "colltrace_get_latest_coll_id",
          [](TorchCommMCCL& self) -> py::object {
            std::optional<uint64_t> collId;
            {
              py::gil_scoped_release release;
              collId = self.getLatestLifecycleCollectiveId();
            }
            return collId.has_value() ? py::cast(*collId) : py::none();
          })
      .def("colltrace_get_unread_events", [](TorchCommMCCL& self) {
        return drainLifecycleEventsToPython(self);
      });

  intrusive_ptr_class_<TorchWorkMCCL>(m, "TorchWorkMCCL")
      .def(
          "is_completed",
          &TorchWorkMCCL::isCompleted,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait",
          &TorchWorkMCCL::wait,
          py::call_guard<py::gil_scoped_release>())
      // Fault Tolerance API: explicit binding required due to ARM64/RTLD_LOCAL
      // cross-module pybind11 issue - TorchWorkMCCL doesn't inherit Python
      // methods from TorchWork in _comms module.
      .def(
          "wait_blocking",
          &TorchWorkMCCL::waitBlocking,
          py::call_guard<py::gil_scoped_release>());

  // Patch the MCCL-specific TorchCommBackend methods for ARM64 compatibility.
  // This must be done at the end, after all classes and functions are defined.
  patchMcclMethods(m);
}
