// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "comms/utils/colltrace/nicstats/NicCounterNames.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"
#include "comms/utils/colltrace/v2/CollTraceV2TestAccessor.h"

namespace py = pybind11;

namespace {

using meta::comms::colltracev2::COLLTRACE_INVALID_COLLID;
using meta::comms::colltracev2::CollTraceV2;
using meta::comms::colltracev2::CollTraceV2TestAccessor;
using meta::comms::colltracev2::Event;
using meta::comms::colltracev2::EventKind;

int64_t timestampNs(const Event& e) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             e.ts.time_since_epoch())
      .count();
}

namespace nicstats = meta::comms::colltrace::nicstats;

const char* counterStatusName(nicstats::NicCounterStatus status) {
  switch (status) {
    case nicstats::NicCounterStatus::Ok:
      return "ok";
    case nicstats::NicCounterStatus::Unsupported:
      return "unsupported";
    case nicstats::NicCounterStatus::ReadFailed:
      return "read_failed";
    case nicstats::NicCounterStatus::NotRequested:
      return "not_requested";
  }
  return "unknown";
}

nicstats::NicCounterId counterNamed(const std::string& name) {
  const auto id = nicstats::nicCounterIdFromName(name);
  if (!id.has_value()) {
    throw py::value_error("'" + name + "' is not a NIC counter");
  }
  return *id;
}

// The counters `names` picks, in counter-id order.
std::vector<nicstats::NicCounterId> countersPicked(
    const std::vector<std::string>& names) {
  std::array<bool, nicstats::kNumCounters> picked{};
  for (const auto& name : names) {
    picked[static_cast<uint32_t>(counterNamed(name))] = true;
  }
  std::vector<nicstats::NicCounterId> ids;
  for (uint32_t i = 0; i < nicstats::kNumCounters; ++i) {
    if (picked[i]) {
      ids.push_back(static_cast<nicstats::NicCounterId>(i));
    }
  }
  return ids;
}

/* The counters `reading` was read for, NICSTATS_COUNTERS as NIC stats read it,
 * in counter-id order: every other counter is NotRequested. */
std::vector<nicstats::NicCounterId> countersAskedFor(
    const nicstats::NicStatsReading& reading) {
  std::vector<nicstats::NicCounterId> ids;
  for (uint32_t i = 0; i < nicstats::kNumCounters; ++i) {
    if (reading.counters.status[i] !=
        nicstats::NicCounterStatus::NotRequested) {
      ids.push_back(static_cast<nicstats::NicCounterId>(i));
    }
  }
  return ids;
}

constexpr uint32_t kNumCounterStatuses = 4;
static_assert(
    static_cast<uint32_t>(nicstats::NicCounterStatus::NotRequested) + 1 ==
    kNumCounterStatuses);

/* The Python objects every reading repeats, made once at import. Never freed:
 * their destructors would run after the interpreter has finalized. */
struct NicStatsPyObjects {
  std::array<py::str, nicstats::kNumCounters> counterNames;
  std::array<py::str, nicstats::kNumNicStatsSources> sourceNames;
  // (None, status) for each status: an Ok counter gets a fresh tuple instead.
  std::array<py::tuple, kNumCounterStatuses> noValue;
  std::array<py::str, kNumCounterStatuses> statusNames;
  py::str rail{"rail"};
  py::str port{"port"};
  py::str sequence{"sequence"};
  py::str unixNs{"unix_ns"};
  py::str ageNs{"age_ns"};
  py::str lagNs{"lag_ns"};
  py::str samplePeriodNs{"sample_period_ns"};
  py::str source{"source"};
  py::str counterSources{"counter_sources"};
  py::str sourceReadNs{"source_read_ns"};
  py::str priority{"priority"};
  py::str counters{"counters"};
};

const NicStatsPyObjects* makeNicStatsPyObjects() {
  auto* objects = new NicStatsPyObjects();
  for (uint32_t i = 0; i < nicstats::kNumCounters; ++i) {
    const auto name =
        nicstats::nicCounterName(static_cast<nicstats::NicCounterId>(i));
    objects->counterNames[i] = py::str(name.data(), name.size());
  }
  for (uint32_t i = 0; i < nicstats::kNumNicStatsSources; ++i) {
    const auto name =
        nicstats::nicSourceKindName(static_cast<nicstats::NicStatsSource>(i));
    objects->sourceNames[i] = py::str(name.data(), name.size());
  }
  for (uint32_t i = 0; i < kNumCounterStatuses; ++i) {
    objects->statusNames[i] =
        py::str(counterStatusName(static_cast<nicstats::NicCounterStatus>(i)));
    objects->noValue[i] = py::make_tuple(py::none(), objects->statusNames[i]);
  }
  return objects;
}

// Made at import, with the GIL held.
const NicStatsPyObjects& nicStatsPy() {
  static const NicStatsPyObjects* const objects = makeNicStatsPyObjects();
  return *objects;
}

/* A counter whose status is not ok carries no number: None, so a caller cannot
 * do arithmetic on the sentinel. Which mechanism read a counter is given only
 * for counters one could read; where none could, the reading's `source` is a
 * placeholder and is None. */
py::dict nicStatsReadingToPython(
    const nicstats::NicStatsReading& reading,
    const std::vector<nicstats::NicCounterId>& picked,
    int64_t nowNs) {
  const auto& objs = nicStatsPy();
  py::dict counters;
  py::dict sources;
  bool anyRead = false;
  for (const auto id : picked) {
    const auto i = static_cast<uint32_t>(id);
    const auto& key = objs.counterNames[i];
    const auto status = reading.counters.status[i];
    const auto s = static_cast<uint32_t>(status);
    if (status == nicstats::NicCounterStatus::Ok) {
      counters[key] = py::make_tuple(
          py::int_(reading.counters.values[i]), objs.statusNames[s]);
    } else {
      counters[key] = objs.noValue[s];
    }
    if (status == nicstats::NicCounterStatus::Ok ||
        status == nicstats::NicCounterStatus::ReadFailed) {
      anyRead = true;
      sources[key] =
          objs.sourceNames[static_cast<uint32_t>(reading.counterSource[i])];
    }
  }
  py::dict readNs;
  for (uint32_t k = 0; k < nicstats::kNumNicStatsSources; ++k) {
    if (const auto ns = reading.sourceReadNs[k]; ns != 0) {
      readNs[objs.sourceNames[k]] = ns;
    }
  }
  const auto& device = reading.rail.device;
  py::dict out;
  out[objs.rail] =
      std::string{device.data(), ::strnlen(device.data(), device.size())};
  out[objs.port] = reading.rail.port;
  out[objs.sequence] = reading.sequence;
  out[objs.unixNs] = reading.counters.unixNs;
  out[objs.ageNs] = nowNs - static_cast<int64_t>(reading.counters.unixNs);
  out[objs.lagNs] = reading.counters.lagNs;
  out[objs.samplePeriodNs] = reading.counters.samplePeriodNs;
  out[objs.source] = anyRead
      ? py::object(objs.sourceNames[static_cast<uint32_t>(reading.source)])
      : py::object(py::none());
  out[objs.counterSources] = sources;
  out[objs.sourceReadNs] = readNs;
  out[objs.priority] = reading.priority.has_value()
      ? py::object(py::int_(*reading.priority))
      : py::object(py::none());
  out[objs.counters] = counters;
  return out;
}

int64_t nowUnixNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

} // namespace

PYBIND11_MODULE(colltrace, m) {
  m.doc() =
      "colltrace v2: per-communicator collective start/end events, keyed by "
      "the communicator's commDesc (the torchcomms/tclite comm name).";

  m.attr("INVALID_COLLID") = py::int_(COLLTRACE_INVALID_COLLID);
  nicStatsPy();

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

  m.def(
      "get_nic_stats",
      [](const std::optional<std::vector<std::string>>& counters) {
        const auto named = counters.has_value()
            ? countersPicked(*counters)
            : std::vector<nicstats::NicCounterId>{};
        // The GIL stays held: the copy takes a microsecond, and getting the GIL
        // back after releasing it can wait out another thread's switch
        // interval.
        const auto readings = CollTraceV2::instance().nicStats();
        py::list out;
        if (readings.empty()) {
          return out;
        }
        const int64_t nowNs = nowUnixNs();
        for (const auto& reading : readings) {
          if (counters.has_value()) {
            out.append(nicStatsReadingToPython(reading, named, nowNs));
          } else {
            out.append(nicStatsReadingToPython(
                reading, countersAskedFor(reading), nowNs));
          }
        }
        return out;
      },
      py::arg("counters") = py::none(),
      "The latest reading of each of the process's NICs, one dict per NIC, "
      "ordered by rail and port: rail, port, sequence (per rail; unchanged "
      "while the NIC has not been read again), unix_ns, age_ns, lag_ns, "
      "sample_period_ns, source, counter_sources, source_read_ns, priority, "
      "and counters, name -> (value or None, status). counters names the "
      "counters to return, NICSTATS_COUNTERS' when None; one outside "
      "NICSTATS_COUNTERS is (None, 'not_requested'), and an unknown name "
      "raises ValueError. Empty while NIC stats are off or have read "
      "nothing.");

  m.def(
      "_put_nic_reading_for_test",
      [](const std::string& rail,
         uint8_t port,
         const std::map<std::string, uint64_t>& counters,
         uint64_t unixNs) {
        nicstats::NicStatsReading reading;
        rail.copy(reading.rail.device.data(), reading.rail.device.size() - 1);
        reading.rail.port = port;
        reading.counters.unixNs = unixNs;
        for (const auto& [name, value] : counters) {
          const auto i = static_cast<uint32_t>(counterNamed(name));
          reading.counters.values[i] = value;
          reading.counters.status[i] = nicstats::NicCounterStatus::Ok;
        }
        CollTraceV2TestAccessor::putNicStats(
            CollTraceV2::instance(), std::move(reading));
      },
      py::arg("rail"),
      py::arg("port"),
      py::arg("counters"),
      py::arg("unix_ns"),
      "Put a reading where get_nic_stats() copies from. Tests only.");

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
