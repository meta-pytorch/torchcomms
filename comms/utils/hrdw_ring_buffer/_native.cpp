// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Python bindings for HRDWRingBuffer.
//
// One `RingBuffer` / `Reader` pair handles both `MemoryCoherenceScope`
// variants by holding a `std::variant<Device..., System...>` under the
// hood and dispatching via `std::visit`. Callers pick the scope at
// construction time with the `Scope` enum:
//
//     ring = RingBuffer(8192, Scope.DEVICE)
//     reader = Reader(ring)
//     entries, result = reader.poll(stream=my_stream)
//
//     ring = RingBuffer(8192, Scope.SYSTEM)
//     reader = Reader(ring)
//     entries, result =
//     reader.poll(timeout=datetime.timedelta(milliseconds=10))
//
// DataT is fixed to uint64_t — sufficient for tag-style telemetry events.

#include <chrono>
#include <cstdint>
#include <variant>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

namespace py = pybind11;

using namespace hrdw_ring_buffer;

using DataT = uint64_t;

using DeviceRing = HRDWRingBuffer<DataT, MemoryCoherenceScope::Device>;
using SystemRing = HRDWRingBuffer<DataT, MemoryCoherenceScope::System>;
using DeviceReader = HRDWRingBufferReader<DataT, MemoryCoherenceScope::Device>;
using SystemReader = HRDWRingBufferReader<DataT, MemoryCoherenceScope::System>;

namespace {

// Python-facing scope enum. The integer values are kept stable so
// callers can serialize.
enum class PyScope : int { Device = 0, System = 1 };

// Single Python Entry struct shared across scopes. Both
// HRDWEntry<u64, Device> and HRDWEntry<u64, System> have identical
// {timestamp, epoch, data} layouts, so the poll lambda copies fields
// into this scope-agnostic representation. The raw on-device timestamp
// is a uint32 tick (=globaltimer ns >> US_TICK_TIMESTAMP_SHIFT, 10 →
// ~1024 ns per tick) that fits the 16-byte slot; the poll lambda runs
// it through GlobaltimerCalibration::toWallClock(uint32_t) before
// handing to Python so the timestamp is reconstructed against the
// process-wide host↔GPU calibration anchor and arrives as wall-clock
// nanoseconds since the system_clock epoch — directly comparable with
// `time.time_ns()` on the host. The 32-bit tick reconstruction is
// accurate as long as the event is within ~73 minutes of "now"; for
// long-running processes call `refresh_clock()` periodically (~1 Hz is
// what colltrace does) to bound oscillator drift between anchors.
struct PyEntry {
  uint64_t timestamp;
  uint32_t epoch;
  uint64_t data;
};

// Per-scope Python PollResult variants. Each carries the shared
// entries_read / entries_lost counters plus the scope-specific field its
// poll() implementation can populate. `Reader.poll()` returns one or the
// other depending on the ring's scope (bound to a Python typed union).
struct PyDevicePollResult {
  uint64_t entries_read{0};
  uint64_t entries_lost{0};
  // Error from cudaStreamSynchronize / cudaMemcpy during the drain.
  cudaError_t error{cudaSuccess};
};

struct PySystemPollResult {
  uint64_t entries_read{0};
  uint64_t entries_lost{0};
  // True if poll() exited because the timeout elapsed before any new
  // entries arrived.
  bool timed_out{false};
};

// PIMPL wrapper for the ring buffer. The variant carries the actual
// HRDWRingBuffer<...> instance; the scope tag is cached separately so
// `scope()` and dispatch checks don't need to std::visit.
class PyRingBuffer {
 public:
  PyRingBuffer(uint32_t size, PyScope scope)
      : scope_(scope), ring_(make_(size, scope)) {}

  bool valid() const {
    return std::visit([](const auto& r) { return r.valid(); }, ring_);
  }

  uint32_t size() const {
    return std::visit([](const auto& r) { return r.size(); }, ring_);
  }

  PyScope scope() const {
    return scope_;
  }

  // (ring_ptr, write_index_ptr, mask, shift) as int64s, for passing to
  // device kernels.
  std::tuple<int64_t, int64_t, uint32_t, uint32_t> device_handle() const {
    return std::visit(
        [](const auto& r) -> std::tuple<int64_t, int64_t, uint32_t, uint32_t> {
          auto h = r.deviceHandle();
          return {
              reinterpret_cast<int64_t>(h.ring),
              reinterpret_cast<int64_t>(h.writeIndex),
              h.mask,
              h.shift};
        },
        ring_);
  }

  // Host-side single-thread kernel write — only intended for tests
  // that don't want to launch a real device-side writer.
  void write(int64_t stream, uint64_t data) {
    auto err = std::visit(
        [stream, data](auto& r) {
          // Python boundary: torch passes cudaStream_t as int64
          // (torch.cuda.current_stream().cuda_stream).
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          return r.write(reinterpret_cast<cudaStream_t>(stream), DataT{data});
        },
        ring_);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("write failed: ") + cudaGetErrorString(err));
    }
  }

 private:
  friend class PyReader;

  PyScope scope_;
  std::variant<DeviceRing, SystemRing> ring_;

  static std::variant<DeviceRing, SystemRing> make_(
      uint32_t size,
      PyScope scope) {
    if (scope == PyScope::Device) {
      return DeviceRing(size);
    }
    return SystemRing(size);
  }
};

// PIMPL wrapper for the reader. Variant alternative is chosen from the
// ring's scope at construction.
class PyReader {
 public:
  explicit PyReader(const PyRingBuffer& ring)
      : scope_(ring.scope_), reader_(make_(ring)) {}

  // Device scope: stream is required; returns PyDevicePollResult.
  // System scope: timeout is honored, stream is ignored; returns
  // PySystemPollResult.
  //
  // pybind11 maps `std::variant<A, B>` to a Python typed union — the
  // returned object is the concrete variant for the active scope, so
  // callers can use `.error` (device) or `.timed_out` (system) without
  // isinstance branching when the scope is statically known.
  std::pair<
      std::vector<PyEntry>,
      std::variant<PyDevicePollResult, PySystemPollResult>>
  poll(int64_t stream, std::chrono::milliseconds timeout) {
    std::vector<PyEntry> out;
    // Hoist the calibration singleton fetch out of the per-entry lambda
    // (same pattern as comms/utils/colltrace/CollTrace.cc) — get() is
    // function-local-static so the per-call check is cheap but pointless.
    // Used by both scope branches below so `Entry.timestamp` is wall-clock
    // ns since the system_clock epoch regardless of ring scope.
    const auto& cal = GlobaltimerCalibration::get();
    auto toWallNs = [&cal](uint32_t tick) -> uint64_t {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              cal.toWallClock(tick).time_since_epoch())
              .count());
    };
    if (scope_ == PyScope::Device) {
      auto& r = std::get<DeviceReader>(reader_);
      // Python boundary: torch passes cudaStream_t as int64
      // (torch.cuda.current_stream().cuda_stream).
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      auto raw = r.poll(
          reinterpret_cast<cudaStream_t>(stream), [&](const auto& e, uint64_t) {
            out.push_back({toWallNs(e.timestamp), e.epoch, e.data});
          });
      if (raw.error != cudaSuccess) {
        throw std::runtime_error(
            std::string("poll failed: ") + cudaGetErrorString(raw.error));
      }
      PyDevicePollResult result;
      result.entries_read = raw.entriesRead;
      result.entries_lost = raw.entriesLost;
      result.error = raw.error;
      return {std::move(out), result};
    }
    auto& r = std::get<SystemReader>(reader_);
    auto raw = r.poll(
        [&](const auto& e, uint64_t) {
          out.push_back({toWallNs(e.timestamp), e.epoch, e.data});
        },
        timeout);
    PySystemPollResult result;
    result.entries_read = raw.entriesRead;
    result.entries_lost = raw.entriesLost;
    result.timed_out = raw.timedOut;
    return {std::move(out), result};
  }

 private:
  PyScope scope_;
  std::variant<DeviceReader, SystemReader> reader_;

  static std::variant<DeviceReader, SystemReader> make_(
      const PyRingBuffer& ring) {
    if (ring.scope_ == PyScope::Device) {
      return DeviceReader(std::get<DeviceRing>(ring.ring_));
    }
    return SystemReader(std::get<SystemRing>(ring.ring_));
  }
};

} // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = R"(
Low-latency GPU ring buffer for device-side telemetry.

Two coherence scopes are supported via the `Scope` enum:

  - `Scope.DEVICE`: cudaMalloc'd device memory, device-scope 128b
    atomic writes, host drains via cudaMemcpy after
    cudaStreamSynchronize. Use this for graph-captured kernel telemetry
    where the host reads between training steps.

  - `Scope.SYSTEM`: pinned mapped host memory, system-scope 128b
    atomic writes, host polls concurrently via mapped pointers
    (lock-free, optional timeout). Use this when the host needs to
    observe events as they happen.

Both scopes require sm_90+ for atom.exch.b128 and use the same
per-slot epoch validation to discard torn / stale-generation entries.

    from hrdw_ring_buffer._native import RingBuffer, Reader, Scope

    ring = RingBuffer(8192, Scope.DEVICE)
    reader = Reader(ring)
    handle = ring.device_handle()  # (ring_ptr, write_idx_ptr, mask, shift)

    # ... launch kernel that writes via the handle ...

    entries, result = reader.poll(stream=torch.cuda.current_stream().cuda_stream)
    for e in entries:
        print(f"timestamp={e.timestamp} tag={e.data}")
)";

  py::enum_<PyScope>(m, "Scope")
      .value(
          "DEVICE",
          PyScope::Device,
          "Device-scope ring: cudaMalloc memory, host drain via cudaMemcpy.")
      .value(
          "SYSTEM",
          PyScope::System,
          "System-scope ring: pinned mapped memory, lock-free host polling.");

  py::class_<PyEntry>(
      m, "Entry", "A single ring buffer entry (timestamp, epoch, data).")
      .def_readonly(
          "timestamp",
          &PyEntry::timestamp,
          "Wall-clock nanoseconds since the system_clock epoch — directly "
          "comparable with `time.time_ns()` on the host. The on-device "
          "tick (uint32 globaltimer_ns >> 10) is run through "
          "GlobaltimerCalibration::toWallClock(uint32_t) inside the poll "
          "lambda; the same conversion is applied to both Device and "
          "System scope rings so Entry.timestamp has the same reference "
          "frame regardless of scope. Reconstruction is accurate as long "
          "as the event is within ~73 minutes of \"now\"; call "
          "`refresh_clock()` periodically (~1 Hz, what colltrace does) to "
          "bound oscillator drift between anchors for long-running "
          "processes.")
      .def_readonly(
          "epoch",
          &PyEntry::epoch,
          "Slot generation, used by the reader for torn-write detection.")
      .def_readonly("data", &PyEntry::data, "User-supplied uint64 tag.")
      .def("__repr__", [](const PyEntry& e) {
        return "Entry(timestamp=" + std::to_string(e.timestamp) +
            ", epoch=" + std::to_string(e.epoch) +
            ", data=" + std::to_string(e.data) + ")";
      });

  py::class_<PyDevicePollResult>(
      m,
      "DevicePollResult",
      "Result metadata from a device-scope Reader.poll() call.")
      .def_readonly("entries_read", &PyDevicePollResult::entries_read)
      .def_readonly("entries_lost", &PyDevicePollResult::entries_lost)
      .def_readonly(
          "error",
          &PyDevicePollResult::error,
          "cudaError_t from the stream sync / memcpy drain (always "
          "cudaSuccess on a successful poll — non-success is also raised "
          "as RuntimeError).")
      .def("__repr__", [](const PyDevicePollResult& r) {
        return "DevicePollResult(entries_read=" +
            std::to_string(r.entries_read) +
            ", entries_lost=" + std::to_string(r.entries_lost) +
            ", error=" + std::to_string(static_cast<int>(r.error)) + ")";
      });

  py::class_<PySystemPollResult>(
      m,
      "SystemPollResult",
      "Result metadata from a system-scope Reader.poll() call.")
      .def_readonly("entries_read", &PySystemPollResult::entries_read)
      .def_readonly("entries_lost", &PySystemPollResult::entries_lost)
      .def_readonly(
          "timed_out",
          &PySystemPollResult::timed_out,
          "True if poll returned because the timeout elapsed without any "
          "new entries arriving.")
      .def("__repr__", [](const PySystemPollResult& r) {
        return "SystemPollResult(entries_read=" +
            std::to_string(r.entries_read) +
            ", entries_lost=" + std::to_string(r.entries_lost) +
            ", timed_out=" + (r.timed_out ? "True" : "False") + ")";
      });

  py::class_<PyRingBuffer>(m, "RingBuffer", R"(
GPU ring buffer with uint64 data entries.

Args:
    size: Number of slots (rounded up to next power of 2).
    scope: `Scope.DEVICE` (default) or `Scope.SYSTEM`. See module docs
        for the trade-offs.
)")
      .def(
          py::init<uint32_t, PyScope>(),
          py::arg("size"),
          py::arg("scope") = PyScope::Device)
      .def(
          "valid",
          &PyRingBuffer::valid,
          "True if the ring buffer was allocated successfully.")
      .def_property_readonly(
          "size", &PyRingBuffer::size, "Number of slots (power of 2).")
      .def_property_readonly(
          "scope",
          &PyRingBuffer::scope,
          "The MemoryCoherenceScope of this ring.")
      .def(
          "device_handle",
          &PyRingBuffer::device_handle,
          R"(
Returns (ring_ptr, write_index_ptr, mask, shift) for passing to
device kernels. Pointers are raw device pointers as int64.
)")
      .def(
          "write",
          &PyRingBuffer::write,
          py::arg("stream"),
          py::arg("data"),
          // Releases the GIL while the host launch + cudaGetLastError run
          // so concurrent Python threads aren't blocked on the kernel
          // launch's host overhead.
          py::call_guard<py::gil_scoped_release>(),
          "Enqueue a single-thread kernel that writes one entry (for testing).");

  py::class_<PyReader>(m, "Reader", R"(
Host-side consumer for a RingBuffer. The reader binds to the ring's
scope at construction; `poll()` takes the args appropriate for that
scope (and ignores the others).

Args:
    ring: The RingBuffer to consume from. Must outlive the reader.
)")
      .def(
          py::init<const PyRingBuffer&>(),
          py::arg("ring"),
          // The underlying HRDWRingBufferReader stores non-owning
          // pointers into the ring; pin the ring (arg 2) to the reader
          // (arg 1) so a Python caller can't drop the RingBuffer while
          // still holding the Reader.
          py::keep_alive<1, 2>())
      .def(
          "poll",
          &PyReader::poll,
          py::arg("stream") = 0,
          py::arg("timeout") = std::chrono::milliseconds{0},
          // Release the GIL for the duration of the poll: Device-scope
          // calls cudaStreamSynchronize + cudaMemcpy; System-scope can
          // wait up to `timeout` for the first entry. Both block the
          // calling thread and would otherwise starve other Python
          // threads.
          py::call_guard<py::gil_scoped_release>(),
          R"(
Drain newly-written entries from the ring buffer.

Device scope: requires `stream` (raw cudaStream_t as int64). The reader
synchronizes the stream, cudaMemcpys the ring window to host, and
returns the valid entries.

System scope: optional `timeout` (a datetime.timedelta). The reader
polls the pinned mapped memory directly; if no entries are available
yet, it waits up to `timeout` for the first one before returning.

The internal read cursor auto-advances on every call, so successive
polls only return entries written since the previous one. No reset
is required between polls.

Returns:
    (entries, result) — `entries` is a list of `Entry`, and `result`
    is a `DevicePollResult` (device scope) or `SystemPollResult`
    (system scope) carrying `entries_read`, `entries_lost`, and the
    scope-specific `error` / `timed_out` field.
)");

  m.def(
      "refresh_clock",
      []() { return GlobaltimerCalibration::get().refresh(); },
      py::call_guard<py::gil_scoped_release>(),
      R"(
Re-anchor the process-wide GPU-globaltimer ↔ wall-clock calibration that
`Entry.timestamp` uses. Cheap (one single-thread kernel + one stream sync
on a dedicated side stream); colltrace invokes it at ~1 Hz to keep
residual oscillator drift under ~100 ns at 100 ppm. Call periodically
from the polling thread for long-running processes. Idempotent and
thread-safe — concurrent callers race on an internal flag and the loser
short-circuits without touching the CUDA stream.

Returns True when the anchor was successfully replaced, False when a
concurrent caller already had a refresh in flight or the refresh failed
(the prior anchor is kept; failure is logged).
)");
}
