// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <comms/torchcomms/RemovableHandle.hpp>
#include <comms/torchcomms/TorchComm.hpp>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace torch {
namespace comms {

// Time type for nanoseconds since epoch (matching c10::time_t)
using time_ns_t = int64_t;

/**
 * FlightRecorderHook is a hook that tracks all collective operations
 * in flight for a TorchComm communicator. It uses the pre/post hook
 * mechanism from TorchComm to record operations and their states.
 *
 * The output format matches the OSS FlightRecorder format from
 * torch/csrc/distributed/c10d/FlightRecorder.cpp so traces can be
 * analyzed by the same tooling.
 */
class FlightRecorderHook {
 public:
  // Entry states matching OSS FlightRecorder
  static constexpr const char* kStateScheduled = "scheduled";
  static constexpr const char* kStateStarted = "started";
  static constexpr const char* kStateCompleted = "completed";

  // Version matching OSS FlightRecorder
  static constexpr const char* kVersion = "2.10";

  /**
   * Entry struct representing a single collective operation record.
   * Matches the format from OSS FlightRecorder for compatibility
   * with existing trace analysis tools.
   */
  struct Entry {
    size_t record_id{0};
    size_t pg_id{0};
    std::tuple<std::string, std::string> pg_name; // (name, desc)
    size_t collective_seq_id{0};
    size_t p2p_seq_id{0};
    size_t op_id{0};
    std::string profiling_name;
    std::vector<std::vector<int64_t>> input_sizes;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int64_t>> output_sizes;
    std::vector<std::string> output_dtypes;
    time_ns_t time_created_ns{0};
    std::optional<time_ns_t> time_discovered_started_ns;
    std::optional<time_ns_t> time_discovered_completed_ns;
    int64_t timeout_ms{600000}; // default 10 minutes
    bool is_p2p{false};
    bool retired{false};

    // Backend-generic events for accurate device timing
    // These are used to compute duration when the operation completes
    // Using shared_ptr to allow Entry to be copyable
    std::shared_ptr<c10::Event> start_event;
    std::shared_ptr<c10::Event> end_event;

    // Duration in milliseconds computed from device events
    std::optional<float> duration_ms;

    // Device for the operation (nullopt for CPU-only)
    std::optional<c10::Device> device;

    // Returns "scheduled", "started", or "completed"
    std::string getState() const;
  };

  /**
   * Create a FlightRecorderHook with the specified buffer size.
   * @param max_entries Maximum number of entries to keep in the ring buffer.
   *                    Older entries are overwritten when the buffer is full.
   */
  explicit FlightRecorderHook(size_t max_entries = 2048);

  ~FlightRecorderHook();

  // Disable copy
  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;

  /**
   * Register this hook with a TorchComm communicator.
   * @param comm The communicator to register with.
   * @param pg_id Optional process group ID for this communicator.
   * @param pg_desc Optional description for this process group.
   */
  void registerWithComm(
      std::shared_ptr<TorchComm> comm,
      size_t pg_id = 0,
      const std::string& pg_desc = "");

  /**
   * Unregister this hook from all communicators.
   */
  void unregister();

  /**
   * Dump all entries as a JSON string in the OSS FlightRecorder format.
   * This format is compatible with the fr_trace analyzer tools.
   * @param include_completed If false, only return entries that are not
   * completed.
   */
  std::string dump_json(bool include_completed = true) const;

  /**
   * Get a copy of all current entries.
   */
  std::vector<Entry> getEntries() const;

  /**
   * Clear all entries and reset sequence counters.
   */
  void reset();

  /**
   * Check if the hook is enabled (has registered communicators).
   */
  bool isEnabled() const;

  /**
   * Get the current number of entries.
   */
  size_t size() const;

 private:
  // Hook callbacks - returns record_id for tracking completion
  size_t onPreHook(
      const std::string& comm_name,
      size_t pg_id,
      const std::string& pg_desc,
      const TorchComm::PreHookArgs& args);

  void onPostHook(size_t record_id);

  // Helper to determine if an operation is P2P
  static bool isP2POp(OpName name);

  // Helper to extract tensor sizes
  static std::vector<int64_t> getTensorSizes(const at::Tensor& tensor);

  // Helper to get dtype string
  static std::string getDtypeString(const at::Tensor& tensor);

  // Find entry index by record_id (caller must hold mutex_)
  std::optional<size_t> findEntryIdx(size_t record_id) const;

  // Helper to record CUDA event for a tensor's device
  void recordStartEvent(Entry& entry, const at::Tensor* tensor);

  // Helper to record end event and compute duration
  void recordEndEventAndComputeDuration(Entry& entry);

  // Helper to get current time relative to reference event
  time_ns_t getCurrentTimeNs() const;

  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_;
  size_t next_entry_idx_{0};
  size_t next_record_id_{0};
  size_t collective_seq_id_{0};
  size_t p2p_seq_id_{0};
  size_t op_id_{0};

  // Reference event and wall clock time for calibration
  // All event timings are computed relative to this reference
  std::unique_ptr<c10::Event> reference_event_;
  time_ns_t reference_wall_time_ns_{0};
  bool has_device_reference_{false};

  // Track registered communicators with their handles for unregistration
  struct CommRegistration {
    std::weak_ptr<TorchComm> comm;
    size_t pg_id;
    std::string pg_desc;
    std::unique_ptr<RemovableHandle> pre_hook_handle;
    std::unique_ptr<RemovableHandle> post_hook_handle;

    CommRegistration(
        std::weak_ptr<TorchComm> c,
        size_t id,
        std::string desc,
        std::unique_ptr<RemovableHandle> pre,
        std::unique_ptr<RemovableHandle> post)
        : comm(std::move(c)),
          pg_id(id),
          pg_desc(std::move(desc)),
          pre_hook_handle(std::move(pre)),
          post_hook_handle(std::move(post)) {}
  };
  std::vector<CommRegistration> registrations_;
  bool enabled_{false};

  // Thread-local storage to pass record_id from pre-hook to post-hook
  static thread_local size_t current_record_id_;
};

} // namespace comms
} // namespace torch
