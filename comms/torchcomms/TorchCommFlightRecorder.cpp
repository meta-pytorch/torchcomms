// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommFlightRecorder.hpp"

#include <ATen/DeviceAccelerator.h>
#include <nlohmann/json.hpp>

namespace torch {
namespace comms {

using json = nlohmann::json;

// Initialize thread-local storage
thread_local size_t FlightRecorderHook::current_record_id_ = 0;

std::string FlightRecorderHook::Entry::getState() const {
  if (time_discovered_completed_ns.has_value()) {
    return FlightRecorderHook::kStateCompleted;
  } else if (time_discovered_started_ns.has_value()) {
    return FlightRecorderHook::kStateStarted;
  }
  return FlightRecorderHook::kStateScheduled;
}

FlightRecorderHook::FlightRecorderHook(size_t max_entries)
    : max_entries_(max_entries) {
  entries_.reserve(max_entries_);
  // Note: Reference event initialization is done lazily when we first
  // encounter a device tensor, since we don't know the device type ahead of
  // time
}

FlightRecorderHook::~FlightRecorderHook() {
  unregister();
}

void FlightRecorderHook::registerWithComm(
    std::shared_ptr<TorchComm> comm,
    size_t pg_id,
    const std::string& pg_desc) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string comm_name(comm->getCommName());

  // Register pre-hook - stores record_id in thread-local storage
  auto pre_hook_handle = comm->registerPreHook(
      [this, comm_name, pg_id, pg_desc](const TorchComm::PreHookArgs& args) {
        size_t record_id = this->onPreHook(comm_name, pg_id, pg_desc, args);
        // Store record_id in thread-local storage for post-hook
        current_record_id_ = record_id;
      });

  // Register post-hook - called via work callback when work completes
  // The post-hook is invoked by TorchComm when the work's callback fires
  auto post_hook_handle =
      comm->registerPostHook([this](const TorchComm::PostHookArgs& /* args */) {
        // Use the record_id stored by pre-hook to mark completion
        this->onPostHook(current_record_id_);
      });

  // Store registration with handles for proper cleanup
  registrations_.emplace_back(
      comm,
      pg_id,
      pg_desc,
      std::move(pre_hook_handle),
      std::move(post_hook_handle));
  enabled_ = true;
}

void FlightRecorderHook::unregister() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Call remove() on all handles to properly unregister hooks
  for (auto& reg : registrations_) {
    if (reg.pre_hook_handle) {
      reg.pre_hook_handle->remove();
    }
    if (reg.post_hook_handle) {
      reg.post_hook_handle->remove();
    }
  }
  registrations_.clear();
  enabled_ = false;
}

bool FlightRecorderHook::isP2POp(OpName name) {
  return name == OpName::send || name == OpName::recv;
}

std::vector<int64_t> FlightRecorderHook::getTensorSizes(
    const at::Tensor& tensor) {
  auto sizes = tensor.sizes();
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

std::string FlightRecorderHook::getDtypeString(const at::Tensor& tensor) {
  return c10::toString(tensor.scalar_type());
}

std::optional<size_t> FlightRecorderHook::findEntryIdx(size_t record_id) const {
  // Search for entry with matching record_id
  // Since entries can be overwritten in ring buffer, we need to search
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].record_id == record_id) {
      return i;
    }
  }
  return std::nullopt;
}

void FlightRecorderHook::recordStartEvent(
    Entry& entry,
    const at::Tensor* tensor) {
  if (tensor && !tensor->is_cpu()) {
    c10::Device device = tensor->device();
    entry.device = device;
    auto stream = at::accelerator::getCurrentStream(device.index());
    // Create start event with timing
    entry.start_event = std::make_shared<c10::Event>(
        device.type(), c10::EventFlag::BACKEND_DEFAULT);
    entry.start_event->record(stream);
  }
}

void FlightRecorderHook::recordEndEventAndComputeDuration(Entry& entry) {
  if (entry.start_event && entry.device.has_value()) {
    auto stream = at::accelerator::getCurrentStream(entry.device->index());
    // Create end event with timing
    entry.end_event = std::make_shared<c10::Event>(
        entry.device->type(), c10::EventFlag::BACKEND_DEFAULT);
    entry.end_event->record(stream);
    entry.end_event->synchronize();
    entry.duration_ms =
        static_cast<float>(entry.start_event->elapsedTime(*entry.end_event));
  }
}

time_ns_t FlightRecorderHook::getCurrentTimeNs() const {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

size_t FlightRecorderHook::onPreHook(
    const std::string& comm_name,
    size_t pg_id,
    const std::string& pg_desc,
    const TorchComm::PreHookArgs& args) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!enabled_) {
    return 0; // Not recording
  }

  Entry entry;
  entry.record_id = next_record_id_++;
  entry.pg_id = pg_id;
  entry.pg_name = std::make_tuple(comm_name, pg_desc);
  entry.op_id = op_id_++;
  // Use "nccl:" prefix as expected by the FR trace analyzer
  // (TorchComm wraps NCCL under the hood)
  entry.profiling_name = "nccl:" + std::string(toString(args.name));
  entry.time_created_ns = getCurrentTimeNs();
  entry.is_p2p = isP2POp(args.name);

  if (entry.is_p2p) {
    entry.p2p_seq_id = p2p_seq_id_++;
    entry.collective_seq_id = 0;
  } else {
    entry.collective_seq_id = collective_seq_id_++;
    entry.p2p_seq_id = 0;
  }

  // Extract input tensor info
  if (args.input_tensor != nullptr) {
    entry.input_sizes.push_back(getTensorSizes(*args.input_tensor));
    entry.input_dtypes.push_back(getDtypeString(*args.input_tensor));
  }
  if (args.input_tensors != nullptr) {
    for (const auto& tensor : *args.input_tensors) {
      entry.input_sizes.push_back(getTensorSizes(tensor));
      entry.input_dtypes.push_back(getDtypeString(tensor));
    }
  }

  // Extract output tensor info
  if (args.output_tensor != nullptr) {
    entry.output_sizes.push_back(getTensorSizes(*args.output_tensor));
    entry.output_dtypes.push_back(getDtypeString(*args.output_tensor));
  }
  if (args.output_tensors != nullptr) {
    for (const auto& tensor : *args.output_tensors) {
      entry.output_sizes.push_back(getTensorSizes(tensor));
      entry.output_dtypes.push_back(getDtypeString(tensor));
    }
  }

  // For in-place operations where input == output, copy input to output
  if (entry.output_sizes.empty() && !entry.input_sizes.empty()) {
    entry.output_sizes = entry.input_sizes;
    entry.output_dtypes = entry.input_dtypes;
  }

  // Mark as started since we're in the pre-hook (about to execute)
  entry.time_discovered_started_ns = getCurrentTimeNs();

  // Record CUDA start event if input is CUDA tensor
  if (args.input_tensor != nullptr) {
    recordStartEvent(entry, args.input_tensor);
  } else if (args.input_tensors != nullptr && !args.input_tensors->empty()) {
    recordStartEvent(entry, &(*args.input_tensors)[0]);
  }

  size_t record_id = entry.record_id;

  // Add to ring buffer
  if (entries_.size() < max_entries_) {
    entries_.push_back(std::move(entry));
  } else {
    entries_[next_entry_idx_] = std::move(entry);
    next_entry_idx_ = (next_entry_idx_ + 1) % max_entries_;
  }

  return record_id;
}

void FlightRecorderHook::onPostHook(size_t record_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!enabled_ || record_id == 0) {
    return; // Not recording or invalid record_id
  }

  auto entry_idx = findEntryIdx(record_id);
  if (entry_idx.has_value()) {
    // Record end event and compute CUDA duration
    recordEndEventAndComputeDuration(entries_[*entry_idx]);
    entries_[*entry_idx].time_discovered_completed_ns = getCurrentTimeNs();
    entries_[*entry_idx].retired = true;
  }
}

std::string FlightRecorderHook::dump_json(bool include_completed) const {
  std::lock_guard<std::mutex> lock(mutex_);

  json result;
  result["version"] = kVersion;
  result["pg_config"] = json::object();
  result["pg_status"] = json::object();

  // Build pg_config from registered comms
  for (const auto& reg : registrations_) {
    if (auto comm = reg.comm.lock()) {
      std::string comm_name(comm->getCommName());
      json pg_info;
      pg_info["name"] = comm_name;
      pg_info["desc"] = reg.pg_desc;
      pg_info["ranks"] = ""; // Would need to get ranks from comm
      result["pg_config"][comm_name] = pg_info;
    }
  }

  json entries_json = json::array();

  // Output entries in order (oldest to newest)
  auto add_entry = [&](const Entry& entry) {
    if (!include_completed && entry.time_discovered_completed_ns.has_value()) {
      return;
    }

    json j;
    j["record_id"] = static_cast<int64_t>(entry.record_id);
    j["pg_id"] = static_cast<int64_t>(entry.pg_id);
    j["process_group"] = {
        std::get<0>(entry.pg_name), std::get<1>(entry.pg_name)};
    j["collective_seq_id"] = static_cast<int64_t>(entry.collective_seq_id);
    j["p2p_seq_id"] = static_cast<int64_t>(entry.p2p_seq_id);
    j["op_id"] = static_cast<int64_t>(entry.op_id);
    j["profiling_name"] = entry.profiling_name;
    j["input_sizes"] = entry.input_sizes;
    j["input_dtypes"] = entry.input_dtypes;
    j["output_sizes"] = entry.output_sizes;
    j["output_dtypes"] = entry.output_dtypes;
    j["time_created_ns"] = static_cast<int64_t>(entry.time_created_ns);
    j["time_discovered_started_ns"] =
        entry.time_discovered_started_ns.has_value()
        ? static_cast<int64_t>(*entry.time_discovered_started_ns)
        : 0;
    j["time_discovered_completed_ns"] =
        entry.time_discovered_completed_ns.has_value()
        ? static_cast<int64_t>(*entry.time_discovered_completed_ns)
        : 0;
    j["timeout_ms"] = entry.timeout_ms;
    if (entry.duration_ms.has_value()) {
      j["duration_ms"] = *entry.duration_ms;
    }
    j["is_p2p"] = entry.is_p2p;
    j["retired"] = entry.retired;
    j["state"] = entry.getState();
    j["frames"] = json::array(); // Stack traces not captured in this version

    entries_json.push_back(std::move(j));
  };

  // Handle ring buffer ordering
  if (entries_.size() < max_entries_) {
    // Buffer not full, entries are in order
    for (const auto& entry : entries_) {
      add_entry(entry);
    }
  } else {
    // Buffer is full, start from next_entry_idx_ (oldest)
    for (size_t i = 0; i < entries_.size(); ++i) {
      size_t idx = (next_entry_idx_ + i) % max_entries_;
      add_entry(entries_[idx]);
    }
  }

  if (!entries_json.empty()) {
    result["entries"] = entries_json;
  }

  return result.dump();
}

std::vector<FlightRecorderHook::Entry> FlightRecorderHook::getEntries() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<Entry> result;
  result.reserve(entries_.size());

  if (entries_.size() < max_entries_) {
    result = entries_;
  } else {
    // Return in chronological order
    for (size_t i = 0; i < entries_.size(); ++i) {
      size_t idx = (next_entry_idx_ + i) % max_entries_;
      result.push_back(entries_[idx]);
    }
  }

  return result;
}

void FlightRecorderHook::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  next_entry_idx_ = 0;
  next_record_id_ = 0;
  collective_seq_id_ = 0;
  p2p_seq_id_ = 0;
  op_id_ = 0;
}

bool FlightRecorderHook::isEnabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return enabled_;
}

size_t FlightRecorderHook::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

} // namespace comms
} // namespace torch
