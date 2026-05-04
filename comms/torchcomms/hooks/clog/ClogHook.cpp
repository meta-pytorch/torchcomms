// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/hooks/clog/ClogHook.hpp"
#include "comms/torchcomms/TorchComm.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <variant>

#include <fmt/core.h>

#if __has_include(<cuda_runtime_api.h>) && __has_include(<ATen/cuda/CUDAContext.h>)
#include <ATen/cuda/CUDAContext.h>
#define CLOG_HAS_CUDA 1
#endif

namespace torch::comms {

namespace {
constexpr int kLogVersion = 1;

bool getAsyncOp(const PreHookArgs& args) {
  return std::visit(
      [](const auto& a) -> bool {
        using T = std::decay_t<decltype(a)>;
        if constexpr (
            std::is_same_v<T, SplitPreHookArgs> ||
            std::is_same_v<T, NewWindowPreHookArgs>) {
          return false;
        } else {
          return a.async_op;
        }
      },
      args);
}
} // namespace

ClogHook::ClogHook(
    const std::string& output,
    const std::vector<std::string>& events,
    const std::vector<std::string>& verbose) {
  // Parse events
  for (const auto& ev : events) {
    if (ev == "ALL" || ev == "LIFECYCLE") {
      log_lifecycle_ = true;
    }
  }

  // Parse verbose options
  for (const auto& v : verbose) {
    if (v == "buffers") {
      log_buffers_ = true;
    }
  }

  log_file_.open(output);

  // Write version header with base timestamp
  base_ts_ = now();
  log_file_.writeLine(
      fmt::format("V|{}|base_timestamp={:.3f}", kLogVersion, base_ts_));
}

ClogHook::~ClogHook() {
  unregister();
}

// -- Registration --

void ClogHook::registerWithComm(std::shared_ptr<TorchComm> comm) {
  log_file_.writeLine(
      fmt::format(
          "new_comm|comm={}|rank={}|world_size={}",
          std::string(comm->getCommName()),
          comm->getRank(),
          comm->getSize()));
  registerHooks(comm);
}

void ClogHook::registerHooks(std::shared_ptr<TorchComm> comm) {
  for (const auto& reg : registrations_) {
    if (reg.comm.lock() == comm) {
      throw std::runtime_error(
          "ClogHook: already registered with comm " +
          std::string(comm->getCommName()));
    }
  }

  std::string comm_name(comm->getCommName());

  int device_index = comm->getDevice().index();
  auto pre_hook_handle = comm->registerPreHook(
      [this, comm_name, device_index](
          OpName name, size_t op_id, const PreHookArgs& args) {
        this->onPreHook(comm_name, device_index, name, op_id, args);
      });

  auto post_hook_handle =
      comm->registerPostHook([this](size_t op_id, const PostHookArgs& args) {
        this->onPostHook(op_id, args);
      });

  auto graph_replay_hook_handle =
      comm->registerGraphReplayHook([this, comm_name](
                                        uint64_t graph_id,
                                        uint64_t replay_id,
                                        void* stream,
                                        size_t collective_index,
                                        std::string_view event) {
        this->onGraphReplayEvent(
            comm_name, graph_id, replay_id, stream, collective_index, event);
      });

  registrations_.push_back(
      CommRegistration{
          .comm = comm,
          .pre_hook_handle = std::move(pre_hook_handle),
          .post_hook_handle = std::move(post_hook_handle),
          .graph_replay_hook_handle = std::move(graph_replay_hook_handle),
      });
}

void ClogHook::unregister() {
  for (auto& reg : registrations_) {
    if (reg.pre_hook_handle) {
      reg.pre_hook_handle->remove();
    }
    if (reg.post_hook_handle) {
      reg.post_hook_handle->remove();
    }
    if (reg.graph_replay_hook_handle) {
      reg.graph_replay_hook_handle->remove();
    }
  }
  registrations_.clear();
}

// -- Formatting helpers (static, identical to original Clog) --

double ClogHook::now() {
  auto tp = std::chrono::system_clock::now();
  auto duration = tp.time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  return static_cast<double>(millis.count()) / 1000.0;
}

std::string_view ClogHook::reduceOpToString(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
      return "sum";
    case ReduceOp::RedOpType::PRODUCT:
      return "prod";
    case ReduceOp::RedOpType::MIN:
      return "min";
    case ReduceOp::RedOpType::MAX:
      return "max";
    case ReduceOp::RedOpType::BAND:
      return "band";
    case ReduceOp::RedOpType::BOR:
      return "bor";
    case ReduceOp::RedOpType::BXOR:
      return "bxor";
    case ReduceOp::RedOpType::PREMUL_SUM:
      return "premul_sum";
    case ReduceOp::RedOpType::AVG:
      return "avg";
    default:
      return "unknown";
  }
}

std::string_view ClogHook::dtypeToString(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "f32";
    case at::ScalarType::Double:
      return "f64";
    case at::ScalarType::Half:
      return "f16";
    case at::ScalarType::BFloat16:
      return "bf16";
    case at::ScalarType::Int:
      return "i32";
    case at::ScalarType::Long:
      return "i64";
    case at::ScalarType::Short:
      return "i16";
    case at::ScalarType::Char:
      return "i8";
    case at::ScalarType::Byte:
      return "u8";
    case at::ScalarType::Bool:
      return "bool";
    case at::ScalarType::Float8_e4m3fn:
      return "fp8e4m3";
    case at::ScalarType::Float8_e5m2:
      return "fp8e5m2";
    default:
      return "other";
  }
}

std::string ClogHook::formatCounts(const std::vector<at::Tensor>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += std::to_string(tensors[i].numel());
  }
  return result;
}

std::string ClogHook::formatCounts(const std::vector<uint64_t>& counts) {
  std::string result;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += std::to_string(counts[i]);
  }
  return result;
}

std::string ClogHook::formatPtr(const void* ptr) {
  return fmt::format("{:#x}", reinterpret_cast<uintptr_t>(ptr));
}

std::string ClogHook::formatPtrs(const std::vector<at::Tensor>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += formatPtr(tensors[i].data_ptr());
  }
  return result;
}

// graph_id is globally unique per the CUDA spec:
// https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g150be2211d73d782bc34c497ddb06f2f
ClogHook::GraphCaptureInfo ClogHook::getGraphCaptureInfo(int device_index) {
#ifdef CLOG_HAS_CUDA
  if (device_index < 0) {
    return {};
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
  cudaStreamCaptureStatus status;
  unsigned long long graph_id = 0;
  cudaError_t err = cudaStreamGetCaptureInfo(stream, &status, &graph_id);
  if (err == cudaSuccess && status == cudaStreamCaptureStatusActive) {
    return {stream, static_cast<uint64_t>(graph_id)};
  }
#endif
  return {};
}

// -- Fake stream management --

void* ClogHook::getFakeStream(const std::string& comm_name) {
  auto id = next_fake_stream_.fetch_add(1, std::memory_order_relaxed);
  auto* fake =
      reinterpret_cast<void*>(~id); // NOLINT(performance-no-int-to-ptr)
  return comm_fake_streams_.findOrInsert(comm_name, fake);
}

// -- Stream resolution for replay hooks --

void* ClogHook::resolveReplayStream(
    uint64_t graph_id,
    const std::string& comm_name,
    void* stream) {
  auto stream_map_opt = graph_collectives_.find(graph_id);
  if (!stream_map_opt) {
    return stream;
  }
  auto sit = stream_map_opt->find(stream);
  if (sit != stream_map_opt->end()) {
    return stream;
  }
  // Unknown stream — must be a comm-internal stream; use fake stream
  auto fake_opt = comm_fake_streams_.find(comm_name);
  if (fake_opt) {
    return *fake_opt;
  }
  return stream;
}

// -- I/O --

void ClogHook::logEvent(uint64_t corr_id, std::string_view event) {
  double delta = now() - base_ts_;
  log_file_.writeLine(fmt::format("C{}|{}|+{:.3f}", corr_id, event, delta));
}

void ClogHook::logGraphEvent(
    uint64_t graph_id,
    uint64_t corr_id,
    std::string_view event) {
  double delta = now() - base_ts_;
  log_file_.writeLine(
      fmt::format("G{}|C{}|{}|+{:.3f}", graph_id, corr_id, event, delta));
}

// -- Core logging --

WorkId ClogHook::logCollective(
    std::string_view comm_name,
    std::string sig_body,
    bool async_op,
    void* stream,
    uint64_t graph_id) {
  auto sig_key = fmt::format("{}|comm={}", sig_body, comm_name);

  auto new_corr_id = next_corr_id_.fetch_add(1, std::memory_order_relaxed);
  auto corr_id = sig_map_.findOrInsert(sig_key, new_corr_id);

  // Track correlation IDs per graph per stream for replay hook lookups
  if (graph_id != kNoGraphCapture) {
    void* stream_key =
        async_op ? getFakeStream(std::string(comm_name)) : stream;
    graph_collectives_.insertOrModify(graph_id, [&](auto& stream_map) {
      stream_map[stream_key].push_back(
          GraphCollective{std::string(comm_name), corr_id});
    });
  }

  if (corr_id == new_corr_id) {
    log_file_.writeLine(fmt::format("C{}|sig|{}", corr_id, sig_key));
  }

  uint64_t work_id = next_work_id_.fetch_add(1, std::memory_order_relaxed);
  if (log_lifecycle_) {
    work_corr_map_.insert(work_id, WorkInfo{corr_id, graph_id});
    work_events_map_.insert(
        work_id,
        async_op ? std::vector<std::string>{"S", "E", "W"}
                 : std::vector<std::string>{"S", "E"});
  }

  auto q_event = fmt::format("Q|work_id={}", work_id);
  if (graph_id != kNoGraphCapture) {
    logGraphEvent(graph_id, corr_id, q_event);
  } else {
    logEvent(corr_id, q_event);
  }

  return work_id;
}

// -- Signature builder via std::visit --

std::string ClogHook::buildSignature(OpName name, const PreHookArgs& args)
    const {
  return std::visit(
      [this, name](const auto& a) -> std::string {
        using T = std::decay_t<decltype(a)>;
        auto op = opToString(name);

        // In-place collectives (single tensor is both input and output)
        if constexpr (std::is_same_v<T, AllReducePreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              count,
              count,
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, BroadcastPreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              count,
              count,
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, ReducePreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|root={}|async_op={}",
              op,
              count,
              count,
              dtype,
              red_op,
              a.root,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;

          // Point-to-point
        } else if constexpr (std::is_same_v<T, SendPreHookArgs>) {
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count=0|dtype={}|peer={}|async_op={}",
              op,
              a.tensor.numel(),
              dtype,
              a.peer,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format("|in={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, RecvPreHookArgs>) {
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count=0|out_count={}|dtype={}|peer={}|async_op={}",
              op,
              a.tensor.numel(),
              dtype,
              a.peer,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format("|out={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;

          // Distinct input/output (single tensors)
        } else if constexpr (
            std::is_same_v<T, AllGatherSinglePreHookArgs> ||
            std::is_same_v<T, AllToAllSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::
                                 is_same_v<T, ReduceScatterSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, AllToAllVSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_splits={}|out_splits={}|dtype={}|async_op={}",
              op,
              formatCounts(a.input_split_sizes),
              formatCounts(a.output_split_sizes),
              dtype,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;

          // Input tensor -> output tensor list
        } else if constexpr (
            std::is_same_v<T, AllGatherPreHookArgs> ||
            std::is_same_v<T, AllGatherVPreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_counts={}|dtype={}|async_op={}",
              op,
              a.input.numel(),
              formatCounts(a.output),
              dtype,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtrs(a.output));
          }
          return sig;

          // Input tensor list -> output tensor
        } else if constexpr (
            std::is_same_v<T, ReduceScatterPreHookArgs> ||
            std::is_same_v<T, ReduceScatterVPreHookArgs>) {
          auto dtype = dtypeToString(a.output.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_counts={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              formatCounts(a.input),
              a.output.numel(),
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtrs(a.input),
                formatPtr(a.output.data_ptr()));
          }
          return sig;

          // Tensor lists on both sides
        } else if constexpr (std::is_same_v<T, AllToAllPreHookArgs>) {
          if (a.input.empty()) {
            return std::string();
          }
          auto dtype = dtypeToString(a.input[0].scalar_type());
          auto sig = fmt::format(
              "{}|in_counts={}|out_counts={}|dtype={}|async_op={}",
              op,
              formatCounts(a.input),
              formatCounts(a.output),
              dtype,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}", formatPtrs(a.input), formatPtrs(a.output));
          }
          return sig;

          // Scatter/gather with root
        } else if constexpr (std::is_same_v<T, ScatterPreHookArgs>) {
          auto dtype = dtypeToString(a.output.scalar_type());
          auto sig = fmt::format(
              "{}|in_counts={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              formatCounts(a.input),
              a.output.numel(),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtrs(a.input),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, GatherPreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_counts={}|dtype={}|root={}|async_op={}",
              op,
              a.input.numel(),
              formatCounts(a.output),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtrs(a.output));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, GatherSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (log_buffers_) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;

          // Special (no tensors)
        } else if constexpr (std::is_same_v<T, BarrierPreHookArgs>) {
          return fmt::format("{}|async_op={}", op, a.async_op ? 't' : 'f');
        } else if constexpr (std::is_same_v<T, BatchOpIssuePreHookArgs>) {
          return fmt::format(
              "{}|num_ops={}|async_op={}",
              op,
              a.num_ops,
              a.async_op ? 't' : 'f');

          // Communicator management (split logged separately)
        } else if constexpr (std::is_same_v<T, SplitPreHookArgs>) {
          return std::string();
        } else if constexpr (std::is_same_v<T, NewWindowPreHookArgs>) {
          return std::string();
        } else {
          return std::string();
        }
      },
      args);
}

// -- Pre-hook --

void ClogHook::onPreHook(
    const std::string& comm_name,
    int device_index,
    OpName name,
    size_t op_id,
    const PreHookArgs& args) {
  // Handle split specially: log the split event, not a signature.
  if (auto* split = std::get_if<SplitPreHookArgs>(&args)) {
    std::string ranks_str;
    for (size_t i = 0; i < split->ranks.size(); ++i) {
      if (i > 0) {
        ranks_str += ',';
      }
      ranks_str += std::to_string(split->ranks[i]);
    }
    log_file_.writeLine(
        fmt::format(
            "split|parent={}|child={}|ranks={}",
            comm_name,
            split->name,
            ranks_str));
    return;
  }

  // Skip new_window (no logging needed)
  if (std::get_if<NewWindowPreHookArgs>(&args)) {
    return;
  }

  auto sig = buildSignature(name, args);
  if (sig.empty()) {
    return;
  }

  bool async_op = getAsyncOp(args);
  auto [stream, graph_id] = getGraphCaptureInfo(device_index);
  auto work_id =
      logCollective(comm_name, std::move(sig), async_op, stream, graph_id);

  if (work_id != kWorkIdInvalid) {
    op_to_work_.insert(op_id, work_id);
  }
}

// -- Post-hook --

void ClogHook::onPostHook(size_t op_id, const PostHookArgs& args) {
  // For split, register the new communicator.
  if (auto* split = std::get_if<SplitPostHookArgs>(&args)) {
    if (auto new_comm = split->new_comm.lock()) {
      registerHooks(new_comm);
    }
    return;
  }

  // TODO: add logging for window operations.
  if (std::get_if<NewWindowPostHookArgs>(&args)) {
    return;
  }

  // Retrieve the work_id assigned in the pre-hook.
  WorkId work_id = op_to_work_.findAndErase(op_id).value_or(kWorkIdInvalid);

  if (work_id == kWorkIdInvalid) {
    log_file_.writeLine(
        fmt::format("WARN|post-hook missing work_id for op_id {}", op_id));
    return;
  }

  // For collectives with a work object, register lifecycle hooks.
  std::visit(
      [this, work_id](const auto& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_base_of_v<CollectivePostHookArgs, T>) {
          if (auto work = a.work.lock()) {
            if (!log_lifecycle_) {
              return;
            }
            work->registerWorkStartHook(
                [this, work_id]() { logLifecycleEvent(work_id, "S"); });
            work->registerWorkEndHook(
                [this, work_id]() { logLifecycleEvent(work_id, "E"); });
            work->registerWorkWaitHook(
                [this, work_id]() { logLifecycleEvent(work_id, "W"); });
          }
        }
      },
      args);
}

// -- Lifecycle events --

void ClogHook::logLifecycleEvent(WorkId work_id, std::string_view event) {
  if (work_id == kWorkIdInvalid) {
    return;
  }

  auto work_info_opt = work_corr_map_.find(work_id);
  if (!work_info_opt) {
    log_file_.writeLine(
        fmt::format(
            "WARN|lifecycle event {} for unknown work_id {}", event, work_id));
    return;
  }
  auto corr_id = work_info_opt->corr_id;
  auto graph_id = work_info_opt->graph_id;

  if (!work_events_map_.valueRemove(work_id, std::string(event))) {
    log_file_.writeLine(
        fmt::format(
            "WARN|unexpected lifecycle event {} for work_id {}",
            event,
            work_id));
    return;
  }

  // Clean up corr_id entry if no more events remain
  if (!work_events_map_.find(work_id)) {
    work_corr_map_.findAndErase(work_id);
  }

  if (graph_id != kNoGraphCapture) {
    logGraphEvent(graph_id, corr_id, event);
  } else {
    logEvent(corr_id, event);
  }
}

// -- Graph replay events --

void ClogHook::onGraphReplayEvent(
    const std::string& comm_name,
    uint64_t graph_id,
    uint64_t replay_id,
    void* stream,
    size_t collective_index,
    std::string_view event) {
  void* resolved = resolveReplayStream(graph_id, comm_name, stream);

  auto stream_map_opt = graph_collectives_.find(graph_id);
  if (!stream_map_opt) {
    log_file_.writeLine(
        fmt::format(
            "WARN|graph replay event for unknown graph_id {}", graph_id));
    return;
  }
  auto sit = stream_map_opt->find(resolved);
  if (sit == stream_map_opt->end() || collective_index >= sit->second.size()) {
    log_file_.writeLine(
        fmt::format(
            "WARN|graph replay event for unknown stream in graph_id {} index {}",
            graph_id,
            collective_index));
    return;
  }

  auto corr_id = sit->second[collective_index].corr_id;

  if (!log_lifecycle_) {
    return;
  }

  double delta = now() - base_ts_;
  log_file_.writeLine(
      fmt::format(
          "G{}|R{}|C{}|{}|+{:.3f}",
          graph_id,
          replay_id,
          corr_id,
          event,
          delta));
}

} // namespace torch::comms
