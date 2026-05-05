// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/hooks/chash/ChashHook.hpp"
#include "comms/torchcomms/hooks/common/OpNameHelper.hpp"
#include "comms/torchcomms/hooks/common/SignatureBuilder.hpp"

#include <cassert>
#include <variant>

#include <fmt/core.h>

#if __has_include(<cuda_runtime_api.h>) && __has_include(<ATen/cuda/CUDAContext.h>)
#include <ATen/cuda/CUDAContext.h>
#define CHASH_HAS_CUDA 1
#endif

namespace torch::comms {

namespace {
c10::intrusive_ptr<TorchWork> getWork(const PostHookArgs& args) {
  return std::visit(
      [](const auto& a) -> c10::intrusive_ptr<TorchWork> {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_base_of_v<CollectivePostHookArgs, T>) {
          return a.work.lock();
        }
        return nullptr;
      },
      args);
}
} // namespace

// -- Flush --

void ChashHook::flushAllHashes() {
  for (auto& [comm_name, buf] : comm_hash_buffers_) {
    uint64_t next_empty =
        *static_cast<volatile uint64_t*>(buf->next_empty_hash_entry);

    if (next_empty > buf->next_unflushed_hash_entry + buf->num_hash_entries) {
      size_t overflow =
          next_empty - buf->next_unflushed_hash_entry - buf->num_hash_entries;
      log_file_.writeLine(
          fmt::format("WARN|comm={}|overflow|count={}", comm_name, overflow));
      buf->next_unflushed_hash_entry = next_empty - buf->num_hash_entries;
    }

    for (size_t i = buf->next_unflushed_hash_entry; i < next_empty; ++i) {
      const auto& entry = buf->hash_entries[i % buf->num_hash_entries];
      log_file_.writeLine(
          fmt::format(
              "C{}|{}|hash={:#018x}",
              entry.user_context & LABEL_MASK,
              (entry.user_context & PHASE_MASK) == PHASE_POST ? "E" : "S",
              entry.hash));
    }

    buf->next_unflushed_hash_entry = next_empty;
  }
}

// -- Launch helpers --

void ChashHook::launchHashKernel(
    int device_index,
    const at::Tensor& tensor,
    HashBuffer& buf,
    uint64_t user_context) {
#ifdef CHASH_HAS_CUDA
  if (device_index < 0 || !buf.hash_entries) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
  int tpb = max_threads_per_block_ > 0 ? max_threads_per_block_ : 1024;
  launchHash(
      stream,
      tensor.data_ptr(),
      tensor.nbytes(),
      buf.hash_entries,
      buf.num_hash_entries,
      buf.next_empty_hash_entry,
      user_context,
      num_blocks_,
      tpb);
#endif
}

// -- Pre-hook --

void ChashHook::onPreHook(
    TorchComm* comm,
    size_t op_id,
    const PreHookArgs& args) {
  std::string comm_name(comm->getCommName());
  int device_index = comm->getDevice().index();

  auto op = getOpName(args);
  if (op == OpName::split) {
    log_file_.writeLine(
        buildSplitLine(comm_name, *std::get_if<SplitPreHookArgs>(&args)));
    return;
  }
  if (op == OpName::finalize || op == OpName::new_window ||
      op == OpName::barrier) {
    return;
  }

  auto sig = buildSignature(comm_name, args, true);
  assert(!sig.empty());

  auto buf_it = comm_hash_buffers_.find(comm_name);
  assert(buf_it != comm_hash_buffers_.end());

  uint64_t label = next_label_.fetch_add(1, std::memory_order_relaxed);

  log_file_.writeLine(fmt::format("C{}|sig|{}", label, sig));

  auto& buf = *buf_it->second;

  auto stashOutput = [&](std::vector<at::Tensor> out_tensors) {
    op_labels_[op_id] = label;
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_async_[label] =
        TensorInfo{std::move(out_tensors), label, comm_name, device_index};
  };

  // In-place ops: single tensor is both input and output
  if (auto* a = std::get_if<SendPreHookArgs>(&args)) {
    launchHashKernel(device_index, a->tensor, buf, label | PHASE_PRE);
    stashOutput({a->tensor});
  } else if (auto* a = std::get_if<BroadcastPreHookArgs>(&args)) {
    launchHashKernel(device_index, a->tensor, buf, label | PHASE_PRE);
    stashOutput({a->tensor});
  } else if (auto* a = std::get_if<AllReducePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->tensor, buf, label | PHASE_PRE);
    stashOutput({a->tensor});
  } else if (auto* a = std::get_if<ReducePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->tensor, buf, label | PHASE_PRE);
    stashOutput({a->tensor});
  } else if (auto* a = std::get_if<RecvPreHookArgs>(&args)) {
    stashOutput({a->tensor});

    // Single input, single output
  } else if (auto* a = std::get_if<AllGatherSinglePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output});
  } else if (auto* a = std::get_if<ReduceScatterSinglePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output});
  } else if (auto* a = std::get_if<AllToAllSinglePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output});
  } else if (auto* a = std::get_if<AllToAllVSinglePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output});
  } else if (auto* a = std::get_if<GatherSinglePreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output});

    // Single input, list output
  } else if (auto* a = std::get_if<AllGatherPreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output.begin(), a->output.end()});
  } else if (auto* a = std::get_if<AllGatherVPreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output.begin(), a->output.end()});
  } else if (auto* a = std::get_if<GatherPreHookArgs>(&args)) {
    launchHashKernel(device_index, a->input, buf, label | PHASE_PRE);
    stashOutput({a->output.begin(), a->output.end()});

    // List input, single output
  } else if (auto* a = std::get_if<ReduceScatterPreHookArgs>(&args)) {
    for (const auto& t : a->input) {
      launchHashKernel(device_index, t, buf, label | PHASE_PRE);
    }
    stashOutput({a->output});
  } else if (auto* a = std::get_if<ReduceScatterVPreHookArgs>(&args)) {
    for (const auto& t : a->input) {
      launchHashKernel(device_index, t, buf, label | PHASE_PRE);
    }
    stashOutput({a->output});
  } else if (auto* a = std::get_if<ScatterPreHookArgs>(&args)) {
    for (const auto& t : a->input) {
      launchHashKernel(device_index, t, buf, label | PHASE_PRE);
    }
    stashOutput({a->output});

    // List input, list output
  } else if (auto* a = std::get_if<AllToAllPreHookArgs>(&args)) {
    for (const auto& t : a->input) {
      launchHashKernel(device_index, t, buf, label | PHASE_PRE);
    }
    stashOutput({a->output.begin(), a->output.end()});
  }
}

// -- Post-hook --

void ChashHook::onPostHook(
    TorchComm* comm,
    size_t op_id,
    const PostHookArgs& args) {
  std::string comm_name(comm->getCommName());
  auto work = getWork(args);
  if (!work) {
    if (auto* split = std::get_if<SplitPostHookArgs>(&args)) {
      if (auto new_comm = split->new_comm.lock()) {
        registerWithComm(new_comm);
      }
    }
    return;
  }

  auto label_it = op_labels_.find(op_id);
  if (label_it == op_labels_.end()) {
    return;
  }
  uint64_t label = label_it->second;
  op_labels_.erase(label_it);

  work->registerWorkStartHook([this]() { flushAllHashes(); });
  work->registerWorkEndHook([this]() { flushAllHashes(); });
  work->registerWorkWaitPostHook([this, label]() {
    TensorInfo info;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto pit = pending_async_.find(label);
      if (pit == pending_async_.end()) {
        return;
      }
      info = pit->second;
      pending_async_.erase(pit);
    }
    auto buf_it = comm_hash_buffers_.find(info.comm_name);
    assert(buf_it != comm_hash_buffers_.end());
    for (const auto& t : info.tensors) {
      launchHashKernel(
          info.device_index, t, *buf_it->second, label | PHASE_POST);
    }
  });
}

} // namespace torch::comms
