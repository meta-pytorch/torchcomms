// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Implements TorchCommRCCLX::getInitHandle() and TorchCommRCCLX::reconfigure()
// aligned with the NCCL reference in TorchCommNCCLReconfigure.cpp.
//
// Each rank's init handle encodes its rank, uuid, and store address from the
// previous configuration.  findQuorum() groups handles by uuid to find the
// largest set of ranks that share a common previous communicator.
//
// Cases (determined by quorum analysis):
//
// 1. Empty quorum (fresh init or all ranks are new): abort old comm if any,
//    bootstrap via commInitRankConfig using the shared reconfigure_store_.
//
// 2. Identity reconfigure (quorum == new world size, no new members): the old
//    comm may be unhealthy (e.g. revoked after abort()), so treat the same as
//    case 1: abort and call commInitRankConfig.
//
// 3. In quorum with shrink and/or grow: call commShrink to remove departed
//    ranks, then commGrow to add new ones.  Rank 0 of the post-shrink comm
//    publishes the grow unique ID via a prefix store.
//
// 4. Not in quorum (new rank joining): wait for quorum rank 0 to publish the
//    unique ID, then call commGrow(nullptr, ...) to join.

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ATen/hip/HIPContext.h> // @manual=//caffe2:ATen-custom-hip
#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu

#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/torchcomms/utils/StoreManager.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

// ---------------------------------------------------------------------------
// Handle encoding / quorum logic (mirrors TorchCommNCCLReconfigure.cpp)
// ---------------------------------------------------------------------------

InitHandle TorchCommRCCLX::getInitHandle() const {
  std::string storeAddr;
  if (reconfigure_store_) {
    auto* tcpStore = dynamic_cast<c10d::TCPStore*>(reconfigure_store_.get());
    if (!tcpStore) {
      auto* prefixStore =
          dynamic_cast<c10d::PrefixStore*>(reconfigure_store_.get());
      if (prefixStore) {
        tcpStore = dynamic_cast<c10d::TCPStore*>(
            prefixStore->getUnderlyingNonPrefixStore().get());
      }
    }
    if (tcpStore) {
      storeAddr =
          fmt::format("{}:{}", tcpStore->getHost(), tcpStore->getPort());
    }
  }
  int rank = nccl_comm_ ? rank_ : static_cast<int>(query_ranksize().first);
  return fmt::format("rcclx:{}:{}:{}", rank, uuid_, storeAddr);
}

namespace {

struct HandleInfo {
  int rank;
  int64_t uuid;
  std::string storeAddr;
};

HandleInfo parseHandle(const InitHandle& handle) {
  // Format: "rcclx:<rank>:<uuid>:<storeAddr>"
  auto first = handle.find(':');
  if (first == std::string::npos) {
    return {-1, -1, ""};
  }
  auto second = handle.find(':', first + 1);
  if (second == std::string::npos) {
    return {-1, -1, ""};
  }
  int rank = std::stoi(handle.substr(first + 1, second - first - 1));
  auto third = handle.find(':', second + 1);
  if (third == std::string::npos) {
    return {rank, std::stoll(handle.substr(second + 1)), ""};
  }
  int64_t uuid = std::stoll(handle.substr(second + 1, third - second - 1));
  std::string storeAddr = handle.substr(third + 1);
  return {rank, uuid, storeAddr};
}

struct QuorumInfo {
  int64_t uuid = -1;
  std::unordered_set<int> ranks;
  size_t newMemberCount = 0;
};

int findRankInHandles(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles,
    const InitHandle& myHandle) {
  return std::visit(
      [&](const auto& h) -> int {
        int result = -1;
        int idx = 0;
        for (const auto& handle : h) {
          if (handle == myHandle) {
            if (result >= 0) {
              return -1;
            }
            result = idx;
          }
          idx++;
        }
        return result;
      },
      handles);
}

QuorumInfo findQuorum(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles) {
  std::unordered_map<int64_t, std::vector<int>> groupByUuid;
  size_t totalHandles = 0;

  auto processHandle = [&](const InitHandle& handle) {
    totalHandles++;
    auto info = parseHandle(handle);
    groupByUuid[info.uuid].push_back(info.rank);
  };

  std::visit(
      [&](const auto& h) {
        for (const auto& handle : h) {
          processHandle(handle);
        }
      },
      handles);

  QuorumInfo quorum;
  for (const auto& [uuid, ranks] : groupByUuid) {
    if (uuid < 0) {
      continue;
    }
    std::unordered_set<int> uniqueRanks(ranks.begin(), ranks.end());
    if (uniqueRanks.size() != ranks.size()) {
      continue; // Duplicate ranks in this uuid group — skip
    }
    if (uniqueRanks.size() > quorum.ranks.size()) {
      quorum.uuid = uuid;
      quorum.ranks = uniqueRanks;
    }
  }

  quorum.newMemberCount = totalHandles - quorum.ranks.size();
  return quorum;
}

} // namespace

// ---------------------------------------------------------------------------
// reconfigure()
// ---------------------------------------------------------------------------

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::reconfigure(
    const ReconfigureOptions& opts) {
  TC_LOG(INFO, this) << "TorchCommRCCLX reconfigure starting";
  TracingGuard tracingGuard(name_, comm_size_, "reconfigure", rank_);

  int new_size = static_cast<int>(
      std::visit([](const auto& h) { return h.size(); }, opts.handles));
  auto reconfigureTimeout = opts.timeout.value_or(options_.timeout);

  auto quorum = findQuorum(opts.handles);
  bool inQuorum = nccl_comm_ && uuid_ >= 0 && uuid_ == quorum.uuid;

  // Fall back to fresh init when shrink/grow has no advantage:
  // - Single-rank quorum: a 1-rank comm has no bootstrap networking, so
  //   commGrow will fail. Must clear unconditionally (not just when
  //   inQuorum) so all ranks take the same fresh init path.
  // - Identity reconfigure: same world size, no membership change — old comm
  //   may be unhealthy (e.g. revoked after abort()).
  if (quorum.ranks.size() == 1 ||
      (inQuorum && quorum.ranks.size() == static_cast<size_t>(new_size) &&
       quorum.newMemberCount == 0)) {
    inQuorum = false;
    quorum.ranks.clear();
  }

  // Clean up the existing communicator before any reconfiguration.
  if (nccl_comm_) {
    // Check for in-flight work before deciding whether to revoke.
    // Mirrors NCCL's workq_.garbageCollect() pattern: snapshot the queued
    // work items (under the mutex), then query their HIP event status without
    // holding the lock (checkStatus() calls hipEventQuery).
    bool workInFlight = false;
    {
      std::vector<c10::intrusive_ptr<TorchWorkRCCLX>> pendingWork;
      {
        std::lock_guard<std::mutex> lock(work_queues_mutex_);
        for (auto& [stream, q] : stream_work_queues_) {
          auto tmp = q;
          while (!tmp.empty()) {
            pendingWork.push_back(tmp.front());
            tmp.pop();
          }
        }
      }
      for (auto& work : pendingWork) {
        auto s = work->checkStatus();
        if (s == TorchWork::WorkStatus::NOT_STARTED ||
            s == TorchWork::WorkStatus::INPROGRESS) {
          workInFlight = true;
          break;
        }
      }
    }

    // Only revoke when this rank is leaving the quorum (new joiner or fresh
    // init) AND there is work in flight that needs to be cancelled. In-quorum
    // ranks keep the comm alive for commShrink; revoking it would invalidate
    // the comm before shrink runs.
    if (!inQuorum && workInFlight) {
      RCCLX_CHECK_IGNORE(
          rcclx_api_,
          rcclx_api_->commRevoke(nccl_comm_),
          "RCCLX commRevoke failed during reconfigure");
    }

    detachMemoryHook();

    if (timeout_thread_.joinable()) {
      shutdown_ = true;
      {
        std::lock_guard<std::mutex> lock(timeout_mutex_);
        timeout_cv_.notify_all();
      }
      timeout_thread_.join();
    }

    {
      std::lock_guard<std::mutex> lock(work_queues_mutex_);
      stream_work_queues_.clear();
      std::queue<c10::intrusive_ptr<TorchWorkRCCLX>> empty;
      std::swap(completed_works_, empty);
    }

    if (!inQuorum) {
      // This rank is not in the quorum (new joiner, or identity/fresh-init).
      // Abort the old comm so we can bootstrap a fresh one.
      RCCLX_CHECK_IGNORE(
          rcclx_api_,
          rcclx_api_->commAbort(nccl_comm_),
          "RCCLX commAbort failed during reconfigure");
      nccl_comm_ = nullptr;
    }
  }

  if (quorum.ranks.empty()) {
    // -----------------------------------------------------------------------
    // Case 1 & 2: Fresh init or identity reconfigure.
    // All ranks bootstrap a new communicator via commInitRankConfig using the
    // shared reconfigure_store_ to exchange the unique ID.
    // -----------------------------------------------------------------------
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;

    HIP_CHECK(
        hip_api_,
        hip_api_->setDevice(device_.index()),
        fmt::format("Failed to set HIP device to {}", device_.index()));

    ncclUniqueId uniqueId{};
    int myRank = findRankInHandles(opts.handles, getInitHandle());

    if (new_size > 1) {
      auto store = c10::make_intrusive<c10d::PrefixStore>(
          fmt::format("{}/{}", name_, opts.uuid), reconfigure_store_);

      if (myRank < 0) {
        myRank = static_cast<int>(store->add("rank_counter", 1)) - 1;
      }

      if (myRank == 0) {
        RCCLX_CHECK(
            rcclx_api_,
            nccl_comm_,
            rcclx_api_->getUniqueId(&uniqueId),
            "RCCLX getUniqueId failed during reconfigure");
        std::vector<uint8_t> vec(
            reinterpret_cast<uint8_t*>(&uniqueId),
            reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
        store->set("unique_id", vec);
      } else {
        store->wait({"unique_id"}, reconfigureTimeout);
        auto vec = store->get("unique_id");
        std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));
      }
    } else {
      // Single-rank comm: no bootstrap networking needed, every rank
      // creates a private 1-rank communicator with myRank=0.
      RCCLX_CHECK(
          rcclx_api_,
          nccl_comm_,
          rcclx_api_->getUniqueId(&uniqueId),
          "RCCLX getUniqueId failed during reconfigure");
      if (myRank < 0) {
        myRank = 0;
      }
    }

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclComm_t new_comm = nullptr;
    RCCLX_CHECK(
        rcclx_api_,
        nccl_comm_,
        rcclx_api_->commInitRankConfig(
            &new_comm, new_size, uniqueId, myRank, &config),
        "RCCLX commInitRankConfig failed during reconfigure");
    nccl_comm_ = new_comm;

    initRcclxResources();

  } else if (inQuorum) {
    // -----------------------------------------------------------------------
    // Case 3: This rank is in the quorum — shrink departed ranks, then grow
    //         to add new members if any.
    // -----------------------------------------------------------------------
    ncclComm_t current = nccl_comm_;

    if (quorum.ranks.size() < static_cast<size_t>(comm_size_)) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
      // Build the exclude list: ranks in the old comm that are NOT in quorum.
      std::vector<int> excludeRanks;
      for (int r = 0; r < comm_size_; ++r) {
        if (quorum.ranks.find(r) == quorum.ranks.end()) {
          excludeRanks.push_back(r);
        }
      }

      ncclComm_t shrunk = nullptr;
      RCCLX_CHECK(
          rcclx_api_,
          current,
          rcclx_api_->commShrink(
              current,
              excludeRanks.data(),
              static_cast<int>(excludeRanks.size()),
              &shrunk,
              nullptr,
              NCCL_SHRINK_ABORT),
          "RCCLX commShrink failed during reconfigure");
      current = shrunk;
#else
      throw std::runtime_error(
          "RCCLX commShrink (NCCL_SHRINK_ABORT) requires RCCL >= 2.27; "
          "reconfigure with shrink is not supported on this RCCL version");
#endif
    }

    if (quorum.newMemberCount > 0) {
      // Rank 0 of the post-shrink comm publishes the grow unique ID.
      int currentRank = 0;
      RCCLX_CHECK(
          rcclx_api_,
          current,
          rcclx_api_->commUserRank(current, &currentRank),
          "RCCLX commUserRank failed during grow");

      if (currentRank == 0) {
        ncclUniqueId uniqueId{};
        RCCLX_CHECK(
            rcclx_api_,
            current,
            rcclx_api_->commGetUniqueId(current, &uniqueId),
            "RCCLX commGetUniqueId failed during grow");

        auto store = c10::make_intrusive<c10d::PrefixStore>(
            fmt::format("{}/{}", name_, opts.uuid), reconfigure_store_);
        std::vector<uint8_t> vec(
            reinterpret_cast<uint8_t*>(&uniqueId),
            reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
        store->set("unique_id", vec);
      }

      ncclComm_t grown = nullptr;
      RCCLX_CHECK(
          rcclx_api_,
          current,
          rcclx_api_->commGrow(current, new_size, nullptr, -1, &grown, nullptr),
          "RCCLX commGrow failed during reconfigure");
      current = grown;
    }

    nccl_comm_ = current;
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;
    initRcclxResources();

  } else {
    // -----------------------------------------------------------------------
    // Case 4: New rank joining an existing quorum via commGrow.
    // -----------------------------------------------------------------------
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;

    int quorumSize = static_cast<int>(quorum.ranks.size());
    auto store = c10::make_intrusive<c10d::PrefixStore>(
        fmt::format("{}/{}", name_, opts.uuid), reconfigure_store_);
    store->wait({"unique_id"}, reconfigureTimeout);

    auto vec = store->get("unique_id");
    ncclUniqueId uniqueId{};
    std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));

    int growRank =
        quorumSize + static_cast<int>(store->add("rank_counter", 1)) - 1;

    ncclComm_t new_comm = nullptr;
    RCCLX_CHECK(
        rcclx_api_,
        nccl_comm_,
        rcclx_api_->commGrow(
            nullptr, new_size, &uniqueId, growRank, &new_comm, nullptr),
        "RCCLX commGrow failed for new rank during reconfigure");

    nccl_comm_ = new_comm;
    initRcclxResources();
  }

  init_state_ = InitializationState::INITIALIZED;
  uuid_ = opts.uuid;

  TC_LOG(INFO, this) << "TorchCommRCCLX reconfigure completed for rank: "
                     << rank_;

  return c10::make_intrusive<TorchWorkCompleted>();
}

} // namespace torch::comms
