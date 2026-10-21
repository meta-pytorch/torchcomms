// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/v2/CollTraceV2.h"

#include <utility>

#include <cuda_runtime.h> // @manual

#include "comms/utils/colltrace/nicstats/NicStatsAttach.h"
#include "comms/utils/colltrace/nicstats/NicStatsLatest.h"
#include "comms/utils/colltrace/v2/CollEvents.h"
#include "comms/utils/colltrace/v2/CommState.h"
#include "comms/utils/colltrace/v2/Env.h"
#include "comms/utils/colltrace/v2/Log.h"
#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"

namespace meta::comms::colltracev2 {

namespace {

template <typename F>
void forEachMode(std::string_view modes, F&& f) {
  while (!modes.empty()) {
    const auto comma = modes.find(',');
    f(modes.substr(0, comma));
    modes = comma == std::string_view::npos ? std::string_view{}
                                            : modes.substr(comma + 1);
  }
}

bool turnsOn(std::string_view mode) {
  return mode == "all" || mode == "ALL" || mode == "collevents";
}

} // namespace

bool selected() {
  static const bool kSelected = envUint("NCCL_COLLTRACE_VERSION", 1) == 2;
  return kSelected;
}

bool validMode(std::string_view mode) {
  return mode == "none" || turnsOn(mode);
}

bool enabled() {
  static const bool kEnabled = [] {
    if (!selected()) {
      return false;
    }
    bool on = false;
    forEachMode(envStr("NCCL_COLLTRACE"), [&on](std::string_view mode) {
      on = on || turnsOn(mode);
    });
    return on;
  }();
  return kEnabled;
}

CollTraceV2& CollTraceV2::instance() {
  // Leaked: captured graphs may replay into its rings until process exit.
  static auto* inst = new CollTraceV2();
  return *inst;
}

CollTraceV2::CollTraceV2()
    : CollTraceV2(
          [](std::shared_ptr<colltrace::nicstats::NicStatsLatest> latest)
              -> std::shared_ptr<void> {
            int cudaDev = -1;
            if (cudaGetDevice(&cudaDev) != cudaSuccess) {
              return nullptr;
            }
            return colltrace::nicstats::startNicStats(
                cudaDev, std::move(latest));
          }) {}

CollTraceV2::CollTraceV2(const NicStatsStarter& startNicStats)
    : ringEntries_(
          static_cast<uint32_t>(envUint("COLLTRACE_RING_ENTRIES", 65536))),
      collHistory_(
          static_cast<std::size_t>(
              envUint("COLLTRACE_COLL_HISTORY", uint64_t{64} << 20))),
      nicLatest_(std::make_shared<colltrace::nicstats::NicStatsLatest>()),
      nicReaders_(startNicStats(nicLatest_)),
      collEvents_(std::make_unique<CollEvents>()) {}

CollTraceV2::~CollTraceV2() = default;

bool CollTraceV2::registerComm(std::string_view commDesc) {
  // First use launches a kernel, which must not happen inside graph capture;
  // taken before commsMu_ so the hot-path lookups never wait on it.
  ::hrdw_ring_buffer::GlobaltimerCalibration::get();
  std::string key{commDesc};
  std::lock_guard<std::mutex> lock(commsMu_);
  if (commsByDesc_.contains(key)) {
    COLLTRACE_LOG_WARN() << "commDesc '" << key << "' registered twice";
    return false;
  }
  auto ring = std::make_unique<Ring>(ringEntries_);
  if (!ring->valid()) {
    COLLTRACE_LOG_ERROR() << "failed to allocate ring for commDesc '" << key
                          << "'";
    return false;
  }
  // kMaxCommIndex itself is never used: with a saturated counter it would form
  // COLLTRACE_INVALID_COLLID.
  if (nextCommIndex_ >= kMaxCommIndex) {
    COLLTRACE_LOG_ERROR() << "out of communicator indices; commDesc '" << key
                          << "' will not be traced";
    return false;
  }
  const auto index = nextCommIndex_++;
  auto state = std::make_shared<CommState>(key, index, std::move(ring));
  commsByDesc_.emplace(key, state);
  commsByIndex_.emplace(index, state);
  collEvents_->watch(std::move(state));
  return true;
}

void CollTraceV2::deregisterComm(std::string_view commDesc) {
  std::lock_guard<std::mutex> lock(commsMu_);
  auto it = commsByDesc_.find(std::string{commDesc});
  if (it == commsByDesc_.end()) {
    return;
  }
  const auto index = it->second->index;
  collEvents_->unwatch(index);
  commsByIndex_.erase(index);
  commsByDesc_.erase(it);
}

std::shared_ptr<CommState> CollTraceV2::find(std::string_view commDesc) {
  std::lock_guard<std::mutex> lock(commsMu_);
  auto it = commsByDesc_.find(std::string{commDesc});
  return it == commsByDesc_.end() ? nullptr : it->second;
}

std::shared_ptr<CommState> CollTraceV2::find(uint32_t commIndex) {
  std::lock_guard<std::mutex> lock(commsMu_);
  auto it = commsByIndex_.find(commIndex);
  return it == commsByIndex_.end() ? nullptr : it->second;
}

uint64_t CollTraceV2::newCollId(std::string_view commDesc) {
  auto state = find(commDesc);
  if (state == nullptr) {
    return COLLTRACE_INVALID_COLLID;
  }
  std::lock_guard<std::mutex> lock(state->mu);
  const auto collId = makeCollId(state->index, state->nextLocalCollId++);
  state->latestCollId = collId;
  state->kernelCounts.push_back(0);
  if (state->kernelCounts.size() > collHistory_) {
    state->kernelCounts.pop_front();
    ++state->firstCountedLocal;
  }
  return collId;
}

colltrace::ColltraceDeviceHandle CollTraceV2::getKernelHandle(uint64_t collId) {
  if (collId == COLLTRACE_INVALID_COLLID) {
    return {};
  }
  auto state = find(static_cast<uint32_t>(collId >> kLocalBits));
  if (state == nullptr) {
    return {};
  }
  colltrace::ColltraceDeviceHandle handle{};
  handle.collId = static_cast<uint32_t>(collId);
  handle.emitStart = true;
  handle.emitEnd = true;
  handle.ring = state->ring->deviceHandle();
  std::lock_guard<std::mutex> lock(state->mu);
  if (auto* count = state->kernelCountFor(collId & kLocalMask)) {
    ++*count;
  }
  return handle;
}

uint64_t CollTraceV2::getLatestCollId(std::string_view commDesc) {
  auto state = find(commDesc);
  if (state == nullptr) {
    return COLLTRACE_INVALID_COLLID;
  }
  std::lock_guard<std::mutex> lock(state->mu);
  return state->latestCollId;
}

int CollTraceV2::getNumKernels(uint64_t collId) {
  if (collId == COLLTRACE_INVALID_COLLID) {
    return -1;
  }
  auto state = find(static_cast<uint32_t>(collId >> kLocalBits));
  if (state == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(state->mu);
  const auto* count = state->kernelCountFor(collId & kLocalMask);
  return count == nullptr ? -1 : static_cast<int>(*count);
}

std::vector<colltrace::nicstats::NicStatsReading> CollTraceV2::nicStats()
    const {
  return nicLatest_->snapshot();
}

std::vector<Event> CollTraceV2::getCollEvents(std::string_view commDesc) {
  auto state = find(commDesc);
  if (state == nullptr) {
    return {};
  }
  return collEvents_->collectEvents(*state);
}

} // namespace meta::comms::colltracev2
