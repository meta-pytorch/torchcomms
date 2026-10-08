/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/relay_helper_devices.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <string>
#include <utility>

#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "utils.h"

namespace rcclx::relay {

namespace {

constexpr int kMaxCandidates = 16;

struct CandidateRecord {
  int32_t count;
  int64_t busIds[kMaxCandidates];
};

bool deviceAllowed(int64_t busId) {
  const char* env = ncclGetEnv("NCCL_RELAY_HELPER_DEVICES");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  const std::string list(env);
  size_t start = 0;
  while (start <= list.size()) {
    size_t end = list.find(',', start);
    if (end == std::string::npos) {
      end = list.size();
    }
    const std::string item = list.substr(start, end - start);
    int64_t id = 0;
    if (!item.empty() && busIdToInt64(item.c_str(), &id) == ncclSuccess &&
        id == busId) {
      return true;
    }
    start = end + 1;
  }
  return false;
}

// (bus ID, device ordinal) of every visible device.
std::vector<std::pair<int64_t, int>> visibleDevices() {
  std::vector<std::pair<int64_t, int>> out;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    return out;
  }
  for (int device = 0; device < count; ++device) {
    int64_t busId = 0;
    if (getBusId(device, &busId) == ncclSuccess) {
      out.emplace_back(busId, device);
    }
  }
  return out;
}

bool inComm(ncclComm_t comm, int64_t busId) {
  for (int r = 0; r < comm->nRanks; ++r) {
    if (comm->peerInfo[r].busId == busId) {
      return true;
    }
  }
  return false;
}

} // namespace

RelayDeviceGuard::RelayDeviceGuard(int device) {
  if (hipGetDevice(&previous_) == hipSuccess && previous_ != device) {
    (void)hipSetDevice(device);
  }
}

RelayDeviceGuard::~RelayDeviceGuard() {
  if (previous_ >= 0) {
    (void)hipSetDevice(previous_);
  }
}

void enableRelayPeerAccess(int device) {
  const hipError_t result = hipDeviceEnablePeerAccess(device, 0);
  if (result == hipSuccess) {
    return;
  }
  (void)hipGetLastError();
  if (result != hipErrorPeerAccessAlreadyEnabled) {
    int current = -1;
    (void)hipGetDevice(&current);
    WARN(
        "Relay: enabling peer access from device %d to helper device %d failed: %s; relay traffic through it takes the slow IPC path",
        current,
        device,
        hipGetErrorString(result));
  }
}

ncclResult_t agreeRelayHelperDevices(
    ncclComm_t comm,
    int maxHelpers,
    std::vector<int>* devices) {
  std::vector<std::pair<int64_t, int>> mine;
  for (const auto& [busId, device] : visibleDevices()) {
    int canAccess = 0;
    if (inComm(comm, busId) || !deviceAllowed(busId) ||
        hipDeviceCanAccessPeer(&canAccess, comm->cudaDev, device) !=
            hipSuccess ||
        canAccess == 0) {
      continue;
    }
    mine.emplace_back(busId, device);
  }
  std::sort(mine.begin(), mine.end());
  if (mine.size() > static_cast<size_t>(kMaxCandidates)) {
    mine.resize(kMaxCandidates);
  }
  std::vector<CandidateRecord> records(comm->nRanks);
  records[comm->rank].count = static_cast<int32_t>(mine.size());
  for (size_t i = 0; i < mine.size(); ++i) {
    records[comm->rank].busIds[i] = mine[i].first;
  }
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, records.data(), sizeof(CandidateRecord)));
  devices->clear();
  for (const auto& [busId, device] : mine) {
    bool everyone = true;
    for (const CandidateRecord& record : records) {
      everyone = everyone &&
          std::find(record.busIds, record.busIds + record.count, busId) !=
              record.busIds + record.count;
    }
    if (everyone && static_cast<int>(devices->size()) < maxHelpers) {
      devices->push_back(device);
    }
  }
  return ncclSuccess;
}

} // namespace rcclx::relay
