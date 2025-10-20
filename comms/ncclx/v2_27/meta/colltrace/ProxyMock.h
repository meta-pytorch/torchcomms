// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef PROXY_MOCK_H
#define PROXY_MOCK_H

#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include "comms/utils/StrUtils.h"
#include "meta/wrapper/DataTypeStrUtils.h"

#define PROXYMOCK_MATCH_ANY (-1)

class NetSendFailureCond {
 public:
  uint64_t commHash;
  uint64_t opCount;
  int proxyOpId;
  int rank;
  int remoteRank;
  int step;

  NetSendFailureCond() {}
  NetSendFailureCond(
      uint64_t commHash,
      uint64_t opCount,
      int proxyOpId,
      int rank,
      int remoteRank,
      int step)
      : commHash(commHash),
        opCount(opCount),
        proxyOpId(proxyOpId),
        rank(rank),
        remoteRank(remoteRank),
        step(step){};

  std::string toString() {
    std::stringstream ss;
    ss << "commHash:" << hashToHexStr(commHash) << ", opCount:" << opCount
       << ", proxyOpId:" << proxyOpId << ", rank:" << rank
       << ", remoteRank:" << remoteRank << ", step:" << step;
    return ss.str();
  }

  bool operator==(const NetSendFailureCond& c) const {
    if (c.commHash == this->commHash && c.opCount == this->opCount &&
        c.proxyOpId == this->proxyOpId && c.rank == this->rank &&
        c.remoteRank == this->remoteRank) {
      return true;
    } else {
      return false;
    }
  }
};

template <>
struct std::hash<NetSendFailureCond> {
  size_t operator()(const NetSendFailureCond& x) const {
    return x.commHash ^ x.opCount ^ x.proxyOpId ^ x.rank ^ x.remoteRank ^
        x.step;
  }
};

// Allow proxy thread to mock a send failure if the current send operation
// matches user specified config (see NCCL_PROXYMOCK_NET_SEND_FAILURE).
class ProxyMockNetSendFailure {
 public:
  ProxyMockNetSendFailure(ProxyMockNetSendFailure& other) = delete;
  ProxyMockNetSendFailure& operator=(const ProxyMockNetSendFailure&) = delete;

  // Return failure config serialized as json format string
  std::string serialize(bool quoted = false);

  // Return true if mocked, otherwise return false.
  static bool mock(struct ncclProxySubArgs* sub, int step, void** request);

  // Return singleton instance
  static ProxyMockNetSendFailure& getInstance();

  // Reset any existing state in the mock instance and reinitialize based on
  // NCCL_PROXYMOCK_NET_SEND_FAILURE. For testing only where we want to change
  // NCCL_PROXYMOCK_NET_SEND_FAILURE during a process lifetime. Note that direct
  // call to initialize is not thread safe. Test should call it only after
  // destroyed all existing communicators to ensure all proxy threads have been
  // terminated.
  void initialize();

 private:
  ProxyMockNetSendFailure();
  ~ProxyMockNetSendFailure(){};

  bool mockImpl(struct ncclProxySubArgs* sub, int step, void** request);

  bool enabled_{false};
  uint64_t opCount_{0};
  int rank_{0};
  int remoteRank_{0};
  int step_{0};
  int numMatch_{0};
  int delaySec_{0};
  std::mutex mutex_; // protect initialize/reset and mock by multiple threads
  std::unordered_map<
      NetSendFailureCond,
      std::chrono::time_point<std::chrono::high_resolution_clock>>
      mockStartMap_;
  int numMatched_{0};
};

#endif
