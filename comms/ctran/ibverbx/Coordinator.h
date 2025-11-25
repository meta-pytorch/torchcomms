// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <unordered_map>

#include <folly/Expected.h>
#include "comms/ctran/ibverbx/IbvCommon.h"

namespace ibverbx {

class IbvVirtualQp;
class IbvVirtualCq;

// Coordinator class responsible for routing commands and responses between
// IbvVirtualQp and IbvVirtualCq. Maintains mappings from physical QP numbers to
// IbvVirtualQp pointers, and from virtual CQ numbers to IbvVirtualCq pointers.
// Acts as a router to forward requests between these two classes.
//
// NOTE: The Coordinator APIs are NOT thread-safe. Users must ensure proper
// synchronization when accessing Coordinator methods from multiple threads.
// Thread-safe support can be added in the future if needed.
class Coordinator {
 public:
  Coordinator() = default;
  ~Coordinator() = default;

  // Disable copy constructor and assignment operator
  Coordinator(const Coordinator&) = delete;
  Coordinator& operator=(const Coordinator&) = delete;

  // Allow default move constructor and assignment operator
  Coordinator(Coordinator&&) = default;
  Coordinator& operator=(Coordinator&&) = default;

  inline void submitRequestToVirtualCq(VirtualCqRequest&& request);
  inline folly::Expected<VirtualQpResponse, Error> submitRequestToVirtualQp(
      VirtualQpRequest&& request);

  // Register APIs for mapping management
  void registerVirtualQp(uint32_t virtualQpNum, IbvVirtualQp* virtualQp);
  void registerVirtualCq(uint32_t virtualCqNum, IbvVirtualCq* virtualCq);
  void registerPhysicalQpToVirtualQp(int physicalQpNum, uint32_t virtualQpNum);
  void registerVirtualQpToVirtualSendCq(
      uint32_t virtualQpNum,
      uint32_t virtualSendCqNum);
  void registerVirtualQpToVirtualRecvCq(
      uint32_t virtualQpNum,
      uint32_t virtualRecvCqNum);

  // Consolidated registration API for IbvVirtualQp - registers the virtual QP
  // along with all its physical QPs and CQ relationships in one call
  void registerVirtualQpWithVirtualCqMappings(
      IbvVirtualQp* virtualQp,
      uint32_t virtualSendCqNum,
      uint32_t virtualRecvCqNum);

  // Getter APIs for accessing mappings
  inline IbvVirtualCq* getVirtualSendCq(uint32_t virtualQpNum) const;
  inline IbvVirtualCq* getVirtualRecvCq(uint32_t virtualQpNum) const;
  inline IbvVirtualQp* getVirtualQpByPhysicalQpNum(int physicalQpNum) const;
  inline IbvVirtualQp* getVirtualQpById(uint32_t virtualQpNum) const;
  inline IbvVirtualCq* getVirtualCqById(uint32_t virtualCqNum) const;

  // Access APIs for testing and internal use
  const std::unordered_map<uint32_t, IbvVirtualQp*>& getVirtualQpMap() const;
  const std::unordered_map<uint32_t, IbvVirtualCq*>& getVirtualCqMap() const;
  const std::unordered_map<int, uint32_t>& getPhysicalQpToVirtualQpMap() const;
  const std::unordered_map<uint32_t, uint32_t>& getVirtualQpToVirtualSendCqMap()
      const;
  const std::unordered_map<uint32_t, uint32_t>& getVirtualQpToVirtualRecvCqMap()
      const;

  // Update API for move operations - only need to update pointer maps
  void updateVirtualQpPointer(uint32_t virtualQpNum, IbvVirtualQp* newPtr);
  void updateVirtualCqPointer(uint32_t virtualCqNum, IbvVirtualCq* newPtr);

  // Unregister API for cleanup during destruction
  void unregisterVirtualQp(uint32_t virtualQpNum, IbvVirtualQp* ptr);
  void unregisterVirtualCq(uint32_t virtualCqNum, IbvVirtualCq* ptr);

  static std::shared_ptr<Coordinator> getCoordinator();

 private:
  // Map 1: Virtual QP Num -> Virtual QP pointer
  std::unordered_map<uint32_t, IbvVirtualQp*> virtualQpNumToVirtualQp_;

  // Map 2: Virtual CQ Num -> Virtual CQ pointer
  std::unordered_map<uint32_t, IbvVirtualCq*> virtualCqNumToVirtualCq_;

  // Map 3: Virtual QP Num -> Virtual Send CQ Num (relationship)
  std::unordered_map<uint32_t, uint32_t> virtualQpNumToVirtualSendCqNum_;

  // Map 4: Virtual QP Num -> Virtual Recv CQ Num (relationship)
  std::unordered_map<uint32_t, uint32_t> virtualQpNumToVirtualRecvCqNum_;

  // Map 5: Physical QP number -> Virtual QP Num (for routing)
  std::unordered_map<int, uint32_t> physicalQpNumToVirtualQpNum_;
};

// Coordinator inline functions
inline IbvVirtualCq* Coordinator::getVirtualSendCq(
    uint32_t virtualQpNum) const {
  auto it = virtualQpNumToVirtualSendCqNum_.find(virtualQpNum);
  if (it == virtualQpNumToVirtualSendCqNum_.end()) {
    return nullptr;
  }
  return getVirtualCqById(it->second);
}

inline IbvVirtualCq* Coordinator::getVirtualRecvCq(
    uint32_t virtualQpNum) const {
  auto it = virtualQpNumToVirtualRecvCqNum_.find(virtualQpNum);
  if (it == virtualQpNumToVirtualRecvCqNum_.end()) {
    return nullptr;
  }
  return getVirtualCqById(it->second);
}

inline IbvVirtualQp* Coordinator::getVirtualQpByPhysicalQpNum(
    int physicalQpNum) const {
  auto it = physicalQpNumToVirtualQpNum_.find(physicalQpNum);
  if (it == physicalQpNumToVirtualQpNum_.end()) {
    return nullptr;
  }
  return getVirtualQpById(it->second);
}

inline IbvVirtualQp* Coordinator::getVirtualQpById(
    uint32_t virtualQpNum) const {
  auto it = virtualQpNumToVirtualQp_.find(virtualQpNum);
  if (it == virtualQpNumToVirtualQp_.end()) {
    return nullptr;
  }
  return it->second;
}

inline IbvVirtualCq* Coordinator::getVirtualCqById(
    uint32_t virtualCqNum) const {
  auto it = virtualCqNumToVirtualCq_.find(virtualCqNum);
  if (it == virtualCqNumToVirtualCq_.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace ibverbx
