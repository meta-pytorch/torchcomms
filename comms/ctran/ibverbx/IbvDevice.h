// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbvPd.h"
#include "comms/ctran/ibverbx/IbvVirtualCq.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// IbvDevice
class IbvDevice {
 public:
  static Expected<std::vector<IbvDevice>> ibvGetDeviceList(
      const std::vector<std::string>& hcaList = kDefaultHcaList,
      const std::string& hcaPrefix = std::string(kDefaultHcaPrefix),
      int defaultPort = kIbAnyPort,
      int ibDataDirect = kDefaultIbDataDirect);
  IbvDevice(ibv_device* ibvDevice, int port, bool dataDirect = false);
  ~IbvDevice();

  // disable copy constructor
  IbvDevice(const IbvDevice&) = delete;
  IbvDevice& operator=(const IbvDevice&) = delete;

  // move constructor
  IbvDevice(IbvDevice&& other) noexcept;
  IbvDevice& operator=(IbvDevice&& other) noexcept;

  ibv_device* device() const;
  ibv_context* context() const;
  int port() const;
  int32_t getDeviceId() const;

  Expected<IbvPd> allocPd();
  Expected<IbvPd> allocParentDomain(ibv_parent_domain_init_attr* attr);
  Expected<ibv_device_attr> queryDevice() const;
  Expected<ibv_port_attr> queryPort(uint8_t portNum) const;
  Expected<ibv_gid> queryGid(uint8_t portNum, int gidIndex) const;

  Expected<IbvCq> createCq(
      int cqe,
      void* cq_context,
      ibv_comp_channel* channel,
      int comp_vector) const;

  // create Cq with attributes
  Expected<IbvCq> createCq(ibv_cq_init_attr_ex* attr) const;

  // Create a completion channel for event-driven completion handling
  Expected<ibv_comp_channel*> createCompChannel() const;

  // Destroy a completion channel
  Status destroyCompChannel(ibv_comp_channel* channel) const;

  // When creating an IbvVirtualCq for an IbvVirtualQp, ensure that cqe >=
  // (number of QPs * capacity per QP). If send queue and recv queue intend to
  // share the same cqe, then ensure cqe >= (2 * number of QPs * capacity per
  // QP). Failing to meet this condition may result in lost CQEs. TODO: Enforce
  // this requirement in the low-level API. If a higher-level API is introduced
  // in the future, ensure this guarantee is handled within Ibverbx when
  // creating a IbvVirtualCq for the user.
  Expected<IbvVirtualCq> createVirtualCq(
      int cqe,
      void* cq_context,
      ibv_comp_channel* channel,
      int comp_vector);

  Expected<bool> isPortActive(
      uint8_t portNum,
      std::unordered_set<int> linkLayers) const;
  Expected<uint8_t> findActivePort(
      std::unordered_set<int> const& linkLayers) const;

 private:
  ibv_device* device_{nullptr};
  ibv_context* context_{nullptr};
  int port_{-1};
  bool dataDirect_{false}; // Relevant only to mlx5

  inline static std::atomic<int32_t> nextDeviceId_{
      0}; // Static counter for assigning unique virtual Device IDs
  int32_t deviceId_{-1}; // The unique device ID assigned to
                         // instance of IbvDevice

  static std::vector<IbvDevice> ibvFilterDeviceList(
      int numDevs,
      ibv_device** devs,
      const std::vector<std::string>& hcaList = kDefaultHcaList,
      const std::string& hcaPrefix = std::string(kDefaultHcaPrefix),
      int defaultPort = kIbAnyPort,
      int ibDataDirect = kDefaultIbDataDirect);
};

} // namespace ibverbx
