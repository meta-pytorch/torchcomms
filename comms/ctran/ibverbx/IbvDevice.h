// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbvPd.h"
#include "comms/ctran/ibverbx/IbvVirtualCq.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// IbvDevice
class IbvDevice {
 public:
  static folly::Expected<std::vector<IbvDevice>, Error> ibvGetDeviceList(
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

  folly::Expected<IbvPd, Error> allocPd();
  folly::Expected<IbvPd, Error> allocParentDomain(
      ibv_parent_domain_init_attr* attr);
  folly::Expected<ibv_device_attr, Error> queryDevice() const;
  folly::Expected<ibv_port_attr, Error> queryPort(uint8_t portNum) const;
  folly::Expected<ibv_gid, Error> queryGid(uint8_t portNum, int gidIndex) const;

  folly::Expected<IbvCq, Error> createCq(
      int cqe,
      void* cq_context,
      ibv_comp_channel* channel,
      int comp_vector) const;

  // create Cq with attributes
  folly::Expected<IbvCq, Error> createCq(ibv_cq_init_attr_ex* attr) const;

  // Create a completion channel for event-driven completion handling
  folly::Expected<ibv_comp_channel*, Error> createCompChannel() const;

  // Destroy a completion channel
  folly::Expected<folly::Unit, Error> destroyCompChannel(
      ibv_comp_channel* channel) const;

  // When creating an IbvVirtualCq for an IbvVirtualQp, ensure that cqe >=
  // (number of QPs * capacity per QP). If send queue and recv queue intend to
  // share the same cqe, then ensure cqe >= (2 * number of QPs * capacity per
  // QP). Failing to meet this condition may result in lost CQEs. TODO: Enforce
  // this requirement in the low-level API. If a higher-level API is introduced
  // in the future, ensure this guarantee is handled within Ibverbx when
  // creating a IbvVirtualCq for the user.
  folly::Expected<IbvVirtualCq, Error> createVirtualCq(
      int cqe,
      void* cq_context,
      ibv_comp_channel* channel,
      int comp_vector);

  folly::Expected<bool, Error> isPortActive(
      uint8_t portNum,
      std::unordered_set<int> linkLayers) const;
  folly::Expected<uint8_t, Error> findActivePort(
      std::unordered_set<int> const& linkLayers) const;

 private:
  ibv_device* device_{nullptr};
  ibv_context* context_{nullptr};
  int port_{-1};
  bool dataDirect_{false}; // Relevant only to mlx5

  static std::vector<IbvDevice> ibvFilterDeviceList(
      int numDevs,
      ibv_device** devs,
      const std::vector<std::string>& hcaList = kDefaultHcaList,
      const std::string& hcaPrefix = std::string(kDefaultHcaPrefix),
      int defaultPort = kIbAnyPort,
      int ibDataDirect = kDefaultIbDataDirect);
};

} // namespace ibverbx
