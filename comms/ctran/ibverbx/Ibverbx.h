// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>
#include <vector>

#include "comms/ctran/ibverbx/Coordinator.h"
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvVirtualQp.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Forward declarations
class IbvVirtualQp;
class Coordinator;

/*** ibverbx APIs ***/

folly::Expected<folly::Unit, Error> ibvInit();

// Get a completion event from the completion channel
folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context);

// Acknowledge completion events
void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents);

// IbvMr: Memory Region
class IbvMr {
 public:
  ~IbvMr();

  // disable copy constructor
  IbvMr(const IbvMr&) = delete;
  IbvMr& operator=(const IbvMr&) = delete;

  // move constructor
  IbvMr(IbvMr&& other) noexcept;
  IbvMr& operator=(IbvMr&& other) noexcept;

  ibv_mr* mr() const;

 private:
  friend class IbvPd;

  explicit IbvMr(ibv_mr* mr);

  ibv_mr* mr_{nullptr};
};

// IbvPd: Protection Domain
class IbvPd {
 public:
  ~IbvPd();

  // disable copy constructor
  IbvPd(const IbvPd&) = delete;
  IbvPd& operator=(const IbvPd&) = delete;

  // move constructor
  IbvPd(IbvPd&& other) noexcept;
  IbvPd& operator=(IbvPd&& other) noexcept;

  ibv_pd* pd() const;
  bool useDataDirect() const;

  folly::Expected<IbvMr, Error>
  regMr(void* addr, size_t length, ibv_access_flags access) const;

  folly::Expected<IbvMr, Error> regDmabufMr(
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      ibv_access_flags access) const;

  folly::Expected<IbvQp, Error> createQp(ibv_qp_init_attr* initAttr) const;

  // The send_cq and recv_cq fields in initAttr are ignored.
  // Instead, initAttr.send_cq and initAttr.recv_cq will be set to the physical
  // CQs contained within sendCq and recvCq, respectively.
  folly::Expected<IbvVirtualQp, Error> createVirtualQp(
      int totalQps,
      ibv_qp_init_attr* initAttr,
      IbvVirtualCq* sendCq,
      IbvVirtualCq* recvCq,
      int maxMsgCntPerQp = kIbMaxMsgCntPerQp,
      int maxMsgSize = kIbMaxMsgSizeByte,
      LoadBalancingScheme loadBalancingScheme =
          LoadBalancingScheme::SPRAY) const;

 private:
  friend class IbvDevice;

  IbvPd(ibv_pd* pd, bool dataDirect = false);

  ibv_pd* pd_{nullptr};
  bool dataDirect_{false}; // Relevant only to mlx5
};

// IbvDevice
class IbvDevice {
 public:
  static folly::Expected<std::vector<IbvDevice>, Error> ibvGetDeviceList(
      const std::vector<std::string>& hcaList = kDefaultHcaList,
      const std::string& hcaPrefix = std::string(kDefaultHcaPrefix),
      int defaultPort = kIbAnyPort);
  IbvDevice(ibv_device* ibvDevice, int port);
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
      int defaultPort = kIbAnyPort);
};

class RoceHca {
 public:
  RoceHca(std::string hcaStr, int defaultPort);
  std::string name;
  int port{-1};
};

class Mlx5dv {
 public:
  static folly::Expected<folly::Unit, Error> initObj(
      mlx5dv_obj* obj,
      uint64_t obj_type);
};

//
// Inline function definitions
//

} // namespace ibverbx
