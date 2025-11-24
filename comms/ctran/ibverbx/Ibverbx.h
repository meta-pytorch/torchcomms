// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>
#include <deque>
#include <vector>

#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Forward declarations
class IbvVirtualQp;
class Coordinator;

// Default HCA prefix
constexpr std::string_view kDefaultHcaPrefix = "";
// Default HCA list
const std::vector<std::string> kDefaultHcaList{};
// Default port
constexpr int kIbAnyPort = -1;
constexpr int kIbMaxMsgCntPerQp = 100;
constexpr int kIbMaxMsgSizeByte = 100;
constexpr int kIbMaxCqe_ = 100;
constexpr int kNotifyBit = 31;
constexpr uint32_t kSeqNumMask = 0xFFFFFF; // 24 bits

// Command types for coordinator routing and operations
enum class RequestType { SEND = 0, RECV = 1, SEND_NOTIFY = 2 };
enum class LoadBalancingScheme { SPRAY = 0, DQPLB = 1 };

struct IbvSymbols {
  int (*ibv_internal_fork_init)(void) = nullptr;
  struct ibv_device** (*ibv_internal_get_device_list)(int* num_devices) =
      nullptr;
  void (*ibv_internal_free_device_list)(struct ibv_device** list) = nullptr;
  const char* (*ibv_internal_get_device_name)(struct ibv_device* device) =
      nullptr;
  struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device) =
      nullptr;
  int (*ibv_internal_close_device)(struct ibv_context* context) = nullptr;
  int (*ibv_internal_get_async_event)(
      struct ibv_context* context,
      struct ibv_async_event* event) = nullptr;
  void (*ibv_internal_ack_async_event)(struct ibv_async_event* event) = nullptr;
  int (*ibv_internal_query_device)(
      struct ibv_context* context,
      struct ibv_device_attr* device_attr) = nullptr;
  int (*ibv_internal_query_port)(
      struct ibv_context* context,
      uint8_t port_num,
      struct ibv_port_attr* port_attr) = nullptr;
  int (*ibv_internal_query_gid)(
      struct ibv_context* context,
      uint8_t port_num,
      int index,
      union ibv_gid* gid) = nullptr;
  int (*ibv_internal_query_qp)(
      struct ibv_qp* qp,
      struct ibv_qp_attr* attr,
      int attr_mask,
      struct ibv_qp_init_attr* init_attr) = nullptr;
  struct ibv_pd* (*ibv_internal_alloc_pd)(struct ibv_context* context) =
      nullptr;
  struct ibv_pd* (*ibv_internal_alloc_parent_domain)(
      struct ibv_context* context,
      struct ibv_parent_domain_init_attr* attr) = nullptr;
  int (*ibv_internal_dealloc_pd)(struct ibv_pd* pd) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_mr)(
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      int access) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_mr_iova2)(
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      uint64_t iova,
      unsigned int access) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_dmabuf_mr)(
      struct ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access) = nullptr;
  int (*ibv_internal_dereg_mr)(struct ibv_mr* mr) = nullptr;
  struct ibv_cq* (*ibv_internal_create_cq)(
      struct ibv_context* context,
      int cqe,
      void* cq_context,
      struct ibv_comp_channel* channel,
      int comp_vector) = nullptr;
  struct ibv_cq_ex* (*ibv_internal_create_cq_ex)(
      struct ibv_context* context,
      struct ibv_cq_init_attr_ex* attr) = nullptr;
  int (*ibv_internal_destroy_cq)(struct ibv_cq* cq) = nullptr;
  struct ibv_comp_channel* (*ibv_internal_create_comp_channel)(
      struct ibv_context* context) = nullptr;
  int (*ibv_internal_destroy_comp_channel)(struct ibv_comp_channel* channel) =
      nullptr;
  int (*ibv_internal_get_cq_event)(
      struct ibv_comp_channel* channel,
      struct ibv_cq** cq,
      void** cq_context) = nullptr;
  void (*ibv_internal_ack_cq_events)(struct ibv_cq* cq, unsigned int nevents) =
      nullptr;
  struct ibv_qp* (*ibv_internal_create_qp)(
      struct ibv_pd* pd,
      struct ibv_qp_init_attr* qp_init_attr) = nullptr;
  int (*ibv_internal_modify_qp)(
      struct ibv_qp* qp,
      struct ibv_qp_attr* attr,
      int attr_mask) = nullptr;
  int (*ibv_internal_destroy_qp)(struct ibv_qp* qp) = nullptr;
  const char* (*ibv_internal_event_type_str)(enum ibv_event_type event) =
      nullptr;
  int (*ibv_internal_query_ece)(struct ibv_qp* qp, struct ibv_ece* ece) =
      nullptr;
  int (*ibv_internal_set_ece)(struct ibv_qp* qp, struct ibv_ece* ece) = nullptr;
  enum ibv_fork_status (*ibv_internal_is_fork_initialized)() = nullptr;

  /* mlx5dv functions */
  int (*mlx5dv_internal_init_obj)(struct mlx5dv_obj* obj, uint64_t obj_type) =
      nullptr;
  bool (*mlx5dv_internal_is_supported)(struct ibv_device* device) = nullptr;
  int (*mlx5dv_internal_get_data_direct_sysfs_path)(
      struct ibv_context* context,
      char* buf,
      size_t buf_len) = nullptr;
  /* DMA-BUF support */
  struct ibv_mr* (*mlx5dv_internal_reg_dmabuf_mr)(
      struct ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access,
      int mlx5_access) = nullptr;
};

int buildIbvSymbols(IbvSymbols& ibvSymbols);

struct Error {
  Error();
  explicit Error(int errNum);
  Error(int errNum, std::string errStr);

  const int errNum{0};
  const std::string errStr;
};

struct VirtualWc {
  VirtualWc() = default;
  ~VirtualWc() = default;

  struct ibv_wc wc{};
  int expectedMsgCnt{0};
  int remainingMsgCnt{0};
  bool sendExtraNotifyImm{
      false}; // Whether to expect an extra notify IMM
              // message to be sent for the current virtualWc
};

struct VirtualSendWr {
  inline VirtualSendWr(
      const ibv_send_wr& wr,
      int expectedMsgCnt,
      int remainingMsgCnt,
      bool sendExtraNotifyImm);
  VirtualSendWr() = default;
  ~VirtualSendWr() = default;

  ibv_send_wr wr{}; // Copy of the work request being posted by the user
  std::vector<ibv_sge> sgList; // Copy of the scatter-gather list
  int expectedMsgCnt{0}; // Expected message count resulting from splitting a
                         // large user message into multiple parts
  int remainingMsgCnt{0}; // Number of message segments left to transmit after
                          // splitting a large user messaget
  int offset{
      0}; // Address offset to be used for the next message send operation
  bool sendExtraNotifyImm{false}; // Whether to send an extra notify IMM message
                                  // for the current VirtualSendWr
};

struct VirtualRecvWr {
  inline VirtualRecvWr(
      const ibv_recv_wr& wr,
      int expectedMsgCnt,
      int remainingMsgCnt);
  VirtualRecvWr() = default;
  ~VirtualRecvWr() = default;

  ibv_recv_wr wr{}; // Copy of the work request being posted by the user
  std::vector<ibv_sge> sgList; // Copy of the scatter-gather list
  int expectedMsgCnt{0}; // Expected message count resulting from splitting a
                         // large user message into multiple parts
  int remainingMsgCnt{0}; // Number of message segments left to transmit after
                          // splitting a large user messaget
  int offset{
      0}; // Address offset to be used for the next message send operation
};

struct VirtualQpRequest {
  RequestType type{RequestType::SEND};
  uint64_t wrId{0};
  uint32_t physicalQpNum{0};
  uint32_t immData{0};
};

struct VirtualQpResponse {
  uint64_t virtualWrId{0};
  bool useDqplb{false};
  int notifyCount{0};
};

struct VirtualCqRequest {
  RequestType type{RequestType::SEND};
  int virtualQpNum{-1};
  int expectedMsgCnt{-1};
  ibv_send_wr* sendWr{nullptr};
  ibv_recv_wr* recvWr{nullptr};
  bool sendExtraNotifyImm{false};
};

class DqplbSeqTracker {
 public:
  DqplbSeqTracker() = default;
  ~DqplbSeqTracker() = default;

  // Explicitly default move constructor and move assignment operator
  DqplbSeqTracker(DqplbSeqTracker&&) = default;
  DqplbSeqTracker& operator=(DqplbSeqTracker&&) = default;

  // This helper function calculates sender IMM message in DQPLB mode.
  inline uint32_t getSendImm(int remainingMsgCnt);
  // This helper function processes received IMM message and update
  // receivedSeqNums_ map and receiveNext_ field.
  inline int processReceivedImm(uint32_t receivedImm);

 private:
  int sendNext_{0};
  int receiveNext_{0};
  std::unordered_map<uint32_t, bool> receivedSeqNums_;
};

std::ostream& operator<<(std::ostream&, Error const&);

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

// Ibv CompletionQueue(CQ)
class IbvCq {
 public:
  IbvCq() = default;
  ~IbvCq();

  // disable copy constructor
  IbvCq(const IbvCq&) = delete;
  IbvCq& operator=(const IbvCq&) = delete;

  // move constructor
  IbvCq(IbvCq&& other) noexcept;
  IbvCq& operator=(IbvCq&& other) noexcept;

  ibv_cq* cq() const;
  inline folly::Expected<std::vector<ibv_wc>, Error> pollCq(int numEntries);

  // Request notification when the next completion is added to this CQ
  folly::Expected<folly::Unit, Error> reqNotifyCq(int solicited_only) const;

 private:
  friend class IbvDevice;

  explicit IbvCq(ibv_cq* cq);

  ibv_cq* cq_{nullptr};
};

// Ibv Virtual Completion Queue (CQ): Provides a virtual CQ abstraction for the
// user. When the user calls IbvVirtualQp::postSend() or
// IbvVirtualQp::postRecv(), they can track the completion of messages posted on
// the Virtual QP through this virtual CQ.
class IbvVirtualCq {
 public:
  IbvVirtualCq(IbvCq&& cq, int maxCqe);
  ~IbvVirtualCq();

  // disable copy constructor
  IbvVirtualCq(const IbvVirtualCq&) = delete;
  IbvVirtualCq& operator=(const IbvVirtualCq&) = delete;

  // move constructor
  IbvVirtualCq(IbvVirtualCq&& other) noexcept;
  IbvVirtualCq& operator=(IbvVirtualCq&& other) noexcept;

  inline folly::Expected<std::vector<ibv_wc>, Error> pollCq(int numEntries);

  IbvCq& getPhysicalCqRef();
  uint32_t getVirtualCqNum() const;

  void enqueSendCq(VirtualWc virtualWc);
  void enqueRecvCq(VirtualWc virtualWc);

  inline void processRequest(VirtualCqRequest&& request);

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;

  inline static std::atomic<uint32_t> nextVirtualCqNum_{
      0}; // Static counter for assigning unique virtual CQ numbers
  uint32_t virtualCqNum_{
      0}; // The unique virtual CQ number assigned to instance of IbvVirtualCq

  IbvCq physicalCq_;
  int maxCqe_{0};
  std::deque<VirtualWc> pendingSendVirtualWcQue_;
  std::deque<VirtualWc> pendingRecvVirtualWcQue_;
  inline void updateVirtualWcFromPhysicalWc(
      const ibv_wc& physicalWc,
      VirtualWc* virtualWc);
  std::unordered_map<uint64_t, VirtualWc*> virtualWrIdToVirtualWc_;

  // Helper function for IbvVirtualCq::pollCq.
  // Continuously polls the underlying physical Completion Queue (CQ) in a loop,
  // retrieving all available Completion Queue Entries (CQEs) until none remain.
  // For each physical CQE polled, the corresponding virtual CQE entries in the
  // virtual CQ are also updated. This function ensures that all ready physical
  // CQEs are polled, processed, and reflected in the virtual CQ state.
  inline folly::Expected<folly::Unit, Error> loopPollPhysicalCqUntilEmpty();

  // Helper function for IbvVirtualCq::pollCq.
  // Continuously polls the underlying virtual Completion Queues (CQs) in a
  // loop. The function collects up to numEntries virtual Completion Queue
  // Entries (CQEs), or stops early if there are no more virtual CQEs available
  // to poll. Returns a vector containing the polled virtual CQEs.
  inline std::vector<ibv_wc> loopPollVirtualCqUntil(int numEntries);
};

// Ibv Queue Pair
class IbvQp {
 public:
  ~IbvQp();

  // disable copy constructor
  IbvQp(const IbvQp&) = delete;
  IbvQp& operator=(const IbvQp&) = delete;

  // move constructor
  IbvQp(IbvQp&& other) noexcept;
  IbvQp& operator=(IbvQp&& other) noexcept;

  ibv_qp* qp() const;

  folly::Expected<folly::Unit, Error> modifyQp(ibv_qp_attr* attr, int attrMask);
  folly::Expected<std::pair<ibv_qp_attr, ibv_qp_init_attr>, Error> queryQp(
      int attrMask) const;

  inline uint32_t getQpNum() const;
  inline folly::Expected<folly::Unit, Error> postRecv(
      ibv_recv_wr* recvWr,
      ibv_recv_wr* recvWrBad);
  inline folly::Expected<folly::Unit, Error> postSend(
      ibv_send_wr* sendWr,
      ibv_send_wr* sendWrBad);

  void enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId);
  void enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId);
  void dequePhysicalSendWrStatus();
  void dequePhysicalRecvWrStatus();
  bool isSendQueueAvailable(int maxMsgCntPerQp) const;
  bool isRecvQueueAvailable(int maxMsgCntPerQp) const;

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;
  friend class IbvVirtualCq;

  struct PhysicalSendWrStatus {
    PhysicalSendWrStatus(uint64_t physicalWrId, uint64_t virtualWrId)
        : physicalWrId(physicalWrId), virtualWrId(virtualWrId) {}
    uint64_t physicalWrId{0};
    uint64_t virtualWrId{0};
  };
  struct PhysicalRecvWrStatus {
    PhysicalRecvWrStatus(uint64_t physicalWrId, uint64_t virtualWrId)
        : physicalWrId(physicalWrId), virtualWrId(virtualWrId) {}
    uint64_t physicalWrId{0};
    uint64_t virtualWrId{0};
  };
  explicit IbvQp(ibv_qp* qp);

  ibv_qp* qp_{nullptr};
  std::deque<PhysicalSendWrStatus> physicalSendWrStatus_;
  std::deque<PhysicalRecvWrStatus> physicalRecvWrStatus_;
};

// IbvVirtualQpBusinessCard
struct IbvVirtualQpBusinessCard {
  explicit IbvVirtualQpBusinessCard(
      std::vector<uint32_t> qpNums,
      uint32_t notifyQpNum = 0);
  IbvVirtualQpBusinessCard() = default;
  ~IbvVirtualQpBusinessCard() = default;

  // Default copy constructor and assignment operator
  IbvVirtualQpBusinessCard(const IbvVirtualQpBusinessCard& other) = default;
  IbvVirtualQpBusinessCard& operator=(const IbvVirtualQpBusinessCard& other) =
      default;

  // Default move constructor and assignment operator
  IbvVirtualQpBusinessCard(IbvVirtualQpBusinessCard&& other) = default;
  IbvVirtualQpBusinessCard& operator=(IbvVirtualQpBusinessCard&& other) =
      default;

  // Convert to/from folly::dynamic for serialization
  folly::dynamic toDynamic() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> fromDynamic(
      const folly::dynamic& obj);

  // JSON serialization methods
  std::string serialize() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> deserialize(
      const std::string& jsonStr);

  // The qpNums_ vector is ordered: the ith QP in qpNums_ will be
  // connected to the ith QP in the remote side's qpNums_ vector.
  std::vector<uint32_t> qpNums_;
  uint32_t notifyQpNum_{0};
};

// Ibv Virtual Queue Pair
class IbvVirtualQp {
 public:
  ~IbvVirtualQp();

  // disable copy constructor
  IbvVirtualQp(const IbvVirtualQp&) = delete;
  IbvVirtualQp& operator=(const IbvVirtualQp&) = delete;

  // move constructor
  IbvVirtualQp(IbvVirtualQp&& other) noexcept;
  IbvVirtualQp& operator=(IbvVirtualQp&& other) noexcept;

  size_t getTotalQps() const;
  const std::vector<IbvQp>& getQpsRef() const;
  std::vector<IbvQp>& getQpsRef();
  const IbvQp& getNotifyQpRef() const;
  uint32_t getVirtualQpNum() const;
  // If businessCard is not provided, all physical QPs will be updated with the
  // universal attributes specified in attr. This is typically used for changing
  // the state to INIT or RTS.
  // If businessCard is provided, attr.qp_num for each physical QP will be set
  // individually to the corresponding qpNum stored in qpNums_ within
  // businessCard. This is typically used for changing the state to RTR.
  folly::Expected<folly::Unit, Error> modifyVirtualQp(
      ibv_qp_attr* attr,
      int attrMask,
      const IbvVirtualQpBusinessCard& businessCard =
          IbvVirtualQpBusinessCard());
  IbvVirtualQpBusinessCard getVirtualQpBusinessCard() const;
  LoadBalancingScheme getLoadBalancingScheme() const;

  inline folly::Expected<folly::Unit, Error> postSend(
      ibv_send_wr* sendWr,
      ibv_send_wr* sendWrBad);

  inline folly::Expected<folly::Unit, Error> postRecv(
      ibv_recv_wr* ibvRecvWr,
      ibv_recv_wr* badIbvRecvWr);

  inline int findAvailableSendQp();
  inline int findAvailableRecvQp();

  inline folly::Expected<VirtualQpResponse, Error> processRequest(
      VirtualQpRequest&& request);

 private:
#ifdef IBVERBX_TEST_FRIENDS
  IBVERBX_TEST_FRIENDS
#endif

  // updatePhysicalSendWrFromVirtualSendWr is a helper function to update
  // physical send work request (ibv_send_wr) from virtual send work request
  inline void updatePhysicalSendWrFromVirtualSendWr(
      VirtualSendWr& virtualSendWr,
      ibv_send_wr* sendWr,
      ibv_sge* sendSg);

  friend class IbvPd;
  friend class IbvVirtualCq;

  std::deque<VirtualSendWr> pendingSendVirtualWrQue_;
  std::deque<VirtualRecvWr> pendingRecvVirtualWrQue_;

  inline static std::atomic<uint32_t> nextVirtualQpNum_{
      0}; // Static counter for assigning unique virtual QP numbers
  uint32_t virtualQpNum_{0}; // The unique virtual QP number assigned to
                             // instance of IbvVirtualQp.

  std::vector<IbvQp> physicalQps_;
  std::unordered_map<int, int> qpNumToIdx_;

  int nextSendPhysicalQpIdx_{0};
  int nextRecvPhysicalQpIdx_{0};

  int maxMsgCntPerQp_{
      -1}; // Maximum number of messages that can be sent on each physical QP. A
           // value of -1 indicates there is no limit.
  int maxMsgSize_{0};

  uint64_t nextPhysicalWrId_{0}; // ID of the next physical work request to
                                 // be posted on the physical QP

  LoadBalancingScheme loadBalancingScheme_{
      LoadBalancingScheme::SPRAY}; // Load balancing scheme for this virtual QP

  // Spray mode specific fields
  std::deque<VirtualSendWr> pendingSendNotifyVirtualWrQue_;
  IbvQp notifyQp_;

  // DQPLB mode specific fields and functions
  DqplbSeqTracker dqplbSeqTracker;
  bool dqplbReceiverInitialized_{
      false}; // flag to indicate if dqplb receiver is initialized
  inline folly::Expected<folly::Unit, Error> initializeDqplbReceiver();

  IbvVirtualQp(
      std::vector<IbvQp>&& qps,
      IbvQp&& notifyQp,
      IbvVirtualCq* sendCq,
      IbvVirtualCq* recvCq,
      int maxMsgCntPerQp = kIbMaxMsgCntPerQp,
      int maxMsgSize = kIbMaxMsgSizeByte,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);

  // mapPendingSendQueToPhysicalQp is a helper function to iterate through
  // virtualSendWr in the pendingSendVirtualWrQue_, construct physical wrs and
  // call postSend on physical QP. If qpIdx is provided, this function will
  // postSend physicalWr on qpIdx. If qpIdx is not provided, then the function
  // will find an available Qp to postSend the physical work request on.
  inline folly::Expected<folly::Unit, Error> mapPendingSendQueToPhysicalQp(
      int qpIdx = -1);

  // postSendNotifyImm is a helper function to send IMM notification message
  // after all previous messages are sent in a large message
  inline folly::Expected<folly::Unit, Error> postSendNotifyImm();
  inline folly::Expected<folly::Unit, Error> mapPendingRecvQueToPhysicalQp(
      int qpIdx = -1);
  inline folly::Expected<folly::Unit, Error> postRecvNotifyImm(int qpIdx = -1);
};

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

// IbvQp inline functions
inline uint32_t IbvQp::getQpNum() const {
  XCHECK_NE(qp_, nullptr);
  return qp_->qp_num;
}

inline folly::Expected<folly::Unit, Error> IbvQp::postRecv(
    ibv_recv_wr* recvWr,
    ibv_recv_wr* recvWrBad) {
  int rc = qp_->context->ops.post_recv(qp_, recvWr, &recvWrBad);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvQp::postSend(
    ibv_send_wr* sendWr,
    ibv_send_wr* sendWrBad) {
  int rc = qp_->context->ops.post_send(qp_, sendWr, &sendWrBad);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

// IbvCq inline functions
inline folly::Expected<std::vector<ibv_wc>, Error> IbvCq::pollCq(
    int numEntries) {
  std::vector<ibv_wc> wcs(numEntries);
  int numPolled = cq_->context->ops.poll_cq(cq_, numEntries, wcs.data());
  if (numPolled < 0) {
    wcs.clear();
    return folly::makeUnexpected(
        Error(EINVAL, fmt::format("Call to pollCq() returned {}", numPolled)));
  } else {
    wcs.resize(numPolled);
  }
  return wcs;
}

// IbvVirtualCq inline functions
inline folly::Expected<std::vector<ibv_wc>, Error> IbvVirtualCq::pollCq(
    int numEntries) {
  auto maybeLoopPollPhysicalCq = loopPollPhysicalCqUntilEmpty();
  if (maybeLoopPollPhysicalCq.hasError()) {
    return folly::makeUnexpected(maybeLoopPollPhysicalCq.error());
  }

  return loopPollVirtualCqUntil(numEntries);
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualCq::loopPollPhysicalCqUntilEmpty() {
  // Poll from physical CQ one by one and process immediately
  while (true) {
    // Poll one completion at a time
    auto maybePhysicalWcsVector = physicalCq_.pollCq(1);
    if (maybePhysicalWcsVector.hasError()) {
      return folly::makeUnexpected(maybePhysicalWcsVector.error());
    }

    // If no completions available, break the loop
    if (maybePhysicalWcsVector->empty()) {
      break;
    }

    // Process the single completion immediately
    const auto& physicalWc = maybePhysicalWcsVector->front();

    if (physicalWc.opcode == IBV_WC_RECV ||
        physicalWc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      VirtualQpRequest request = {
          .type = RequestType::RECV,
          .wrId = physicalWc.wr_id,
          .physicalQpNum = physicalWc.qp_num};
      if (physicalWc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        request.immData = physicalWc.imm_data;
      }
      auto coordinator = Coordinator::getCoordinator();
      CHECK(coordinator) << "Coordinator should not be nullptr during pollCq!";
      auto response = coordinator->submitRequestToVirtualQp(std::move(request));
      if (response.hasError()) {
        return folly::makeUnexpected(response.error());
      }

      if (response->useDqplb) {
        int processedCount = 0;
        for (int i = 0; i < pendingRecvVirtualWcQue_.size() &&
             processedCount < response->notifyCount;
             i++) {
          if (pendingRecvVirtualWcQue_.at(i).remainingMsgCnt != 0) {
            pendingRecvVirtualWcQue_.at(i).remainingMsgCnt = 0;
            processedCount++;
          }
        }
      } else {
        auto virtualWc = virtualWrIdToVirtualWc_.at(response->virtualWrId);
        virtualWc->remainingMsgCnt--;
        updateVirtualWcFromPhysicalWc(physicalWc, virtualWc);
      }
    } else {
      // Except for the above two conditions, all other conditions indicate a
      // send message, and we should poll from send queue
      VirtualQpRequest request = {
          .type = RequestType::SEND,
          .wrId = physicalWc.wr_id,
          .physicalQpNum = physicalWc.qp_num};
      auto coordinator = Coordinator::getCoordinator();
      CHECK(coordinator) << "Coordinator should not be nullptr during pollCq!";
      auto response = coordinator->submitRequestToVirtualQp(std::move(request));
      if (response.hasError()) {
        return folly::makeUnexpected(response.error());
      }

      auto virtualWc = virtualWrIdToVirtualWc_.at(response->virtualWrId);
      virtualWc->remainingMsgCnt--;
      updateVirtualWcFromPhysicalWc(physicalWc, virtualWc);
      if (virtualWc->remainingMsgCnt == 1 && virtualWc->sendExtraNotifyImm) {
        VirtualQpRequest request = {
            .type = RequestType::SEND_NOTIFY,
            .wrId = response->virtualWrId,
            .physicalQpNum = physicalWc.qp_num};

        auto coordinator = Coordinator::getCoordinator();
        CHECK(coordinator)
            << "Coordinator should not be nullptr during pollCq!";
        auto response =
            coordinator->submitRequestToVirtualQp(std::move(request));
        if (response.hasError()) {
          return folly::makeUnexpected(response.error());
        }
      }
    }
  }

  return folly::unit;
}

inline std::vector<ibv_wc> IbvVirtualCq::loopPollVirtualCqUntil(
    int numEntries) {
  std::vector<ibv_wc> wcs;
  wcs.reserve(numEntries);
  bool virtualSendCqPollComplete = false;
  bool virtualRecvCqPollComplete = false;
  while (wcs.size() < static_cast<size_t>(numEntries) &&
         (!virtualSendCqPollComplete || !virtualRecvCqPollComplete)) {
    if (!virtualSendCqPollComplete) {
      if (pendingSendVirtualWcQue_.empty() ||
          pendingSendVirtualWcQue_.front().remainingMsgCnt > 0) {
        virtualSendCqPollComplete = true;
      } else {
        auto vSendCqHead = pendingSendVirtualWcQue_.front();
        virtualWrIdToVirtualWc_.erase(vSendCqHead.wc.wr_id);
        wcs.push_back(std::move(vSendCqHead.wc));
        pendingSendVirtualWcQue_.pop_front();
      }
    }

    if (!virtualRecvCqPollComplete) {
      if (pendingRecvVirtualWcQue_.empty() ||
          pendingRecvVirtualWcQue_.front().remainingMsgCnt > 0) {
        virtualRecvCqPollComplete = true;
      } else {
        auto vRecvCqHead = pendingRecvVirtualWcQue_.front();
        virtualWrIdToVirtualWc_.erase(vRecvCqHead.wc.wr_id);
        wcs.push_back(std::move(vRecvCqHead.wc));
        pendingRecvVirtualWcQue_.pop_front();
      }
    }
  }

  return wcs;
}

inline void IbvVirtualCq::updateVirtualWcFromPhysicalWc(
    const ibv_wc& physicalWc,
    VirtualWc* virtualWc) {
  // Updates the vWc status field based on the statuses of all pWc instances.
  // If all physicalWc statuses indicate success, returns success.
  // If any of the physicalWc statuses indicate an error, return the first
  // encountered error code.
  // Additionally, log all error statuses for debug purposes.
  if (physicalWc.status != IBV_WC_SUCCESS) {
    if (virtualWc->wc.status == IBV_WC_SUCCESS) {
      virtualWc->wc.status = physicalWc.status;
    }

    // Log the error
    XLOGF(
        ERR,
        "Physical WC error: status={}, vendor_err={}, qp_num={}, wr_id={}",
        physicalWc.status,
        physicalWc.vendor_err,
        physicalWc.qp_num,
        physicalWc.wr_id);
  }

  // Update the OP code in virtualWc. Note that for the same user message, the
  // opcode must remain consistent, because all sub-messages within that user
  // message will be postSend using the same opcode.
  virtualWc->wc.opcode = physicalWc.opcode;

  // Update the vendor error in virtualWc. For now, assume that all pWc
  // instances will report the same vendor_error across all sub-messages
  // within a single user message.
  virtualWc->wc.vendor_err = physicalWc.vendor_err;

  virtualWc->wc.src_qp = physicalWc.src_qp;
  virtualWc->wc.byte_len += physicalWc.byte_len;
  virtualWc->wc.imm_data = physicalWc.imm_data;
  virtualWc->wc.wc_flags = physicalWc.wc_flags;
  virtualWc->wc.pkey_index = physicalWc.pkey_index;
  virtualWc->wc.slid = physicalWc.slid;
  virtualWc->wc.sl = physicalWc.sl;
  virtualWc->wc.dlid_path_bits = physicalWc.dlid_path_bits;
}

inline void IbvVirtualCq::processRequest(VirtualCqRequest&& request) {
  VirtualWc* virtualWcPtr = nullptr;
  uint64_t wrId;
  if (request.type == RequestType::SEND) {
    wrId = request.sendWr->wr_id;
    if (request.sendWr->send_flags & IBV_SEND_SIGNALED ||
        request.sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
      VirtualWc virtualWc{};
      virtualWc.wc.wr_id = request.sendWr->wr_id;
      virtualWc.wc.qp_num = request.virtualQpNum;
      virtualWc.wc.status = IBV_WC_SUCCESS;
      virtualWc.wc.byte_len = 0;
      virtualWc.expectedMsgCnt = request.expectedMsgCnt;
      virtualWc.remainingMsgCnt = request.expectedMsgCnt;
      virtualWc.sendExtraNotifyImm = request.sendExtraNotifyImm;
      pendingSendVirtualWcQue_.push_back(std::move(virtualWc));
      virtualWcPtr = &pendingSendVirtualWcQue_.back();
    }
  } else {
    wrId = request.recvWr->wr_id;
    VirtualWc virtualWc{};
    virtualWc.wc.wr_id = request.recvWr->wr_id;
    virtualWc.wc.qp_num = request.virtualQpNum;
    virtualWc.wc.status = IBV_WC_SUCCESS;
    virtualWc.wc.byte_len = 0;
    virtualWc.expectedMsgCnt = request.expectedMsgCnt;
    virtualWc.remainingMsgCnt = request.expectedMsgCnt;
    pendingRecvVirtualWcQue_.push_back(std::move(virtualWc));
    virtualWcPtr = &pendingRecvVirtualWcQue_.back();
  }
  virtualWrIdToVirtualWc_[wrId] = virtualWcPtr;
}

// IbvVirtualQp inline functions
inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::mapPendingSendQueToPhysicalQp(int qpIdx) {
  while (!pendingSendVirtualWrQue_.empty()) {
    // Get the front of vSendQ_ and obtain the send information
    VirtualSendWr& virtualSendWr = pendingSendVirtualWrQue_.front();

    // For Send opcodes related to RDMA_WRITE operations, use user selected load
    // balancing scheme specified in loadBalancingScheme_. For all other
    // opcodes, default to using physical QP 0.
    auto availableQpIdx = -1;
    if (virtualSendWr.wr.opcode == IBV_WR_RDMA_WRITE ||
        virtualSendWr.wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM ||
        virtualSendWr.wr.opcode == IBV_WR_RDMA_READ) {
      // Find an available Qp to send
      availableQpIdx = qpIdx == -1 ? findAvailableSendQp() : qpIdx;
      qpIdx = -1; // If qpIdx is provided, it indicates that one slot has been
                  // freed for the corresponding qpIdx. After using this slot,
                  // reset qpIdx to -1.
    } else if (
        physicalQps_.at(0).physicalSendWrStatus_.size() < maxMsgCntPerQp_) {
      availableQpIdx = 0;
    }
    if (availableQpIdx == -1) {
      break;
    }

    // Update the physical send work request with virtual one
    ibv_send_wr sendWr_{};
    ibv_sge sendSg_{};
    updatePhysicalSendWrFromVirtualSendWr(virtualSendWr, &sendWr_, &sendSg_);

    // Call ibv_post_send to send the message
    ibv_send_wr badSendWr_{};
    auto maybeSend =
        physicalQps_.at(availableQpIdx).postSend(&sendWr_, &badSendWr_);
    if (maybeSend.hasError()) {
      return folly::makeUnexpected(maybeSend.error());
    }

    // Enqueue the send information to physicalQps_
    physicalQps_.at(availableQpIdx)
        .physicalSendWrStatus_.emplace_back(
            sendWr_.wr_id, virtualSendWr.wr.wr_id);

    // Decide if need to deque the front of vSendQ_
    virtualSendWr.offset += sendWr_.sg_list->length;
    virtualSendWr.remainingMsgCnt--;
    if (virtualSendWr.remainingMsgCnt == 0) {
      pendingSendVirtualWrQue_.pop_front();
    } else if (
        virtualSendWr.remainingMsgCnt == 1 &&
        virtualSendWr.sendExtraNotifyImm) {
      // Move front entry from pendingSendVirtualWrQue_ to
      // pendingSendNotifyVirtualWrQue_
      pendingSendNotifyVirtualWrQue_.push_back(
          std::move(pendingSendVirtualWrQue_.front()));
      pendingSendVirtualWrQue_.pop_front();
    }
  }
  return folly::unit;
}

inline int IbvVirtualQp::findAvailableSendQp() {
  // maxMsgCntPerQp_ with a value of -1 indicates there is no limit.
  if (maxMsgCntPerQp_ == -1) {
    auto availableQpIdx = nextSendPhysicalQpIdx_;
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
    return availableQpIdx;
  }

  for (int i = 0; i < physicalQps_.size(); i++) {
    if (physicalQps_.at(nextSendPhysicalQpIdx_).physicalSendWrStatus_.size() <
        maxMsgCntPerQp_) {
      auto availableQpIdx = nextSendPhysicalQpIdx_;
      nextSendPhysicalQpIdx_ =
          (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
      return availableQpIdx;
    }
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
  }
  return -1;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSendNotifyImm() {
  auto virtualSendWr = pendingSendNotifyVirtualWrQue_.front();
  ibv_send_wr sendWr_{};
  ibv_send_wr badSendWr_{};
  ibv_sge sendSg_{};
  sendWr_.next = nullptr;
  sendWr_.sg_list = &sendSg_;
  sendWr_.num_sge = 0;
  sendWr_.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  sendWr_.send_flags = IBV_SEND_SIGNALED;
  sendWr_.wr.rdma.remote_addr = virtualSendWr.wr.wr.rdma.remote_addr;
  sendWr_.wr.rdma.rkey = virtualSendWr.wr.wr.rdma.rkey;
  sendWr_.imm_data = virtualSendWr.wr.imm_data;
  sendWr_.wr_id = nextPhysicalWrId_++;
  auto maybeSend = notifyQp_.postSend(&sendWr_, &badSendWr_);
  if (maybeSend.hasError()) {
    return folly::makeUnexpected(maybeSend.error());
  }
  notifyQp_.physicalSendWrStatus_.emplace_back(
      sendWr_.wr_id, virtualSendWr.wr.wr_id);
  virtualSendWr.remainingMsgCnt = 0;
  pendingSendNotifyVirtualWrQue_.pop_front();
  return folly::unit;
}

inline void IbvVirtualQp::updatePhysicalSendWrFromVirtualSendWr(
    VirtualSendWr& virtualSendWr,
    ibv_send_wr* sendWr,
    ibv_sge* sendSg) {
  sendWr->wr_id = nextPhysicalWrId_++;

  auto lenToSend = std::min(
      int(virtualSendWr.wr.sg_list->length - virtualSendWr.offset),
      maxMsgSize_);
  sendSg->addr = virtualSendWr.wr.sg_list->addr + virtualSendWr.offset;
  sendSg->length = lenToSend;
  sendSg->lkey = virtualSendWr.wr.sg_list->lkey;
  sendWr->next = nullptr;
  sendWr->sg_list = sendSg;
  sendWr->num_sge = 1;

  // Set the opcode to the same as virtual wr, except for RDMA_WRITE_WITH_IMM,
  // we'll handle the notification message separately
  switch (virtualSendWr.wr.opcode) {
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_READ:
      sendWr->opcode = virtualSendWr.wr.opcode;
      sendWr->send_flags = virtualSendWr.wr.send_flags;
      sendWr->wr.rdma.remote_addr =
          virtualSendWr.wr.wr.rdma.remote_addr + virtualSendWr.offset;
      sendWr->wr.rdma.rkey = virtualSendWr.wr.wr.rdma.rkey;
      break;
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      sendWr->opcode = (loadBalancingScheme_ == LoadBalancingScheme::SPRAY)
          ? IBV_WR_RDMA_WRITE
          : IBV_WR_RDMA_WRITE_WITH_IMM;
      sendWr->send_flags = IBV_SEND_SIGNALED;
      sendWr->wr.rdma.remote_addr =
          virtualSendWr.wr.wr.rdma.remote_addr + virtualSendWr.offset;
      sendWr->wr.rdma.rkey = virtualSendWr.wr.wr.rdma.rkey;
      break;
    case IBV_WR_SEND:
      sendWr->opcode = virtualSendWr.wr.opcode;
      sendWr->send_flags = virtualSendWr.wr.send_flags;
      break;

    default:
      break;
  }

  if (sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
      loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
    sendWr->imm_data =
        dqplbSeqTracker.getSendImm(virtualSendWr.remainingMsgCnt);
  }
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSend(
    ibv_send_wr* sendWr,
    ibv_send_wr* sendWrBad) {
  // Report error if num_sge is more than 1
  if (sendWr->num_sge > 1) {
    return folly::makeUnexpected(Error(
        EINVAL, "In IbvVirtualQp::postSend, num_sge > 1 is not supported"));
  }

  // Report error if opcode is not supported by Ibverbx virtualQp
  switch (sendWr->opcode) {
    case IBV_WR_SEND_WITH_IMM:
    case IBV_WR_ATOMIC_CMP_AND_SWP:
    case IBV_WR_ATOMIC_FETCH_AND_ADD:
      return folly::makeUnexpected(Error(
          EINVAL,
          "In IbvVirtualQp::postSend, opcode IBV_WR_SEND_WITH_IMM, IBV_WR_ATOMIC_CMP_AND_SWP, IBV_WR_ATOMIC_FETCH_AND_ADD are not supported"));

    default:
      break;
  }

  // Calculate the chunk number for the current message and update sendWqe
  bool sendExtraNotifyImm =
      (sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
       loadBalancingScheme_ == LoadBalancingScheme::SPRAY);
  int expectedMsgCnt =
      (sendWr->sg_list->length + maxMsgSize_ - 1) / maxMsgSize_;
  if (sendExtraNotifyImm) {
    expectedMsgCnt += 1; // After post send all data messages, will post send
                         // 1 more notification message on QP 0
  }

  // Submit request to virtualCq to enqueue VirtualWc
  VirtualCqRequest request = {
      .type = RequestType::SEND,
      .virtualQpNum = (int)virtualQpNum_,
      .expectedMsgCnt = expectedMsgCnt,
      .sendWr = sendWr,
      .sendExtraNotifyImm = sendExtraNotifyImm};
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator) << "Coordinator should not be nullptr during postSend!";
  coordinator->submitRequestToVirtualCq(std::move(request));

  // Set up the send work request with the completion queue entry and enqueue
  // Note: virtualWcPtr can be nullptr - this is intentional and supported
  // The VirtualSendWr constructor will handle deep copying of sendWr and
  // sg_list
  pendingSendVirtualWrQue_.emplace_back(
      *sendWr, expectedMsgCnt, expectedMsgCnt, sendExtraNotifyImm);

  // Map large messages from vSendQ_ to pQps_
  if (mapPendingSendQueToPhysicalQp().hasError()) {
    *sendWrBad = *sendWr;
    return folly::makeUnexpected(Error(errno));
  }

  return folly::unit;
}

inline folly::Expected<VirtualQpResponse, Error> IbvVirtualQp::processRequest(
    VirtualQpRequest&& request) {
  VirtualQpResponse response;
  // If request.physicalQpNum differs from notifyQpNum, locate the corresponding
  // physical qpIdx to process this request.
  auto qpIdx = request.physicalQpNum == notifyQp_.getQpNum()
      ? -1
      : qpNumToIdx_.at(request.physicalQpNum);
  // If qpIdx is -1, physicalQp is notifyQp; otherwise, physicalQp is the qpIdx
  // entry of physicalQps_
  auto& physicalQp = qpIdx == -1 ? notifyQp_ : physicalQps_.at(qpIdx);

  if (request.type == RequestType::RECV) {
    if (physicalQp.physicalRecvWrStatus_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalRecvWrStatus_ at physicalQp {} is empty!",
              request.physicalQpNum)));
    }

    auto& physicalRecvWrStatus = physicalQp.physicalRecvWrStatus_.front();

    if (physicalRecvWrStatus.physicalWrId != request.wrId) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalRecvWrStatus.physicalWrId({}) != request.wrId({})",
              physicalRecvWrStatus.physicalWrId,
              request.wrId)));
    }

    response.virtualWrId = physicalRecvWrStatus.virtualWrId;
    physicalQp.physicalRecvWrStatus_.pop_front();
    if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
      if (postRecvNotifyImm(qpIdx).hasError()) {
        return folly::makeUnexpected(
            Error(errno, fmt::format("postRecvNotifyImm() failed!")));
      }
      response.notifyCount =
          dqplbSeqTracker.processReceivedImm(request.immData);
      response.useDqplb = true;
    } else if (qpIdx != -1) {
      if (mapPendingRecvQueToPhysicalQp(qpIdx).hasError()) {
        return folly::makeUnexpected(Error(
            errno,
            fmt::format("mapPendingRecvQueToPhysicalQp({}) failed!", qpIdx)));
      }
    }
  } else if (request.type == RequestType::SEND) {
    if (physicalQp.physicalSendWrStatus_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalSendWrStatus_ at physicalQp {} is empty!",
              request.physicalQpNum)));
    }

    auto physicalSendWrStatus = physicalQp.physicalSendWrStatus_.front();

    if (physicalSendWrStatus.physicalWrId != request.wrId) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalSendWrStatus.physicalWrId({}) != request.wrId({})",
              physicalSendWrStatus.physicalWrId,
              request.wrId)));
    }

    response.virtualWrId = physicalSendWrStatus.virtualWrId;
    physicalQp.physicalSendWrStatus_.pop_front();
    if (qpIdx != -1) {
      if (mapPendingSendQueToPhysicalQp(qpIdx).hasError()) {
        return folly::makeUnexpected(Error(
            errno,
            fmt::format("mapPendingSendQueToPhysicalQp({}) failed!", qpIdx)));
      }
    }
  } else if (request.type == RequestType::SEND_NOTIFY) {
    if (pendingSendNotifyVirtualWrQue_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Tried to post send notify IMM message for wrId {} when pendingSendNotifyVirtualWrQue_ is empty",
              request.wrId)));
    }

    if (pendingSendNotifyVirtualWrQue_.front().wr.wr_id == request.wrId) {
      if (postSendNotifyImm().hasError()) {
        return folly::makeUnexpected(
            Error(errno, fmt::format("postSendNotifyImm() failed!")));
      }
    }
  }
  return response;
}

// Currently, this function is only invoked to receive messages with opcode
// IBV_WR_SEND. Therefore, we restrict its usage to physical QP 0.
// Note: If Dynamic QP Load Balancing (DQPLB) or other load balancing techniques
// are required in the future, this function can be updated to support more
// advanced usage.
inline int IbvVirtualQp::findAvailableRecvQp() {
  // maxMsgCntPerQp_ with a value of -1 indicates there is no limit.
  auto availableQpIdx = -1;
  if (maxMsgCntPerQp_ == -1 ||
      physicalQps_.at(0).physicalRecvWrStatus_.size() < maxMsgCntPerQp_) {
    availableQpIdx = 0;
  }

  return availableQpIdx;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvNotifyImm(
    int qpIdx) {
  auto& qp = qpIdx == -1 ? notifyQp_ : physicalQps_.at(qpIdx);
  auto virtualRecvWrId = loadBalancingScheme_ == LoadBalancingScheme::SPRAY
      ? pendingRecvVirtualWrQue_.front().wr.wr_id
      : -1;
  ibv_recv_wr recvWr_{};
  ibv_recv_wr badRecvWr_{};
  ibv_sge recvSg_{};
  recvWr_.next = nullptr;
  recvWr_.sg_list = &recvSg_;
  recvWr_.num_sge = 0;
  recvWr_.wr_id = nextPhysicalWrId_++;
  auto maybeRecv = qp.postRecv(&recvWr_, &badRecvWr_);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }
  qp.physicalRecvWrStatus_.emplace_back(recvWr_.wr_id, virtualRecvWrId);

  if (loadBalancingScheme_ == LoadBalancingScheme::SPRAY) {
    pendingRecvVirtualWrQue_.pop_front();
  }
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::initializeDqplbReceiver() {
  ibv_recv_wr recvWr_{};
  ibv_recv_wr badRecvWr_{};
  ibv_sge recvSg_{};
  recvWr_.next = nullptr;
  recvWr_.sg_list = &recvSg_;
  recvWr_.num_sge = 0;
  for (int i = 0; i < maxMsgCntPerQp_; i++) {
    for (int j = 0; j < physicalQps_.size(); j++) {
      recvWr_.wr_id = nextPhysicalWrId_++;
      auto maybeRecv = physicalQps_.at(j).postRecv(&recvWr_, &badRecvWr_);
      if (maybeRecv.hasError()) {
        return folly::makeUnexpected(maybeRecv.error());
      }
      physicalQps_.at(j).physicalRecvWrStatus_.emplace_back(recvWr_.wr_id, -1);
    }
  }

  dqplbReceiverInitialized_ = true;
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::mapPendingRecvQueToPhysicalQp(int qpIdx) {
  while (!pendingRecvVirtualWrQue_.empty()) {
    VirtualRecvWr& virtualRecvWr = pendingRecvVirtualWrQue_.front();

    if (virtualRecvWr.wr.num_sge == 0) {
      auto maybeRecvNotifyImm = postRecvNotifyImm();
      if (maybeRecvNotifyImm.hasError()) {
        return folly::makeUnexpected(maybeRecvNotifyImm.error());
      }
      continue;
    }

    // If num_sge is > 0, then the receive work request is used to receive
    // messages with opcode IBV_WR_SEND. In this scenario, we restrict usage to
    // physical QP 0 only. The reason behind is that, IBV_WR_SEND requires a
    // strict one-to-one correspondence between send and receive WRs. If Dynamic
    // QP Load Balancing (DQPLB) is applied, send and receive WRs may be posted
    // to different physical QPs within the QP list. This mismatch can result in
    // data being delivered to the wrong address, causing data integrity issues.
    auto availableQpIdx = qpIdx != 0 ? findAvailableRecvQp() : qpIdx;
    qpIdx = -1; // If qpIdx is provided, it indicates that one slot has been
                // freed for the corresponding qpIdx. After using this slot,
                // reset qpIdx to -1.
    if (availableQpIdx == -1) {
      break;
    }

    // Get the front of vRecvQ_ and obtain the receive information
    ibv_recv_wr recvWr_{};
    ibv_recv_wr badRecvWr_{};
    ibv_sge recvSg_{};
    int lenToRecv = 0;
    if (virtualRecvWr.wr.num_sge == 1) {
      lenToRecv = std::min(
          int(virtualRecvWr.wr.sg_list->length - virtualRecvWr.offset),
          maxMsgSize_);
      recvSg_.addr = virtualRecvWr.wr.sg_list->addr + virtualRecvWr.offset;
      recvSg_.length = lenToRecv;
      recvSg_.lkey = virtualRecvWr.wr.sg_list->lkey;

      recvWr_.sg_list = &recvSg_;
      recvWr_.num_sge = 1;
    } else {
      recvWr_.sg_list = nullptr;
      recvWr_.num_sge = 0;
    }
    recvWr_.wr_id = nextPhysicalWrId_++;
    recvWr_.next = nullptr;

    // Call ibv_post_recv to receive the message
    auto maybeRecv =
        physicalQps_.at(availableQpIdx).postRecv(&recvWr_, &badRecvWr_);
    if (maybeRecv.hasError()) {
      return folly::makeUnexpected(maybeRecv.error());
    }

    // Enqueue the receive information to physicalQps_
    physicalQps_.at(availableQpIdx)
        .physicalRecvWrStatus_.emplace_back(
            recvWr_.wr_id, virtualRecvWr.wr.wr_id);

    // Decide if need to deque the front of vRecvQ_
    if (virtualRecvWr.wr.num_sge == 1) {
      virtualRecvWr.offset += lenToRecv;
    }
    virtualRecvWr.remainingMsgCnt--;
    if (virtualRecvWr.remainingMsgCnt == 0) {
      pendingRecvVirtualWrQue_.pop_front();
    }
  }
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecv(
    ibv_recv_wr* recvWr,
    ibv_recv_wr* recvWrBad) {
  // Report error if num_sge is more than 1
  if (recvWr->num_sge > 1) {
    return folly::makeUnexpected(Error(EINVAL));
  }

  int expectedMsgCnt = 1;

  if (recvWr->num_sge == 0) { // recvWr->num_sge == 0 mean it's receiving a
                              // IMM notification message
    expectedMsgCnt = 1;
  } else if (recvWr->num_sge == 1) { // Calculate the chunk number for the
                                     // current message and update recvWqe if
                                     // num_sge is 1
    expectedMsgCnt = (recvWr->sg_list->length + maxMsgSize_ - 1) / maxMsgSize_;
  }

  // Submit request to virtualCq to enqueue VirtualWc
  VirtualCqRequest request = {
      .type = RequestType::RECV,
      .virtualQpNum = (int)virtualQpNum_,
      .expectedMsgCnt = expectedMsgCnt,
      .recvWr = recvWr};
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator) << "Coordinator should not be nullptr during postRecv!";
  coordinator->submitRequestToVirtualCq(std::move(request));

  // Set up the recv work request with the completion queue entry and enqueue
  pendingRecvVirtualWrQue_.emplace_back(
      *recvWr, expectedMsgCnt, expectedMsgCnt);

  if (loadBalancingScheme_ != LoadBalancingScheme::DQPLB) {
    if (mapPendingRecvQueToPhysicalQp().hasError()) {
      // For non-DQPLB modes: map messages from pendingRecvVirtualWrQue_ to
      // physicalQps_. In DQPLB mode, this mapping is unnecessary because all
      // receive notify IMM operations are pre-posted to the QPs before postRecv
      // is called.
      *recvWrBad = *recvWr;
      return folly::makeUnexpected(Error(errno));
    }
  } else if (dqplbReceiverInitialized_ == false) {
    if (initializeDqplbReceiver().hasError()) {
      *recvWrBad = *recvWr;
      return folly::makeUnexpected(Error(errno));
    }
  }

  return folly::unit;
}

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

inline folly::Expected<VirtualQpResponse, Error>
Coordinator::submitRequestToVirtualQp(VirtualQpRequest&& request) {
  auto virtualQp = getVirtualQpByPhysicalQpNum(request.physicalQpNum);
  return virtualQp->processRequest(std::move(request));
}

inline void Coordinator::submitRequestToVirtualCq(VirtualCqRequest&& request) {
  if (request.type == RequestType::SEND) {
    auto virtualCq = getVirtualSendCq(request.virtualQpNum);
    virtualCq->processRequest(std::move(request));
  } else {
    auto virtualCq = getVirtualRecvCq(request.virtualQpNum);
    virtualCq->processRequest(std::move(request));
  }
}

// VirtualSendWr inline constructor
inline VirtualSendWr::VirtualSendWr(
    const ibv_send_wr& wr,
    int expectedMsgCnt,
    int remainingMsgCnt,
    bool sendExtraNotifyImm)
    : expectedMsgCnt(expectedMsgCnt),
      remainingMsgCnt(remainingMsgCnt),
      sendExtraNotifyImm(sendExtraNotifyImm) {
  // Make an explicit copy of the ibv_send_wr structure
  this->wr = wr;

  // Deep copy the scatter-gather list
  if (wr.sg_list != nullptr && wr.num_sge > 0) {
    sgList.resize(wr.num_sge);
    std::copy(wr.sg_list, wr.sg_list + wr.num_sge, sgList.begin());
    // Update the copied work request to point to our own scatter-gather list
    this->wr.sg_list = sgList.data();
  } else {
    // Handle case where there's no scatter-gather list
    this->wr.sg_list = nullptr;
    this->wr.num_sge = 0;
  }
}

// VirtualRecvWr inline constructor
inline VirtualRecvWr::VirtualRecvWr(
    const ibv_recv_wr& wr,
    int expectedMsgCnt,
    int remainingMsgCnt)
    : expectedMsgCnt(expectedMsgCnt), remainingMsgCnt(remainingMsgCnt) {
  // Make an explicit copy of the ibv_recv_wr structure
  this->wr = wr;

  // Deep copy the scatter-gather list
  if (wr.sg_list != nullptr && wr.num_sge > 0) {
    sgList.resize(wr.num_sge);
    std::copy(wr.sg_list, wr.sg_list + wr.num_sge, sgList.begin());
    // Update the copied work request to point to our own scatter-gather list
    this->wr.sg_list = sgList.data();
  } else {
    // Handle case where there's no scatter-gather list
    this->wr.sg_list = nullptr;
    this->wr.num_sge = 0;
  }
}

// DqplbSeqTracker inline functions
inline uint32_t DqplbSeqTracker::getSendImm(int remainingMsgCnt) {
  uint32_t immData = sendNext_;
  sendNext_ = (sendNext_ + 1) % kSeqNumMask;
  if (remainingMsgCnt == 1) {
    immData |= (1 << kNotifyBit);
  }
  return immData;
}

inline int DqplbSeqTracker::processReceivedImm(uint32_t immData) {
  int notifyCount = 0;
  receivedSeqNums_[immData & kSeqNumMask] = immData & (1U << kNotifyBit);
  auto it = receivedSeqNums_.find(receiveNext_);

  while (it != receivedSeqNums_.end()) {
    if (it->second) {
      notifyCount++;
    }
    receivedSeqNums_.erase(it);
    receiveNext_ = (receiveNext_ + 1) % kSeqNumMask;
    it = receivedSeqNums_.find(receiveNext_);
  }
  return notifyCount;
}

} // namespace ibverbx
