// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/* Basic IB verbs structs. Needed to dynamically load IB verbs functions without
 * explicit including of IB verbs header.
 */

#include <fmt/format.h>
#include <linux/types.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#if __GNUC__ >= 3
#define __attribute_const __attribute__((const))
#else
#define __attribute_const
#endif

namespace ibverbx {

// always define ibverbx:: own types which should be identical to the IB verbs:
// third-party/rdma-core/stablev53-iris/libibverbs/verbs.h
union ibv_gid {
  uint8_t raw[16];
  struct {
    uint64_t subnet_prefix;
    uint64_t interface_id;
  } global;
};

#ifndef container_of
/**
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:        the pointer to the member.
 * @type:       the type of the container struct this is embedded in.
 * @member:     the name of the member within the struct.
 *
 */
#define container_of(ptr, type, member) \
  ((type*)((uint8_t*)(ptr) - offsetof(type, member)))
#endif

#define vext_field_avail(type, fld, sz) (offsetof(type, fld) < (sz))

/*XXX:__VERBS_ABI_IS_EXTENDED produces warning "integer operation result is out
 * of range" with g++ 4.8.2*/
// static void *__VERBS_ABI_IS_EXTENDED = ((uint8_t *)NULL) - 1;

enum ibv_node_type {
  IBV_NODE_UNKNOWN = -1,
  IBV_NODE_CA = 1,
  IBV_NODE_SWITCH,
  IBV_NODE_ROUTER,
  IBV_NODE_RNIC,

  /* Leave a gap for future node types before starting with
   * experimental node types.
   */
  IBV_EXP_NODE_TYPE_START = 32,
  IBV_EXP_NODE_MIC = IBV_EXP_NODE_TYPE_START
};

enum ibv_transport_type {
  IBV_TRANSPORT_UNKNOWN = -1,
  IBV_TRANSPORT_IB = 0,
  IBV_TRANSPORT_IWARP,

  /* Leave a gap for future transport types before starting with
   * experimental transport types.
   */
  IBV_EXP_TRANSPORT_TYPE_START = 32,
  IBV_EXP_TRANSPORT_SCIF = IBV_EXP_TRANSPORT_TYPE_START
};

enum ibv_device_cap_flags {
  IBV_DEVICE_RESIZE_MAX_WR = 1,
  IBV_DEVICE_BAD_PKEY_CNTR = 1 << 1,
  IBV_DEVICE_BAD_QKEY_CNTR = 1 << 2,
  IBV_DEVICE_RAW_MULTI = 1 << 3,
  IBV_DEVICE_AUTO_PATH_MIG = 1 << 4,
  IBV_DEVICE_CHANGE_PHY_PORT = 1 << 5,
  IBV_DEVICE_UD_AV_PORT_ENFORCE = 1 << 6,
  IBV_DEVICE_CURR_QP_STATE_MOD = 1 << 7,
  IBV_DEVICE_SHUTDOWN_PORT = 1 << 8,
  IBV_DEVICE_INIT_TYPE = 1 << 9,
  IBV_DEVICE_PORT_ACTIVE_EVENT = 1 << 10,
  IBV_DEVICE_SYS_IMAGE_GUID = 1 << 11,
  IBV_DEVICE_RC_RNR_NAK_GEN = 1 << 12,
  IBV_DEVICE_SRQ_RESIZE = 1 << 13,
  IBV_DEVICE_N_NOTIFY_CQ = 1 << 14,
  IBV_DEVICE_XRC = 1 << 20,
  IBV_DEVICE_MANAGED_FLOW_STEERING = 1 << 29
};

enum ibv_atomic_cap { IBV_ATOMIC_NONE, IBV_ATOMIC_HCA, IBV_ATOMIC_GLOB };

struct ibv_device_attr {
  char fw_ver[64];
  uint64_t node_guid;
  uint64_t sys_image_guid;
  uint64_t max_mr_size;
  uint64_t page_size_cap;
  uint32_t vendor_id;
  uint32_t vendor_part_id;
  uint32_t hw_ver;
  int max_qp;
  int max_qp_wr;
  int device_cap_flags;
  int max_sge;
  int max_sge_rd;
  int max_cq;
  int max_cqe;
  int max_mr;
  int max_pd;
  int max_qp_rd_atom;
  int max_ee_rd_atom;
  int max_res_rd_atom;
  int max_qp_init_rd_atom;
  int max_ee_init_rd_atom;
  enum ibv_atomic_cap atomic_cap;
  int max_ee;
  int max_rdd;
  int max_mw;
  int max_raw_ipv6_qp;
  int max_raw_ethy_qp;
  int max_mcast_grp;
  int max_mcast_qp_attach;
  int max_total_mcast_qp_attach;
  int max_ah;
  int max_fmr;
  int max_map_per_fmr;
  int max_srq;
  int max_srq_wr;
  int max_srq_sge;
  uint16_t max_pkeys;
  uint8_t local_ca_ack_delay;
  uint8_t phys_port_cnt;
};

enum ibv_mtu {
  IBV_MTU_256 = 1,
  IBV_MTU_512 = 2,
  IBV_MTU_1024 = 3,
  IBV_MTU_2048 = 4,
  IBV_MTU_4096 = 5
};

enum ibv_port_state {
  IBV_PORT_NOP = 0,
  IBV_PORT_DOWN = 1,
  IBV_PORT_INIT = 2,
  IBV_PORT_ARMED = 3,
  IBV_PORT_ACTIVE = 4,
  IBV_PORT_ACTIVE_DEFER = 5
};

enum {
  IBV_LINK_LAYER_UNSPECIFIED,
  IBV_LINK_LAYER_INFINIBAND,
  IBV_LINK_LAYER_ETHERNET,

  /* Leave a gap for future link layer types before starting with
   * experimental link layer.
   */
  IBV_EXP_LINK_LAYER_START = 32,
  IBV_EXP_LINK_LAYER_SCIF = IBV_EXP_LINK_LAYER_START
};

enum ibv_port_cap_flags {
  IBV_PORT_SM = 1 << 1,
  IBV_PORT_NOTICE_SUP = 1 << 2,
  IBV_PORT_TRAP_SUP = 1 << 3,
  IBV_PORT_OPT_IPD_SUP = 1 << 4,
  IBV_PORT_AUTO_MIGR_SUP = 1 << 5,
  IBV_PORT_SL_MAP_SUP = 1 << 6,
  IBV_PORT_MKEY_NVRAM = 1 << 7,
  IBV_PORT_PKEY_NVRAM = 1 << 8,
  IBV_PORT_LED_INFO_SUP = 1 << 9,
  IBV_PORT_SYS_IMAGE_GUID_SUP = 1 << 11,
  IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP = 1 << 12,
  IBV_PORT_EXTENDED_SPEEDS_SUP = 1 << 14,
  IBV_PORT_CM_SUP = 1 << 16,
  IBV_PORT_SNMP_TUNNEL_SUP = 1 << 17,
  IBV_PORT_REINIT_SUP = 1 << 18,
  IBV_PORT_DEVICE_MGMT_SUP = 1 << 19,
  IBV_PORT_VENDOR_CLASS = 1 << 24,
  IBV_PORT_CLIENT_REG_SUP = 1 << 25,
  IBV_PORT_IP_BASED_GIDS = 1 << 26,
};

struct ibv_port_attr {
  enum ibv_port_state state;
  enum ibv_mtu max_mtu;
  enum ibv_mtu active_mtu;
  int gid_tbl_len;
  uint32_t port_cap_flags;
  uint32_t max_msg_sz;
  uint32_t bad_pkey_cntr;
  uint32_t qkey_viol_cntr;
  uint16_t pkey_tbl_len;
  uint16_t lid;
  uint16_t sm_lid;
  uint8_t lmc;
  uint8_t max_vl_num;
  uint8_t sm_sl;
  uint8_t subnet_timeout;
  uint8_t init_type_reply;
  uint8_t active_width;
  uint8_t active_speed;
  uint8_t phys_state;
  uint8_t link_layer;
  uint8_t reserved;
};

enum ibv_event_type {
  IBV_EVENT_CQ_ERR,
  IBV_EVENT_QP_FATAL,
  IBV_EVENT_QP_REQ_ERR,
  IBV_EVENT_QP_ACCESS_ERR,
  IBV_EVENT_COMM_EST,
  IBV_EVENT_SQ_DRAINED,
  IBV_EVENT_PATH_MIG,
  IBV_EVENT_PATH_MIG_ERR,
  IBV_EVENT_DEVICE_FATAL,
  IBV_EVENT_PORT_ACTIVE,
  IBV_EVENT_PORT_ERR,
  IBV_EVENT_LID_CHANGE,
  IBV_EVENT_PKEY_CHANGE,
  IBV_EVENT_SM_CHANGE,
  IBV_EVENT_SRQ_ERR,
  IBV_EVENT_SRQ_LIMIT_REACHED,
  IBV_EVENT_QP_LAST_WQE_REACHED,
  IBV_EVENT_CLIENT_REREGISTER,
  IBV_EVENT_GID_CHANGE,

  /* new experimental events start here leaving enough
   * room for 14 events which should be enough
   */
  IBV_EXP_EVENT_DCT_KEY_VIOLATION = 32,
  IBV_EXP_EVENT_DCT_ACCESS_ERR,
  IBV_EXP_EVENT_DCT_REQ_ERR,
};

struct ibv_async_event {
  union {
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_srq* srq;
    struct ibv_exp_dct* dct;
    int port_num;
    /* For source compatible with Legacy API */
    uint32_t xrc_qp_num;
  } element;
  enum ibv_event_type event_type;
};

enum ibv_wc_status {
  IBV_WC_SUCCESS,
  IBV_WC_LOC_LEN_ERR,
  IBV_WC_LOC_QP_OP_ERR,
  IBV_WC_LOC_EEC_OP_ERR,
  IBV_WC_LOC_PROT_ERR,
  IBV_WC_WR_FLUSH_ERR,
  IBV_WC_MW_BIND_ERR,
  IBV_WC_BAD_RESP_ERR,
  IBV_WC_LOC_ACCESS_ERR,
  IBV_WC_REM_INV_REQ_ERR,
  IBV_WC_REM_ACCESS_ERR,
  IBV_WC_REM_OP_ERR,
  IBV_WC_RETRY_EXC_ERR,
  IBV_WC_RNR_RETRY_EXC_ERR,
  IBV_WC_LOC_RDD_VIOL_ERR,
  IBV_WC_REM_INV_RD_REQ_ERR,
  IBV_WC_REM_ABORT_ERR,
  IBV_WC_INV_EECN_ERR,
  IBV_WC_INV_EEC_STATE_ERR,
  IBV_WC_FATAL_ERR,
  IBV_WC_RESP_TIMEOUT_ERR,
  IBV_WC_GENERAL_ERR
};

enum ibv_wc_opcode {
  IBV_WC_SEND,
  IBV_WC_RDMA_WRITE,
  IBV_WC_RDMA_READ,
  IBV_WC_COMP_SWAP,
  IBV_WC_FETCH_ADD,
  IBV_WC_BIND_MW,
  /*
   * Set value of IBV_WC_RECV so consumers can test if a completion is a
   * receive by testing (opcode & IBV_WC_RECV).
   */
  IBV_WC_RECV = 1 << 7,
  IBV_WC_RECV_RDMA_WITH_IMM
};

enum ibv_wc_flags { IBV_WC_GRH = 1 << 0, IBV_WC_WITH_IMM = 1 << 1 };

struct ibv_wc {
  uint64_t wr_id;
  enum ibv_wc_status status;
  enum ibv_wc_opcode opcode;
  uint32_t vendor_err;
  uint32_t byte_len;
  uint32_t imm_data; /* in network byte order */
  uint32_t qp_num;
  uint32_t src_qp;
  int wc_flags;
  uint16_t pkey_index;
  uint16_t slid;
  uint8_t sl;
  uint8_t dlid_path_bits;
};

enum ibv_access_flags {
  IBV_ACCESS_LOCAL_WRITE = 1,
  IBV_ACCESS_REMOTE_WRITE = (1 << 1),
  IBV_ACCESS_REMOTE_READ = (1 << 2),
  IBV_ACCESS_REMOTE_ATOMIC = (1 << 3),
  IBV_ACCESS_MW_BIND = (1 << 4),
  IBV_ACCESS_RELAXED_ORDERING = (1 << 20),
};

struct ibv_pd {
  struct ibv_context* context;
  uint32_t handle;
};

enum ibv_xrcd_init_attr_mask {
  IBV_XRCD_INIT_ATTR_FD = 1 << 0,
  IBV_XRCD_INIT_ATTR_OFLAGS = 1 << 1,
  IBV_XRCD_INIT_ATTR_RESERVED = 1 << 2
};

struct ibv_xrcd_init_attr {
  uint32_t comp_mask;
  int fd;
  int oflags;
};

struct ibv_xrcd {
  struct ibv_context* context;
};

enum ibv_rereg_mr_flags {
  IBV_REREG_MR_CHANGE_TRANSLATION = (1 << 0),
  IBV_REREG_MR_CHANGE_PD = (1 << 1),
  IBV_REREG_MR_CHANGE_ACCESS = (1 << 2),
  IBV_REREG_MR_KEEP_VALID = (1 << 3)
};

struct ibv_mr {
  struct ibv_context* context;
  struct ibv_pd* pd;
  void* addr;
  size_t length;
  uint32_t handle;
  uint32_t lkey;
  uint32_t rkey;
};

enum ibv_mw_type { IBV_MW_TYPE_1 = 1, IBV_MW_TYPE_2 = 2 };

struct ibv_mw {
  struct ibv_context* context;
  struct ibv_pd* pd;
  uint32_t rkey;
};

struct ibv_global_route {
  union ibv_gid dgid;
  uint32_t flow_label;
  uint8_t sgid_index;
  uint8_t hop_limit;
  uint8_t traffic_class;
};

struct ibv_grh {
  uint32_t version_tclass_flow;
  uint16_t paylen;
  uint8_t next_hdr;
  uint8_t hop_limit;
  union ibv_gid sgid;
  union ibv_gid dgid;
};

enum ibv_rate {
  IBV_RATE_MAX = 0,
  IBV_RATE_2_5_GBPS = 2,
  IBV_RATE_5_GBPS = 5,
  IBV_RATE_10_GBPS = 3,
  IBV_RATE_20_GBPS = 6,
  IBV_RATE_30_GBPS = 4,
  IBV_RATE_40_GBPS = 7,
  IBV_RATE_60_GBPS = 8,
  IBV_RATE_80_GBPS = 9,
  IBV_RATE_120_GBPS = 10,
  IBV_RATE_14_GBPS = 11,
  IBV_RATE_56_GBPS = 12,
  IBV_RATE_112_GBPS = 13,
  IBV_RATE_168_GBPS = 14,
  IBV_RATE_25_GBPS = 15,
  IBV_RATE_100_GBPS = 16,
  IBV_RATE_200_GBPS = 17,
  IBV_RATE_300_GBPS = 18
};

/**
 * ibv_rate_to_mult - Convert the IB rate enum to a multiple of the
 * base rate of 2.5 Gbit/sec.  For example, IBV_RATE_5_GBPS will be
 * converted to 2, since 5 Gbit/sec is 2 * 2.5 Gbit/sec.
 * @rate: rate to convert.
 */
int ibv_rate_to_mult(enum ibv_rate rate) __attribute_const;

/**
 * mult_to_ibv_rate - Convert a multiple of 2.5 Gbit/sec to an IB rate enum.
 * @mult: multiple to convert.
 */
enum ibv_rate mult_to_ibv_rate(int mult) __attribute_const;

/**
 * ibv_rate_to_mbps - Convert the IB rate enum to Mbit/sec.
 * For example, IBV_RATE_5_GBPS will return the value 5000.
 * @rate: rate to convert.
 */
int ibv_rate_to_mbps(enum ibv_rate rate) __attribute_const;

/**
 * mbps_to_ibv_rate - Convert a Mbit/sec value to an IB rate enum.
 * @mbps: value to convert.
 */
enum ibv_rate mbps_to_ibv_rate(int mbps) __attribute_const;

struct ibv_ah_attr {
  struct ibv_global_route grh;
  uint16_t dlid;
  uint8_t sl;
  uint8_t src_path_bits;
  uint8_t static_rate;
  uint8_t is_global;
  uint8_t port_num;
};

enum ibv_srq_attr_mask { IBV_SRQ_MAX_WR = 1 << 0, IBV_SRQ_LIMIT = 1 << 1 };

struct ibv_srq_attr {
  uint32_t max_wr;
  uint32_t max_sge;
  uint32_t srq_limit;
};

struct ibv_srq_init_attr {
  void* srq_context;
  struct ibv_srq_attr attr;
};

enum ibv_srq_type { IBV_SRQT_BASIC, IBV_SRQT_XRC };

enum ibv_srq_init_attr_mask {
  IBV_SRQ_INIT_ATTR_TYPE = 1 << 0,
  IBV_SRQ_INIT_ATTR_PD = 1 << 1,
  IBV_SRQ_INIT_ATTR_XRCD = 1 << 2,
  IBV_SRQ_INIT_ATTR_CQ = 1 << 3,
  IBV_SRQ_INIT_ATTR_RESERVED = 1 << 4
};

struct ibv_srq_init_attr_ex {
  void* srq_context;
  struct ibv_srq_attr attr;

  uint32_t comp_mask;
  enum ibv_srq_type srq_type;
  struct ibv_pd* pd;
  struct ibv_xrcd* xrcd;
  struct ibv_cq* cq;
};

enum ibv_qp_type {
  IBV_QPT_RC = 2,
  IBV_QPT_UC,
  IBV_QPT_UD,
  /* XRC compatible code */
  IBV_QPT_XRC,
  IBV_QPT_RAW_PACKET = 8,
  IBV_QPT_RAW_ETH = 8,
  IBV_QPT_XRC_SEND = 9,
  IBV_QPT_XRC_RECV,

  /* Leave a gap for future qp types before starting with
   * experimental qp types.
   */
  IBV_EXP_QP_TYPE_START = 32,
  IBV_EXP_QPT_DC_INI = IBV_EXP_QP_TYPE_START
};

struct ibv_qp_cap {
  uint32_t max_send_wr;
  uint32_t max_recv_wr;
  uint32_t max_send_sge;
  uint32_t max_recv_sge;
  uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
  void* qp_context;
  struct ibv_cq* send_cq;
  struct ibv_cq* recv_cq;
  struct ibv_srq* srq;
  struct ibv_qp_cap cap;
  enum ibv_qp_type qp_type;
  int sq_sig_all;
  /* Below is needed for backwards compatabile */
  struct ibv_xrc_domain* xrc_domain;
};

enum ibv_qp_init_attr_mask {
  IBV_QP_INIT_ATTR_PD = 1 << 0,
  IBV_QP_INIT_ATTR_XRCD = 1 << 1,
  IBV_QP_INIT_ATTR_RESERVED = 1 << 2
};

struct ibv_qp_init_attr_ex {
  void* qp_context;
  struct ibv_cq* send_cq;
  struct ibv_cq* recv_cq;
  struct ibv_srq* srq;
  struct ibv_qp_cap cap;
  enum ibv_qp_type qp_type;
  int sq_sig_all;

  uint32_t comp_mask;
  struct ibv_pd* pd;
  struct ibv_xrcd* xrcd;
};

enum ibv_qp_open_attr_mask {
  IBV_QP_OPEN_ATTR_NUM = 1 << 0,
  IBV_QP_OPEN_ATTR_XRCD = 1 << 1,
  IBV_QP_OPEN_ATTR_CONTEXT = 1 << 2,
  IBV_QP_OPEN_ATTR_TYPE = 1 << 3,
  IBV_QP_OPEN_ATTR_RESERVED = 1 << 4
};

struct ibv_qp_open_attr {
  uint32_t comp_mask;
  uint32_t qp_num;
  struct ibv_xrcd* xrcd;
  void* qp_context;
  enum ibv_qp_type qp_type;
};

enum ibv_qp_attr_mask {
  IBV_QP_STATE = 1 << 0,
  IBV_QP_CUR_STATE = 1 << 1,
  IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2,
  IBV_QP_ACCESS_FLAGS = 1 << 3,
  IBV_QP_PKEY_INDEX = 1 << 4,
  IBV_QP_PORT = 1 << 5,
  IBV_QP_QKEY = 1 << 6,
  IBV_QP_AV = 1 << 7,
  IBV_QP_PATH_MTU = 1 << 8,
  IBV_QP_TIMEOUT = 1 << 9,
  IBV_QP_RETRY_CNT = 1 << 10,
  IBV_QP_RNR_RETRY = 1 << 11,
  IBV_QP_RQ_PSN = 1 << 12,
  IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13,
  IBV_QP_ALT_PATH = 1 << 14,
  IBV_QP_MIN_RNR_TIMER = 1 << 15,
  IBV_QP_SQ_PSN = 1 << 16,
  IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
  IBV_QP_PATH_MIG_STATE = 1 << 18,
  IBV_QP_CAP = 1 << 19,
  IBV_QP_DEST_QPN = 1 << 20
};

enum ibv_qp_state {
  IBV_QPS_RESET,
  IBV_QPS_INIT,
  IBV_QPS_RTR,
  IBV_QPS_RTS,
  IBV_QPS_SQD,
  IBV_QPS_SQE,
  IBV_QPS_ERR,
  IBV_QPS_UNKNOWN
};

enum ibv_mig_state { IBV_MIG_MIGRATED, IBV_MIG_REARM, IBV_MIG_ARMED };

struct ibv_qp_attr {
  enum ibv_qp_state qp_state;
  enum ibv_qp_state cur_qp_state;
  enum ibv_mtu path_mtu;
  enum ibv_mig_state path_mig_state;
  uint32_t qkey;
  uint32_t rq_psn;
  uint32_t sq_psn;
  uint32_t dest_qp_num;
  int qp_access_flags;
  struct ibv_qp_cap cap;
  struct ibv_ah_attr ah_attr;
  struct ibv_ah_attr alt_ah_attr;
  uint16_t pkey_index;
  uint16_t alt_pkey_index;
  uint8_t en_sqd_async_notify;
  uint8_t sq_draining;
  uint8_t max_rd_atomic;
  uint8_t max_dest_rd_atomic;
  uint8_t min_rnr_timer;
  uint8_t port_num;
  uint8_t timeout;
  uint8_t retry_cnt;
  uint8_t rnr_retry;
  uint8_t alt_port_num;
  uint8_t alt_timeout;
};

enum ibv_wr_opcode {
  IBV_WR_RDMA_WRITE,
  IBV_WR_RDMA_WRITE_WITH_IMM,
  IBV_WR_SEND,
  IBV_WR_SEND_WITH_IMM,
  IBV_WR_RDMA_READ,
  IBV_WR_ATOMIC_CMP_AND_SWP,
  IBV_WR_ATOMIC_FETCH_AND_ADD
};

enum ibv_send_flags {
  IBV_SEND_FENCE = 1 << 0,
  IBV_SEND_SIGNALED = 1 << 1,
  IBV_SEND_SOLICITED = 1 << 2,
  IBV_SEND_INLINE = 1 << 3
};

struct ibv_sge {
  uint64_t addr;
  uint32_t length;
  uint32_t lkey;
};

struct ibv_send_wr {
  uint64_t wr_id;
  struct ibv_send_wr* next;
  struct ibv_sge* sg_list;
  int num_sge;
  enum ibv_wr_opcode opcode;
  int send_flags;
  uint32_t imm_data; /* in network byte order */
  union {
    struct {
      uint64_t remote_addr;
      uint32_t rkey;
    } rdma;
    struct {
      uint64_t remote_addr;
      uint64_t compare_add;
      uint64_t swap;
      uint32_t rkey;
    } atomic;
    struct {
      struct ibv_ah* ah;
      uint32_t remote_qpn;
      uint32_t remote_qkey;
    } ud;
  } wr;
  union {
    union {
      struct {
        uint32_t remote_srqn;
      } xrc;
    } qp_type;

    uint32_t xrc_remote_srq_num;
  };
};

struct ibv_recv_wr {
  uint64_t wr_id;
  struct ibv_recv_wr* next;
  struct ibv_sge* sg_list;
  int num_sge;
};

struct ibv_mw_bind {
  uint64_t wr_id;
  struct ibv_mr* mr;
  void* addr;
  size_t length;
  int send_flags;
  int mw_access_flags;
};

struct ibv_srq {
  struct ibv_context* context;
  void* srq_context;
  struct ibv_pd* pd;
  uint32_t handle;

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t events_completed;

  /* below are for source compatabilty with legacy XRC,
   *   padding based on ibv_srq_legacy.
   */
  uint32_t xrc_srq_num_bin_compat_padding;
  struct ibv_xrc_domain* xrc_domain_bin_compat_padding;
  struct ibv_cq* xrc_cq_bin_compat_padding;
  void* ibv_srq_padding;

  /* legacy fields */
  uint32_t xrc_srq_num;
  struct ibv_xrc_domain* xrc_domain;
  struct ibv_cq* xrc_cq;
};

/* Not in use in new API, needed for compilation as part of source compat layer
 */
enum ibv_event_flags {
  IBV_XRC_QP_EVENT_FLAG = 0x80000000,
};

struct ibv_qp {
  struct ibv_context* context;
  void* qp_context;
  struct ibv_pd* pd;
  struct ibv_cq* send_cq;
  struct ibv_cq* recv_cq;
  struct ibv_srq* srq;
  uint32_t handle;
  uint32_t qp_num;
  enum ibv_qp_state state;
  enum ibv_qp_type qp_type;

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t events_completed;
};

struct ibv_comp_channel {
  struct ibv_context* context;
  int fd;
  int refcnt;
};

struct ibv_cq {
  struct ibv_context* context;
  struct ibv_comp_channel* channel;
  void* cq_context;
  uint32_t handle;
  int cqe;

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t comp_events_completed;
  uint32_t async_events_completed;
};

struct ibv_ah {
  struct ibv_context* context;
  struct ibv_pd* pd;
  uint32_t handle;
};

enum ibv_flow_flags {
  IBV_FLOW_ATTR_FLAGS_ALLOW_LOOP_BACK = 1,
  IBV_FLOW_ATTR_FLAGS_DONT_TRAP = 1 << 1,
};

enum ibv_flow_attr_type {
  /* steering according to rule specifications */
  IBV_FLOW_ATTR_NORMAL = 0x0,
  /* default unicast and multicast rule -
   * receive all Eth traffic which isn't steered to any QP
   */
  IBV_FLOW_ATTR_ALL_DEFAULT = 0x1,
  /* default multicast rule -
   * receive all Eth multicast traffic which isn't steered to any QP
   */
  IBV_FLOW_ATTR_MC_DEFAULT = 0x2,
};

enum ibv_flow_spec_type {
  IBV_FLOW_SPEC_ETH = 0x20,
  IBV_FLOW_SPEC_IPV4 = 0x30,
  IBV_FLOW_SPEC_TCP = 0x40,
  IBV_FLOW_SPEC_UDP = 0x41,
};

struct ibv_flow_eth_filter {
  uint8_t dst_mac[6];
  uint8_t src_mac[6];
  uint16_t ether_type;
  /*
   * same layout as 802.1q: prio 3, cfi 1, vlan id 12
   */
  uint16_t vlan_tag;
};

struct ibv_flow_spec_eth {
  enum ibv_flow_spec_type type;
  uint16_t size;
  struct ibv_flow_eth_filter val;
  struct ibv_flow_eth_filter mask;
};

struct ibv_flow_ipv4_filter {
  uint32_t src_ip;
  uint32_t dst_ip;
};

struct ibv_flow_spec_ipv4 {
  enum ibv_flow_spec_type type;
  uint16_t size;
  struct ibv_flow_ipv4_filter val;
  struct ibv_flow_ipv4_filter mask;
};

struct ibv_flow_tcp_udp_filter {
  uint16_t dst_port;
  uint16_t src_port;
};

struct ibv_flow_spec_tcp_udp {
  enum ibv_flow_spec_type type;
  uint16_t size;
  struct ibv_flow_tcp_udp_filter val;
  struct ibv_flow_tcp_udp_filter mask;
};

struct ibv_flow_spec {
  union {
    struct {
      enum ibv_flow_spec_type type;
      uint16_t size;
    } hdr;
    struct ibv_flow_spec_eth eth;
    struct ibv_flow_spec_ipv4 ipv4;
    struct ibv_flow_spec_tcp_udp tcp_udp;
  };
};

struct ibv_flow_attr {
  uint32_t comp_mask;
  enum ibv_flow_attr_type type;
  uint16_t size;
  uint16_t priority;
  uint8_t num_of_specs;
  uint8_t port;
  uint32_t flags;
  /* Following are the optional layers according to user request
   * struct ibv_flow_spec_xxx [L2]
   * struct ibv_flow_spec_yyy [L3/L4]
   */
};

struct ibv_flow {
  uint32_t comp_mask;
  struct ibv_context* context;
  uint32_t handle;
};

struct ibv_device;
struct ibv_context;

struct ibv_device_ops {
  struct ibv_context* (*alloc_context)(struct ibv_device* device, int cmd_fd);
  void (*free_context)(struct ibv_context* context);
};

enum { IBV_SYSFS_NAME_MAX = 64, IBV_SYSFS_PATH_MAX = 256 };

struct ibv_device {
  struct ibv_device_ops ops;
  enum ibv_node_type node_type;
  enum ibv_transport_type transport_type;
  /* Name of underlying kernel IB device, eg "mthca0" */
  char name[IBV_SYSFS_NAME_MAX];
  /* Name of uverbs device, eg "uverbs0" */
  char dev_name[IBV_SYSFS_NAME_MAX];
  /* Path to infiniband_verbs class device in sysfs */
  char dev_path[IBV_SYSFS_PATH_MAX];
  /* Path to infiniband class device in sysfs */
  char ibdev_path[IBV_SYSFS_PATH_MAX];
};

struct verbs_device {
  struct ibv_device device; /* Must be first */
  size_t sz;
  size_t size_of_context;
  int (*init_context)(
      struct verbs_device* device,
      struct ibv_context* ctx,
      int cmd_fd);
  void (*uninit_context)(struct verbs_device* device, struct ibv_context* ctx);
  /* future fields added here */
};

struct ibv_context_ops {
  int (*query_device)(
      struct ibv_context* context,
      struct ibv_device_attr* device_attr);
  int (*query_port)(
      struct ibv_context* context,
      uint8_t port_num,
      struct ibv_port_attr* port_attr);
  struct ibv_pd* (*alloc_pd)(struct ibv_context* context);
  int (*dealloc_pd)(struct ibv_pd* pd);
  struct ibv_mr* (
      *reg_mr)(struct ibv_pd* pd, void* addr, size_t length, int access);
  struct ibv_mr* (*rereg_mr)(
      struct ibv_mr* mr,
      int flags,
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      int access);
  int (*dereg_mr)(struct ibv_mr* mr);
  struct ibv_mw* (*alloc_mw)(struct ibv_pd* pd, enum ibv_mw_type type);
  int (*bind_mw)(
      struct ibv_qp* qp,
      struct ibv_mw* mw,
      struct ibv_mw_bind* mw_bind);
  int (*dealloc_mw)(struct ibv_mw* mw);
  struct ibv_cq* (*create_cq)(
      struct ibv_context* context,
      int cqe,
      struct ibv_comp_channel* channel,
      int comp_vector);
  int (*poll_cq)(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc);
  int (*req_notify_cq)(struct ibv_cq* cq, int solicited_only);
  void (*cq_event)(struct ibv_cq* cq);
  int (*resize_cq)(struct ibv_cq* cq, int cqe);
  int (*destroy_cq)(struct ibv_cq* cq);
  struct ibv_srq* (
      *create_srq)(struct ibv_pd* pd, struct ibv_srq_init_attr* srq_init_attr);
  int (*modify_srq)(
      struct ibv_srq* srq,
      struct ibv_srq_attr* srq_attr,
      int srq_attr_mask);
  int (*query_srq)(struct ibv_srq* srq, struct ibv_srq_attr* srq_attr);
  int (*destroy_srq)(struct ibv_srq* srq);
  int (*post_srq_recv)(
      struct ibv_srq* srq,
      struct ibv_recv_wr* recv_wr,
      struct ibv_recv_wr** bad_recv_wr);
  struct ibv_qp* (*create_qp)(struct ibv_pd* pd, struct ibv_qp_init_attr* attr);
  int (*query_qp)(
      struct ibv_qp* qp,
      struct ibv_qp_attr* attr,
      int attr_mask,
      struct ibv_qp_init_attr* init_attr);
  int (*modify_qp)(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask);
  int (*destroy_qp)(struct ibv_qp* qp);
  int (*post_send)(
      struct ibv_qp* qp,
      struct ibv_send_wr* wr,
      struct ibv_send_wr** bad_wr);
  int (*post_recv)(
      struct ibv_qp* qp,
      struct ibv_recv_wr* wr,
      struct ibv_recv_wr** bad_wr);
  struct ibv_ah* (*create_ah)(struct ibv_pd* pd, struct ibv_ah_attr* attr);
  int (*destroy_ah)(struct ibv_ah* ah);
  int (
      *attach_mcast)(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid);
  int (
      *detach_mcast)(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid);
  void (*async_event)(struct ibv_async_event* event);
};

struct ibv_context {
  struct ibv_device* device;
  struct ibv_context_ops ops;
  int cmd_fd;
  int async_fd;
  int num_comp_vectors;
  pthread_mutex_t mutex;
  void* abi_compat;
};

enum verbs_context_mask {
  VERBS_CONTEXT_XRCD = (uint64_t)1 << 0,
  VERBS_CONTEXT_SRQ = (uint64_t)1 << 1,
  VERBS_CONTEXT_QP = (uint64_t)1 << 2,
  VERBS_CONTEXT_RESERVED = (uint64_t)1 << 3,
  VERBS_CONTEXT_EXP = (uint64_t)1 << 62
};

struct verbs_context {
  /*  "grows up" - new fields go here */
  int (*_reserved_2)(void);
  int (*destroy_flow)(struct ibv_flow* flow);
  int (*_reserved_1)(void);
  struct ibv_flow* (
      *create_flow)(struct ibv_qp* qp, struct ibv_flow_attr* flow_attr);
  struct ibv_qp* (
      *open_qp)(struct ibv_context* context, struct ibv_qp_open_attr* attr);
  struct ibv_qp* (*create_qp_ex)(
      struct ibv_context* context,
      struct ibv_qp_init_attr_ex* qp_init_attr_ex);
  int (*get_srq_num)(struct ibv_srq* srq, uint32_t* srq_num);
  struct ibv_srq* (*create_srq_ex)(
      struct ibv_context* context,
      struct ibv_srq_init_attr_ex* srq_init_attr_ex);
  struct ibv_xrcd* (*open_xrcd)(
      struct ibv_context* context,
      struct ibv_xrcd_init_attr* xrcd_init_attr);
  int (*close_xrcd)(struct ibv_xrcd* xrcd);
  uint64_t has_comp_mask;
  size_t sz; /* Must be immediately before struct ibv_context */
  struct ibv_context context; /* Must be last field in the struct */
};

/*XXX:__VERBS_ABI_IS_EXTENDED produces warning "integer operation result is out
 * of range" with g++ 4.8.2*/
/*static inline struct verbs_context *verbs_get_ctx(struct ibv_context *ctx)
{
        return (!ctx || (ctx->abi_compat != __VERBS_ABI_IS_EXTENDED)) ?
                NULL : container_of(ctx, struct verbs_context, context);
}

#define verbs_get_ctx_op(ctx, op) ({ \
        struct verbs_context *_vctx = verbs_get_ctx(ctx); \
        (!_vctx || (_vctx->sz < sizeof(*_vctx) - offsetof(struct verbs_context,
op)) || \
        !_vctx->op) ? NULL : _vctx; })*/

#define verbs_set_ctx_op(_vctx, op, ptr)                                  \
  ({                                                                      \
    struct verbs_context* vctx = _vctx;                                   \
    if (vctx &&                                                           \
        (vctx->sz >= sizeof(*vctx) - offsetof(struct verbs_context, op))) \
      vctx->op = ptr;                                                     \
  })

static inline struct verbs_device* verbs_get_device(struct ibv_device* dev) {
  return (dev->ops.alloc_context)
      ? NULL
      : container_of(dev, struct verbs_device, device);
}

struct ibv_ece {
  /*
   * Unique identifier of the provider vendor on the network.
   * The providers will set IEEE OUI here to distinguish
   * itself in non-homogenius network.
   */
  uint32_t vendor_id;
  /*
   * Provider specific attributes which are supported or
   * needed to be enabled by ECE users.
   */
  uint32_t options;
  uint32_t comp_mask;
};

enum ibv_fork_status {
  IBV_FORK_DISABLED,
  IBV_FORK_ENABLED,
  IBV_FORK_UNNEEDED,
};

// DCT structures for experimental use
struct ibv_exp_dct {
  struct ibv_context* context;
  void* dct_context;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct ibv_srq* srq;
  uint32_t handle;
  uint32_t dct_num;

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t events_completed;
};

// XRC domain for legacy compatibility
struct ibv_xrc_domain {
  struct ibv_context* context;
  uint32_t handle;
};

struct ibv_cq_ex {
  struct ibv_context* context;
  struct ibv_comp_channel* channel;
  void* cq_context;
  uint32_t handle;
  int cqe;

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t comp_events_completed;
  uint32_t async_events_completed;

  uint32_t comp_mask;
  enum ibv_wc_status status;
  uint64_t wr_id;
  int (*start_poll)(struct ibv_cq_ex* current, struct ibv_poll_cq_attr* attr);
  int (*next_poll)(struct ibv_cq_ex* current);
  void (*end_poll)(struct ibv_cq_ex* current);
  enum ibv_wc_opcode (*read_opcode)(struct ibv_cq_ex* current);
  uint32_t (*read_vendor_err)(struct ibv_cq_ex* current);
  uint32_t (*read_byte_len)(struct ibv_cq_ex* current);
  __be32 (*read_imm_data)(struct ibv_cq_ex* current);
  uint32_t (*read_qp_num)(struct ibv_cq_ex* current);
  uint32_t (*read_src_qp)(struct ibv_cq_ex* current);
  unsigned int (*read_wc_flags)(struct ibv_cq_ex* current);
  uint32_t (*read_slid)(struct ibv_cq_ex* current);
  uint8_t (*read_sl)(struct ibv_cq_ex* current);
  uint8_t (*read_dlid_path_bits)(struct ibv_cq_ex* current);
  uint64_t (*read_completion_ts)(struct ibv_cq_ex* current);
  uint16_t (*read_cvlan)(struct ibv_cq_ex* current);
  uint32_t (*read_flow_tag)(struct ibv_cq_ex* current);
  void (
      *read_tm_info)(struct ibv_cq_ex* current, struct ibv_wc_tm_info* tm_info);
  uint64_t (*read_completion_wallclock_ns)(struct ibv_cq_ex* current);
};

inline struct ibv_cq* ibv_cq_ex_to_cq(struct ibv_cq_ex* cq) {
  return (struct ibv_cq*)cq;
}

enum ibv_cq_init_attr_mask {
  IBV_CQ_INIT_ATTR_MASK_FLAGS = 1 << 0,
  IBV_CQ_INIT_ATTR_MASK_PD = 1 << 1,
};

enum ibv_create_cq_attr_flags {
  IBV_CREATE_CQ_ATTR_SINGLE_THREADED = 1 << 0,
  IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN = 1 << 1,
};

struct ibv_cq_init_attr_ex {
  /* Minimum number of entries required for CQ */
  uint32_t cqe;
  /* Consumer-supplied context returned for completion events */
  void* cq_context;
  /* Completion channel where completion events will be queued.
   * May be NULL if completion events will not be used.
   */
  struct ibv_comp_channel* channel;
  /* Completion vector used to signal completion events.
   *  Must be < context->num_comp_vectors.
   */
  uint32_t comp_vector;
  /* Or'ed bit of enum ibv_create_cq_wc_flags. */
  uint64_t wc_flags;
  /* compatibility mask (extended verb). Or'd flags of
   * enum ibv_cq_init_attr_mask
   */
  uint32_t comp_mask;
  /* create cq attr flags - one or more flags from
   * enum ibv_create_cq_attr_flags
   */
  uint32_t flags;
  struct ibv_pd* parent_domain;
};

enum ibv_parent_domain_init_attr_mask {
  IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = 1 << 0,
  IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = 1 << 1,
};

struct ibv_parent_domain_init_attr {
  struct ibv_pd*
      pd; /* reference to a protection domain object, can't be NULL */
  struct ibv_td* td; /* reference to a thread domain object, or NULL */
  uint32_t comp_mask;
  void* (*alloc)(
      struct ibv_pd* pd,
      void* pd_context,
      size_t size,
      size_t alignment,
      uint64_t resource_type);
  void (*free)(
      struct ibv_pd* pd,
      void* pd_context,
      void* ptr,
      uint64_t resource_type);
  void* pd_context;
};

/* mlx5dv structs */

enum mlx5dv_reg_dmabuf_access {
  MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT = (1 << 0),
};

struct ibv_tmh {
  uint8_t opcode; /* from enum ibv_tmh_op */
  uint8_t reserved[3]; /* must be zero */
  __be32 app_ctx; /* opaque user data */
  __be64 tag;
};

struct mlx5_tm_cqe {
  __be32 success;
  __be16 hw_phase_cnt;
  uint8_t rsvd0[12];
};

struct mlx5_cqe64 {
  union {
    struct {
      uint8_t rsvd0[2];
      __be16 wqe_id;
      uint8_t rsvd4[13];
      uint8_t ml_path;
      uint8_t rsvd20[4];
      __be16 slid;
      __be32 flags_rqpn;
      uint8_t hds_ip_ext;
      uint8_t l4_hdr_type_etc;
      __be16 vlan_info;
    };
    struct mlx5_tm_cqe tm_cqe;
    /* TMH is scattered to CQE upon match */
    struct ibv_tmh tmh;
  };
  __be32 srqn_uidx;
  __be32 imm_inval_pkey;
  uint8_t app;
  uint8_t app_op;
  __be16 app_info;
  __be32 byte_cnt;
  __be64 timestamp;
  __be32 sop_drop_qpn;
  __be16 wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

struct mlx5dv_qp {
  __be32* dbrec;
  struct {
    void* buf;
    uint32_t wqe_cnt;
    uint32_t stride;
  } sq;
  struct {
    void* buf;
    uint32_t wqe_cnt;
    uint32_t stride;
  } rq;
  struct {
    void* reg;
    uint32_t size;
  } bf;
  uint64_t comp_mask;
  off_t uar_mmap_offset;
  uint32_t tirn;
  uint32_t tisn;
  uint32_t rqn;
  uint32_t sqn;
  uint64_t tir_icm_addr;
};

struct mlx5dv_cq {
  void* buf;
  __be32* dbrec;
  uint32_t cqe_cnt;
  uint32_t cqe_size;
  void* cq_uar;
  uint32_t cqn;
  uint64_t comp_mask;
};

struct mlx5dv_srq {
  void* buf;
  __be32* dbrec;
  uint32_t stride;
  uint32_t head;
  uint32_t tail;
  uint64_t comp_mask;
  uint32_t srqn;
};

struct mlx5dv_rwq {
  void* buf;
  __be32* dbrec;
  uint32_t wqe_cnt;
  uint32_t stride;
  uint64_t comp_mask;
};

struct mlx5dv_dm {
  void* buf;
  uint64_t length;
  uint64_t comp_mask;
  uint64_t remote_va;
};

struct mlx5_wqe_av;

struct mlx5dv_ah {
  struct mlx5_wqe_av* av;
  uint64_t comp_mask;
};

struct mlx5dv_pd {
  uint32_t pdn;
  uint64_t comp_mask;
};

struct mlx5dv_obj {
  struct {
    struct ibv_qp* in;
    struct mlx5dv_qp* out;
  } qp;
  struct {
    struct ibv_cq* in;
    struct mlx5dv_cq* out;
  } cq;
  struct {
    struct ibv_srq* in;
    struct mlx5dv_srq* out;
  } srq;
  struct {
    struct ibv_wq* in;
    struct mlx5dv_rwq* out;
  } rwq;
  struct {
    struct ibv_dm* in;
    struct mlx5dv_dm* out;
  } dm;
  struct {
    struct ibv_ah* in;
    struct mlx5dv_ah* out;
  } ah;
  struct {
    struct ibv_pd* in;
    struct mlx5dv_pd* out;
  } pd;
};

enum mlx5dv_obj_type {
  MLX5DV_OBJ_QP = 1 << 0,
  MLX5DV_OBJ_CQ = 1 << 1,
  MLX5DV_OBJ_SRQ = 1 << 2,
  MLX5DV_OBJ_RWQ = 1 << 3,
  MLX5DV_OBJ_DM = 1 << 4,
  MLX5DV_OBJ_AH = 1 << 5,
  MLX5DV_OBJ_PD = 1 << 6,
};

/*
 * WQE related part
 */

enum {
  MLX5_CQE_OWNER_MASK = 1,
  MLX5_CQE_REQ = 0,
  MLX5_CQE_RESP_WR_IMM = 1,
  MLX5_CQE_RESP_SEND = 2,
  MLX5_CQE_RESP_SEND_IMM = 3,
  MLX5_CQE_RESP_SEND_INV = 4,
  MLX5_CQE_RESIZE_CQ = 5,
  MLX5_CQE_NO_PACKET = 6,
  MLX5_CQE_SIG_ERR = 12,
  MLX5_CQE_REQ_ERR = 13,
  MLX5_CQE_RESP_ERR = 14,
  MLX5_CQE_INVALID = 15,
};

enum {
  MLX5_INVALID_LKEY = 0x100,
};

enum {
  MLX5_EXTENDED_UD_AV = 0x80000000,
};

enum {
  MLX5_WQE_CTRL_CQ_UPDATE = 2 << 2,
  MLX5_WQE_CTRL_SOLICITED = 1 << 1,
  MLX5_WQE_CTRL_FENCE = 4 << 5,
  MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE = 1 << 5,
};

enum {
  MLX5_SEND_WQE_BB = 64,
  MLX5_SEND_WQE_SHIFT = 6,
};

enum {
  MLX5_INLINE_SEG = 0x80000000,
};

enum {
  MLX5_ETH_WQE_L3_CSUM = (1 << 6),
  MLX5_ETH_WQE_L4_CSUM = (1 << 7),
};

struct mlx5_wqe_srq_next_seg {
  uint8_t rsvd0[2];
  __be16 next_wqe_index;
  uint8_t signature;
  uint8_t rsvd1[11];
};

struct mlx5_wqe_data_seg {
  __be32 byte_count;
  __be32 lkey;
  __be64 addr;
};

struct mlx5_wqe_ctrl_seg {
  __be32 opmod_idx_opcode;
  __be32 qpn_ds;
  uint8_t signature;
  __be16 dci_stream_channel_id;
  uint8_t fm_ce_se;
  __be32 imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct mlx5_wqe_raddr_seg {
  __be64 raddr;
  __be32 rkey;
  __be32 reserved;
};

struct mlx5_wqe_atomic_seg {
  __be64 swap_add;
  __be64 compare;
};

enum {
  MLX5_OPCODE_NOP = 0x00,
  MLX5_OPCODE_SEND_INVAL = 0x01,
  MLX5_OPCODE_RDMA_WRITE = 0x08,
  MLX5_OPCODE_RDMA_WRITE_IMM = 0x09,
  MLX5_OPCODE_SEND = 0x0a,
  MLX5_OPCODE_SEND_IMM = 0x0b,
  MLX5_OPCODE_TSO = 0x0e,
  MLX5_OPCODE_RDMA_READ = 0x10,
  MLX5_OPCODE_ATOMIC_CS = 0x11,
  MLX5_OPCODE_ATOMIC_FA = 0x12,
  MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
  MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,
  MLX5_OPCODE_FMR = 0x19,
  MLX5_OPCODE_LOCAL_INVAL = 0x1b,
  MLX5_OPCODE_CONFIG_CMD = 0x1f,
  MLX5_OPCODE_SET_PSV = 0x20,
  MLX5_OPCODE_UMR = 0x25,
  MLX5_OPCODE_TAG_MATCHING = 0x28,
  MLX5_OPCODE_FLOW_TBL_ACCESS = 0x2c,
  MLX5_OPCODE_MMO = 0x2F,
};

} // namespace ibverbx

template <>
struct fmt::formatter<ibverbx::ibv_event_type> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ibverbx::ibv_event_type status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

template <>
struct fmt::formatter<ibverbx::ibv_wc_status> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ibverbx::ibv_wc_status status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

template <>
struct fmt::formatter<ibverbx::ibv_wc_opcode> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ibverbx::ibv_wc_opcode status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
