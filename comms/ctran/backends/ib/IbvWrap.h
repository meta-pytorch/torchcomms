// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "comms/ctran/ibverbx/Ibvcore.h"
#endif

#include <sys/types.h>
#include <unistd.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

// Copy-pasted from NCCL with minor modifications (ncclResult_t -> commResult_t,
// etc.) [WIP]TODO: replace this c-style interface with comms/ctran/ibverbx

namespace ctran::ibvwrap {

typedef enum ibv_return_enum {
  IBV_SUCCESS = 0, //!< The operation was successful
} ibv_return_t;

commResult_t wrap_ibv_symbols(void);
/* NCCL wrappers of IB verbs functions */
commResult_t wrap_ibv_fork_init(void);
commResult_t wrap_ibv_get_device_list(
    ibverbx::ibv_device*** ret,
    int* num_devices);
commResult_t wrap_ibv_free_device_list(ibverbx::ibv_device** list);
const char* wrap_ibv_get_device_name(ibverbx::ibv_device* device);
commResult_t wrap_ibv_open_device(
    ibverbx::ibv_context** ret,
    ibverbx::ibv_device* device);
commResult_t wrap_ibv_close_device(ibverbx::ibv_context* context);
commResult_t wrap_ibv_get_async_event(
    ibverbx::ibv_context* context,
    ibverbx::ibv_async_event* event);
commResult_t wrap_ibv_ack_async_event(ibverbx::ibv_async_event* event);
commResult_t wrap_ibv_query_device(
    ibverbx::ibv_context* context,
    ibverbx::ibv_device_attr* device_attr);
commResult_t wrap_ibv_query_port(
    ibverbx::ibv_context* context,
    uint8_t port_num,
    ibverbx::ibv_port_attr* port_attr);
commResult_t wrap_ibv_query_gid(
    ibverbx::ibv_context* context,
    uint8_t port_num,
    int index,
    union ibverbx::ibv_gid* gid);
commResult_t wrap_ibv_query_qp(
    ibverbx::ibv_qp* qp,
    ibverbx::ibv_qp_attr* attr,
    int attr_mask,
    ibverbx::ibv_qp_init_attr* init_attr);
commResult_t wrap_ibv_alloc_pd(
    ibverbx::ibv_pd** ret,
    ibverbx::ibv_context* context);
commResult_t wrap_ibv_dealloc_pd(ibverbx::ibv_pd* pd);
commResult_t wrap_ibv_reg_mr(
    ibverbx::ibv_mr** ret,
    ibverbx::ibv_pd* pd,
    void* addr,
    size_t length,
    int access);
ibverbx::ibv_mr* wrap_direct_ibv_reg_mr(
    ibverbx::ibv_pd* pd,
    void* addr,
    size_t length,
    int access);
commResult_t wrap_ibv_reg_mr_iova2(
    ibverbx::ibv_mr** ret,
    ibverbx::ibv_pd* pd,
    void* addr,
    size_t length,
    uint64_t iova,
    int access);
/* DMA-BUF support */
commResult_t wrap_ibv_reg_dmabuf_mr(
    ibverbx::ibv_mr** ret,
    ibverbx::ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access);
ibverbx::ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(
    ibverbx::ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access);
commResult_t wrap_ibv_dereg_mr(ibverbx::ibv_mr* mr);
commResult_t wrap_ibv_create_comp_channel(
    ibverbx::ibv_comp_channel** ret,
    ibverbx::ibv_context* context);
commResult_t wrap_ibv_destroy_comp_channel(ibverbx::ibv_comp_channel* channel);
commResult_t wrap_ibv_create_cq(
    ibverbx::ibv_cq** ret,
    ibverbx::ibv_context* context,
    int cqe,
    void* cq_context,
    ibverbx::ibv_comp_channel* channel,
    int comp_vector);
commResult_t wrap_ibv_destroy_cq(ibverbx::ibv_cq* cq);
static inline commResult_t wrap_ibv_poll_cq(
    ibverbx::ibv_cq* cq,
    int num_entries,
    ibverbx::ibv_wc* wc,
    int* num_done) {
  int done = cq->context->ops.poll_cq(
      cq, num_entries, wc); /*returns the number of wcs or 0 on success, a
                               negative number otherwise*/
  if (done < 0) {
    CLOGF(WARN, "Call to ibv_poll_cq() returned {}", done);
    return commSystemError;
  }
  *num_done = done;
  return commSuccess;
}
commResult_t wrap_ibv_create_qp(
    ibverbx::ibv_qp** ret,
    ibverbx::ibv_pd* pd,
    ibverbx::ibv_qp_init_attr* qp_init_attr);
commResult_t wrap_ibv_modify_qp(
    ibverbx::ibv_qp* qp,
    ibverbx::ibv_qp_attr* attr,
    int attr_mask);
commResult_t wrap_ibv_destroy_qp(ibverbx::ibv_qp* qp);
commResult_t
wrap_ibv_query_ece(ibverbx::ibv_qp* qp, ibverbx::ibv_ece* ece, int* supported);
commResult_t
wrap_ibv_set_ece(ibverbx::ibv_qp* qp, ibverbx::ibv_ece* ece, int* supported);

static inline commResult_t wrap_ibv_post_send(
    ibverbx::ibv_qp* qp,
    ibverbx::ibv_send_wr* wr,
    ibverbx::ibv_send_wr** bad_wr) {
  int ret = qp->context->ops.post_send(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    CLOGF(
        WARN,
        "ibv_post_send() failed with error {}, Bad WR {}, First WR {}",
        strerror(ret),
        (void*)wr,
        (void*)*bad_wr);
    return commSystemError;
  }
  return commSuccess;
}

static inline commResult_t wrap_ibv_post_recv(
    ibverbx::ibv_qp* qp,
    ibverbx::ibv_recv_wr* wr,
    ibverbx::ibv_recv_wr** bad_wr) {
  int ret = qp->context->ops.post_recv(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    CLOGF(WARN, "ibv_post_recv() failed with error {}", strerror(ret));
    return commSystemError;
  }
  return commSuccess;
}

commResult_t wrap_ibv_event_type_str(
    char** ret,
    enum ibverbx::ibv_event_type event);

// converts a GID into a readable string. On success, returns a non-null pointer
// to gidStr. NULL is returned if there was an error, with errno set to indicate
// the error. errno = ENOSPC if the converted string would exceed strLen.
static inline const char*
ibvGetGidStr(union ibverbx::ibv_gid* gid, char* gidStr, size_t strLen) {
  // GID is a 16B handle, to convert it to a readable form, we use inet_ntop
  // sizeof(ibv_gid) == sizeof(struct in6_addr), so using AF_INET6
  static_assert(
      sizeof(union ibverbx::ibv_gid) == sizeof(struct in6_addr),
      "the sizeof struct ibv_gid must be the size of struct in6_addr");
  return inet_ntop(AF_INET6, gid->raw, gidStr, strLen);
}
commResult_t wrap_ibv_is_fork_initialized(enum ibverbx::ibv_fork_status* ret);
} // namespace ctran::ibvwrap
