// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/CtranIbLocalVc.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbQpUtils.h"
#include "comms/ctran/backends/ib/IbvWrap.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran::ibvwrap;

// extern ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t
// portNum, int gidTblLen, int *gidIndex);

namespace ctran::ib {

LocalVirtualConn::LocalVirtualConn(
    std::vector<CtranIbDevice>& devices,
    CommLogData commLogData)
    : devices_(devices), commLogData_(std::move(commLogData)) {
  FB_CHECKABORT(
      devices_.size() == NCCL_CTRAN_IB_DEVICES_PER_RANK,
      "Invalid number of devices {} received in flush virtual connection compared to NCCL_CTRAN_IB_DEVICES_PER_RANK {}",
      devices_.size(),
      NCCL_CTRAN_IB_DEVICES_PER_RANK);
  ibvMrs_.reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  ibvQps_.reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  sgs_.resize(NCCL_CTRAN_IB_DEVICES_PER_RANK);

  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    auto maybeMr = devices_[device].ibvPd->regMr(
        &buf_, sizeof(int), ibverbx::IBV_ACCESS_LOCAL_WRITE);
    FOLLY_EXPECTED_CHECKTHROW_EX(maybeMr, commLogData_);
    ibvMrs_.emplace_back(std::move(*maybeMr));

    memset(&sgs_[device], 0, sizeof(sgs_[device]));
    sgs_[device].addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&buf_));
    sgs_[device].length = sizeof(int);
    sgs_[device].lkey = ibvMrs_[device].mr()->lkey;

    // Create QP connecting to local device
    ibverbx::ibv_port_attr portAttr;
    auto maybePortAttr =
        devices[device].ibvDevice->queryPort(devices[device].port);
    FOLLY_EXPECTED_CHECKTHROW_EX(maybePortAttr, commLogData_);
    portAttr = std::move(*maybePortAttr);

    CtranIbRemoteQpInfo remoteQpInfo = {
        .mtu = portAttr.active_mtu,
        .port = devices[device].port,
        .linkLayer = portAttr.link_layer,
    };

    if (portAttr.link_layer == ibverbx::IBV_LINK_LAYER_ETHERNET) {
      union ibverbx::ibv_gid gid;

      auto maybeGid = devices[device].ibvDevice->queryGid(
          devices[device].port, static_cast<int>(NCCL_IB_GID_INDEX));
      FOLLY_EXPECTED_CHECKTHROW_EX(maybeGid, commLogData_);
      gid = std::move(*maybeGid);
      remoteQpInfo.u.eth.spn = gid.global.subnet_prefix;
      remoteQpInfo.u.eth.iid = gid.global.interface_id;
    } else {
      remoteQpInfo.u.ib.lid = portAttr.lid;
    }

    auto ibvQpCreateResult =
        ctranIbQpCreate(devices_[device].ibvPd, devices_[device].ibvCq->cq());
    FOLLY_EXPECTED_CHECKTHROW_EX(ibvQpCreateResult, commLogData_);
    ibvQps_.emplace_back(std::move(*ibvQpCreateResult));
    FOLLY_EXPECTED_CHECKTHROW_EX(
        ctranIbQpInit(
            ibvQps_[device],
            devices_[device].port,
            ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ),
        commLogData_);

    remoteQpInfo.qpn = ibvQps_[device].qp()->qp_num;
    FOLLY_EXPECTED_CHECKTHROW_EX(
        ctranIbQpRTR(remoteQpInfo, ibvQps_[device], NCCL_IB_TC), commLogData_);

    FOLLY_EXPECTED_CHECKTHROW_EX(ctranIbQpRTS(ibvQps_[device]), commLogData_);

    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: Established connection: commHash {:x}, commDesc {}, flush qpn {} on port {}",
        commLogData_.commHash,
        commLogData_.commDesc,
        ibvQps_[device].qp()->qp_num,
        devices_[device].port);
  }
}

commResult_t LocalVirtualConn::iflush(
    const void* dbuf,
    const void* ibRegElem,
    CtranIbRequest* req) {
  auto mrs = reinterpret_cast<const std::vector<ibverbx::IbvMr>*>(ibRegElem);
  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    ibverbx::ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dbuf);
    wr.wr.rdma.rkey = (*mrs)[device].mr()->rkey;
    wr.sg_list = &sgs_[device];
    wr.num_sge = 1;
    wr.opcode = ibverbx::IBV_WR_RDMA_READ;
    wr.send_flags = ibverbx::IBV_SEND_SIGNALED;

    ibverbx::ibv_send_wr bad_wr{};
    auto maybeSend = ibvQps_[device].postSend(&wr, &bad_wr);
    FOLLY_EXPECTED_CHECK(maybeSend);
    CLOGF_SUBSYS(
        DBG,
        COLL,
        "CTRAN-IB: posted flush on qpn {}, req {}",
        ibvQps_[device].qp()->qp_num,
        (void*)req);
  }

  outstandingReqs_.push_back(req);

  return commSuccess;
}

commResult_t LocalVirtualConn::processCqe(
    const enum ibverbx::ibv_wc_opcode opcode) {
  FB_CHECKABORT(
      opcode == ibverbx::IBV_WC_RDMA_READ,
      "Invalid opcode {} received in flush virtual connection",
      opcode);

  // Since each flush is executed by network one by one, we complete each flush
  // as simple FIFO.
  auto req = outstandingReqs_.front();
  outstandingReqs_.pop_front();
  FB_COMMCHECK(req->complete());

  return commSuccess;
}

uint32_t LocalVirtualConn::qpNum(int device) const {
  return ibvQps_.at(device).qp()->qp_num;
}

} // namespace ctran::ib
