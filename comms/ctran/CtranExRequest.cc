// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranEx.h"
#include "comms/ctran/CtranExImpl.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"

namespace ctran {

CtranExRequest::CtranExRequest() {
  impl_ = reinterpret_cast<void*>(new CtranExRequestImpl());
}

CtranExRequest::~CtranExRequest() {
  delete reinterpret_cast<CtranExRequestImpl*>(impl_);
}

static std::unordered_map<CtranExRequestImpl::Type, const std::string> typeStrs{
    {CtranExRequestImpl::PUT, "PUT"},
    {CtranExRequestImpl::SEND_CTRL, "SEND_CTRL"},
    {CtranExRequestImpl::RECV_CTRL, "RECV_CTRL"},
    {CtranExRequestImpl::SEND_SYNC_CTRL, "SEND_SYNC_CTRL"},
    {CtranExRequestImpl::RECV_SYNC_CTRL, "RECV_SYNC_CTRL"},
    {CtranExRequestImpl::FLUSH, "FLUSH"},
};

void CtranExRequestImpl::initialize(Type type, CtranIb* ctranIb) {
  switch (type) {
    case SEND_CTRL:
      sendCtrl.msg = ControlMsg();
      break;
    case RECV_CTRL:
      recvCtrl.msg = ControlMsg();
      recvCtrl.rBuf = nullptr;
      recvCtrl.rKey = nullptr;
      break;
    case SEND_SYNC_CTRL:
      sendSyncCtrl.msg = ControlMsg();
      break;
    case RECV_SYNC_CTRL:
      recvSyncCtrl.msg = ControlMsg();
      break;
    case PUT:
    case FLUSH:
      break;
    default:
      FB_CHECKABORT(
          false,
          "Invalid CtranExRequest type {} associated with CtranIb instance {}. It is likely a Ctran bug.",
          type,
          (void*)ctranIb);
      break;
  }
  this->type = type;
  this->ctranIb = ctranIb;
}

void CtranExRequestImpl::initialize(Type type, CtranComm* ctranComm) {
  switch (type) {
    case BCAST:
      bcast.complete.clear();
      break;
    default:
      FB_CHECKABORT(
          false,
          "Invalid CtranExRequest type {} associated with CtranComm instance {}. It is likely a Ctran bug.",
          type,
          (void*)ctranComm);
      break;
  }
  this->type = type;
  this->asyncErr = ctranComm->getAsyncError();
}

void CtranExRequestImpl::complete() {
  switch (type) {
    case BCAST:
      bcast.complete.test_and_set();
      break;
    // no op for other types
    default:
      break;
  }
}

void CtranExRequestImpl::atComplete(CtranExRequest* req) {
  switch (type) {
    case RECV_CTRL:
      FB_CHECKABORT(
          recvCtrl.rBuf && recvCtrl.rKey,
          "Invalid nullptr rBuf or rKey in CtranExRequest object, it is likely a NCCL bug.");

      CLOGF_TRACE(
          COLL,
          "CtranExRequest {} received ctrlMsg: {}, to copy to rBuf {}, rKey {}",
          (void*)req,
          recvCtrl.msg.toString(),
          (void*)recvCtrl.rBuf,
          (void*)recvCtrl.rKey);

      // copy received buffer info to user specified pointers
      if (recvCtrl.msg.type == ControlMsgType::IB_EXPORT_MEM) {
        *(recvCtrl.rBuf) =
            reinterpret_cast<void*>(recvCtrl.msg.ibExp.remoteAddr);
        *(recvCtrl.rKey) = recvCtrl.msg.ibExp.rkeys[0];
      }
      break;
    default:
      break;
  }
}

bool CtranExRequest::isComplete() const {
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(impl_);
  if (reqImpl->type == CtranExRequestImpl::BCAST) {
    return reqImpl->bcast.complete.test();
  }
  return reqImpl->ibReq.isComplete();
}

commResult_t CtranExRequest::test(bool& complete) {
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(impl_);

  // GPE thread asynchronously handles collective communication request and
  // marks completion. Thus, no need to polling backend progress by the calling
  // thread.
  if (reqImpl->type == CtranExRequestImpl::BCAST) {
    complete = reqImpl->bcast.complete.test();

    // Check if there is any error reported by the GPE thread;
    // if so, return the error code.
    if (!complete && reqImpl->asyncErr) {
      FB_COMMCHECK(reqImpl->asyncErr->getAsyncResult());
    }
    return commSuccess;
  }

  FB_CHECKABORT(
      reqImpl->ctranIb != nullptr,
      "Invalid nullptr ctranIb in CtranExRequest object, it is a NCCLX bug.")

  CtranIbEpochRAII epoch(reqImpl->ctranIb);

  // Poll IB backend progress once
  FB_COMMCHECK(reqImpl->ctranIb->progress());

  // Check if request is complete
  complete = reqImpl->ibReq.isComplete();
  if (complete) {
    CLOGF_TRACE(
        COLL,
        "CtranExRequest {} type {} completed",
        (void*)this,
        typeStrs.at(reqImpl->type));
    reqImpl->atComplete(this);
  }
  return commSuccess;
}

commResult_t CtranExRequest::wait() {
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(impl_);

  if (reqImpl->type == CtranExRequestImpl::BCAST) {
    // GPE thread is handling the communcation, wait for it to complete
    reqImpl->bcast.complete.wait(true);
    // Check if there is any error reported by the GPE thread;
    // if so, return the error code.
    if (reqImpl->asyncErr) {
      FB_COMMCHECK(reqImpl->asyncErr->getAsyncResult());
    }
    return commSuccess;
  }

  FB_CHECKABORT(
      reqImpl->ctranIb != nullptr,
      "Invalid nullptr ctranIb in CtranExRequest object, it is a NCCLX bug.")

  CtranIbEpochRAII epoch(reqImpl->ctranIb);

  // Wait for request to complete; polling IB backend progress during wait.
  while (!reqImpl->ibReq.isComplete()) {
    FB_COMMCHECK(reqImpl->ctranIb->progress());
  }
  CLOGF_TRACE(
      COLL,
      "CtranExRequest {} type {} completed",
      (void*)this,
      typeStrs.at(reqImpl->type));
  reqImpl->atComplete(this);

  return commSuccess;
}

} // namespace ctran
