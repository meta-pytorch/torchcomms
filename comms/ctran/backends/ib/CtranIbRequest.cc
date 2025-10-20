// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/CtranIbBase.h"

void CtranIbRequest::setRefCount(int refCount) {
  this->refCount_ = refCount;
}

void CtranIbRequest::repost(int refCount) {
  this->refCount_ = refCount;
  this->state_ = CtranIbRequest::INCOMPLETE;
}
