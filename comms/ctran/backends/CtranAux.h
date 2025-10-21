// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef ENABLE_META_COMPRESSION
#pragma once

// This file suppliments iSendCtrl/iRecvCtrl with auxiliary data to be
// communicated between IB peers.

using DefaultAuxType = uint64_t;

template <typename T = DefaultAuxType>
struct AuxData_t {
  T data;

  AuxData_t() {
    data = static_cast<DefaultAuxType>(0);
  }

  AuxData_t(T val) {
    data = val;
  }

  AuxData_t(const AuxData_t& other) {
    data = other.data;
  }

  AuxData_t& operator=(const AuxData_t& other) {
    if (this != &other) {
      data = other.data;
    }
    return *this;
  }
};
#endif // ENABLE_META_COMPRESSION
