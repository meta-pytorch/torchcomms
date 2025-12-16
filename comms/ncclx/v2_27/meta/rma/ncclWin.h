// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/window/CtranWin.h"

struct ncclWindow;

struct ncclWin {
  // communicator associated with this window
  ncclComm_t comm;

  // implementation of ncclWin on top of Ctran
  ctran::CtranWin* ctranWindow;
};

// Global map: ncclWindow_t -> ncclWin* (Meyer's Singleton - single instance
// guaranteed)
inline std::unordered_map<ncclWindow*, ncclWin*>& windowMap() {
  static std::unordered_map<ncclWindow*, ncclWin*> instance;
  return instance;
}
