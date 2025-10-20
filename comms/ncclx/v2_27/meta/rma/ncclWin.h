// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/window/CtranWin.h"

struct ncclWin {
  // communicator associated with this window
  ncclComm_t comm;

  // implementation of ncclWin on top of Ctran
  ctran::CtranWin* ctranWindow;
};
