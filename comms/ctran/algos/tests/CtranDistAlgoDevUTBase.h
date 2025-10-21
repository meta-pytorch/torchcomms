// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <gtest/gtest.h>
#include <stdlib.h>

#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/commSpecs.h"

class CtranDistAlgoDevTest : public NcclxBaseTest {
 public:
  void SetUp() override;

  void TearDown() override;

  template <typename T>
  void assignVal(void* buf, size_t count, T seedVal, bool inc = false);

  template <typename T>
  void initIpcBufs(size_t srcCount, size_t dstCount);
  template <typename T>
  void initIpcBufs(size_t count) {
    return initIpcBufs<T>(count, count);
  }

  template <typename T>
  void checkVals(size_t count, T seedVal = 0, size_t offset = 0);

  void freeIpcBufs();

 protected:
  ncclComm_t comm_{nullptr};
  void* localBuf_{nullptr};
  void* ipcBuf_{nullptr};
  std::unique_ptr<ctran::utils::CtranIpcMem> ipcMem_{nullptr};
  std::vector<std::unique_ptr<ctran::utils::CtranIpcRemMem>> ipcRemMem_;

  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      "testComm",
      0,
      0};
};

// instantiate the template functions
#define DECLAR_ALGO_UT_FUNCS(T)                       \
  template void CtranDistAlgoDevTest::initIpcBufs<T>( \
      size_t srcCount, size_t dstCount);              \
  template void CtranDistAlgoDevTest::assignVal<T>(   \
      void* buf, size_t count, T seedVal, bool inc);  \
  template void CtranDistAlgoDevTest::checkVals<T>(   \
      size_t count, T seedVal, size_t offset);
