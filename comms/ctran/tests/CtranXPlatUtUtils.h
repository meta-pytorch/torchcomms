// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include "comms/ctran/CtranComm.h"
#include "comms/mccl/McclComm.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/tests_common.cuh"
#include "comms/utils/commSpecs.h"

void logGpuMemoryStats(int gpu);

void commSetMyThreadLoggingName(std::string_view name);

commResult_t commMemAllocDisjoint(
    void** ptr,
    std::vector<size_t>& disjointSegmentSizes,
    std::vector<TestMemSegment>& segments,
    bool setRdmaSupport = true,
    std::optional<CUmemAllocationHandleType> handleType =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes);

void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments);
void commMemFree(void* buf, size_t bufSize, MemAllocType memType);

class TestCtranCommRAII {
 public:
  // TODO: construct without mcclComm
  TestCtranCommRAII(std::unique_ptr<mccl::McclComm> mcclComm);
  CtranComm* ctranComm{nullptr};

 private:
  std::unique_ptr<mccl::McclComm> mcclComm_;
};

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm();
