/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include <optional>

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv,
  ncclPatternProfiler,
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;

  /*
   * Start of NCCLX Specific attributes
   */
  // Stochastic Rounding reduction ops only attribute. The random seed being
  // used for the stochastic rounding. This will point to the GPU memory holding
  // the random seed.
  uint64_t* randomSeed{nullptr};
  // The type we should use for transport. This feild will only be set for
  // quantized collectives. For non-quantized collectives, this will be nullopt
  std::optional<ncclDataType_t> transportType{std::nullopt};

  /*
   * NCCLX Specific attributes (Not really being used, no idea what they are for)
   */
  int nThreads{0};
  int nChannels{0};
  int algorithm{0};
  int protocol{0};
  bool userTuned{0};
  int stepSize{0};
  int chunkCount{0};
  int chunkSize{0};
  int channelId{0};
  ncclPattern_t pattern;
};

#endif
