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
  // used for the stochastic rounding. Intentionally using optional to ensure
  // that we differentiate between the case where we have not set the attribute
  // and the case where we set the attribute to 0.
  std::optional<uint64_t> randomSeed{std::nullopt};

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
