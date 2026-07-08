// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>

#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::algoconf {

inline void algoStrToVal(const std::string& str, enum NCCL_SENDRECV_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_SENDRECV_ALGO::ctran;
  } else if (str == "ctzcopy") {
    val = NCCL_SENDRECV_ALGO::ctzcopy;
  } else if (str == "ctp2p") {
    val = NCCL_SENDRECV_ALGO::ctp2p;
  } else if (str == "ctgraph") {
    val = NCCL_SENDRECV_ALGO::ctgraph;
  } else {
    val = NCCL_SENDRECV_ALGO::orig;
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLGATHER_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLGATHER_ALGO::ctran;
  } else if (str == "ctdirect") {
    val = NCCL_ALLGATHER_ALGO::ctdirect;
  } else if (str == "ctring") {
    val = NCCL_ALLGATHER_ALGO::ctring;
  } else if (str == "ctsrd") {
    val = NCCL_ALLGATHER_ALGO::ctsrd;
  } else if (str == "ctbrucks") {
    val = NCCL_ALLGATHER_ALGO::ctbrucks;
  } else if (str == "cthierarchical_ring") {
    val = NCCL_ALLGATHER_ALGO::cthierarchical_ring;
  } else if (str == "ctgraph") {
    val = NCCL_ALLGATHER_ALGO::ctgraph;
  } else if (str == "ctgraph_pipeline") {
    val = NCCL_ALLGATHER_ALGO::ctgraph_pipeline;
  } else if (str == "ctgraph_rdpipeline") {
    val = NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline;
  } else if (str == "ctgraph_ring") {
    val = NCCL_ALLGATHER_ALGO::ctgraph_ring;
  } else if (str == "ctgraph_rd") {
    val = NCCL_ALLGATHER_ALGO::ctgraph_rd;
  } else {
    val = NCCL_ALLGATHER_ALGO::orig;
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLREDUCE_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLREDUCE_ALGO::ctran;
  } else if (str == "ctdirect") {
    val = NCCL_ALLREDUCE_ALGO::ctdirect;
  } else if (str == "ctring") {
    val = NCCL_ALLREDUCE_ALGO::ctring;
  } else if (str == "ctree") {
    val = NCCL_ALLREDUCE_ALGO::ctree;
  } else if (str == "cthierarchical_ring") {
    val = NCCL_ALLREDUCE_ALGO::cthierarchical_ring;
  } else {
    val = NCCL_ALLREDUCE_ALGO::orig;
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLTOALLV_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLTOALLV_ALGO::ctran;
  } else if (str == "compCtran") {
    val = NCCL_ALLTOALLV_ALGO::compCtran;
  } else if (str == "bsCompCtran") {
    val = NCCL_ALLTOALLV_ALGO::bsCompCtran;
  } else {
    val = NCCL_ALLTOALLV_ALGO::orig;
  }
}

inline std::string algoValToStr(enum NCCL_SENDRECV_ALGO val) {
  switch (val) {
    case NCCL_SENDRECV_ALGO::orig:
      return "orig";
    case NCCL_SENDRECV_ALGO::ctran:
      return "ctran";
    case NCCL_SENDRECV_ALGO::ctzcopy:
      return "ctzcopy";
    case NCCL_SENDRECV_ALGO::ctp2p:
      return "ctp2p";
    case NCCL_SENDRECV_ALGO::ctgraph:
      return "ctgraph";
  }
  return "unknown";
}

inline std::string algoValToStr(enum NCCL_ALLGATHER_ALGO val) {
  switch (val) {
    case NCCL_ALLGATHER_ALGO::orig:
      return "orig";
    case NCCL_ALLGATHER_ALGO::ctran:
      return "ctran";
    case NCCL_ALLGATHER_ALGO::ctdirect:
      return "ctdirect";
    case NCCL_ALLGATHER_ALGO::ctring:
      return "ctring";
    case NCCL_ALLGATHER_ALGO::ctsrd:
      return "ctsrd";
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      return "ctbrucks";
    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
      return "cthierarchical_ring";
    case NCCL_ALLGATHER_ALGO::ctgraph:
      return "ctgraph";
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
      return "ctgraph_pipeline";
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline:
      return "ctgraph_rdpipeline";
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
      return "ctgraph_ring";
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      return "ctgraph_rd";
  }
  return "unknown";
}

inline std::string algoValToStr(enum NCCL_ALLREDUCE_ALGO val) {
  switch (val) {
    case NCCL_ALLREDUCE_ALGO::orig:
      return "orig";
    case NCCL_ALLREDUCE_ALGO::ctran:
      return "ctran";
    case NCCL_ALLREDUCE_ALGO::ctdirect:
      return "ctdirect";
    case NCCL_ALLREDUCE_ALGO::ctring:
      return "ctring";
    case NCCL_ALLREDUCE_ALGO::ctree:
      return "ctree";
    case NCCL_ALLREDUCE_ALGO::cthierarchical_ring:
      return "cthierarchical_ring";
  }
  return "unknown";
}

inline std::string algoValToStr(enum NCCL_ALLTOALLV_ALGO val) {
  switch (val) {
    case NCCL_ALLTOALLV_ALGO::orig:
      return "orig";
    case NCCL_ALLTOALLV_ALGO::ctran:
      return "ctran";
    case NCCL_ALLTOALLV_ALGO::compCtran:
      return "compCtran";
    case NCCL_ALLTOALLV_ALGO::bsCompCtran:
      return "bsCompCtran";
  }
  return "unknown";
}

} // namespace ncclx::algoconf
