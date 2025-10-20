// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/Conversion.h"

#include "comms/utils/commSpecs.h"

namespace meta::comms {

std::string_view commPatternToString(CommPattern pattern) {
  switch (pattern) {
    case CommPattern::Ring:
      return "Ring";
    case CommPattern::RingTwice:
      return "RingTwice";
    case CommPattern::PipelineFrom:
      return "PipelineFrom";
    case CommPattern::PipelineTo:
      return "PipelineTo";
    case CommPattern::TreeUp:
      return "TreeUp";
    case CommPattern::TreeDown:
      return "TreeDown";
    case CommPattern::TreeUpDown:
      return "TreeUpDown";
    case CommPattern::CollnetChain:
      return "CollnetChain";
    case CommPattern::CollnetDirect:
      return "CollnetDirect";
    case CommPattern::Nvls:
      return "Nvls";
    case CommPattern::NvlsTree:
      return "NvlsTree";
    case CommPattern::PatUp:
      return "PatUp";
    case CommPattern::PatDown:
      return "PatDown";
    case CommPattern::Send:
      return "Send";
    case CommPattern::Recv:
      return "Recv";
    default:
      return "Unknown";
  }
}

CommPattern stringToCommPattern(std::string_view str) {
  if (str == "Ring") {
    return CommPattern::Ring;
  }
  if (str == "RingTwice") {
    return CommPattern::RingTwice;
  }
  if (str == "PipelineFrom") {
    return CommPattern::PipelineFrom;
  }
  if (str == "PipelineTo") {
    return CommPattern::PipelineTo;
  }
  if (str == "TreeUp") {
    return CommPattern::TreeUp;
  }
  if (str == "TreeDown") {
    return CommPattern::TreeDown;
  }
  if (str == "TreeUpDown") {
    return CommPattern::TreeUpDown;
  }
  if (str == "CollnetChain") {
    return CommPattern::CollnetChain;
  }
  if (str == "CollnetDirect") {
    return CommPattern::CollnetDirect;
  }
  if (str == "Nvls") {
    return CommPattern::Nvls;
  }
  if (str == "NvlsTree") {
    return CommPattern::NvlsTree;
  }
  if (str == "PatUp") {
    return CommPattern::PatUp;
  }
  if (str == "PatDown") {
    return CommPattern::PatDown;
  }
  if (str == "Send") {
    return CommPattern::Send;
  }
  if (str == "Recv") {
    return CommPattern::Recv;
  }

  // Default value if not found
  return CommPattern::NumPatterns;
}

std::string_view commFuncToString(CommFunc func) {
  switch (func) {
    case CommFunc::Broadcast:
      return "Broadcast";
    case CommFunc::Reduce:
      return "Reduce";
    case CommFunc::AllGather:
      return "AllGather";
    case CommFunc::ReduceScatter:
      return "ReduceScatter";
    case CommFunc::AllReduce:
      return "AllReduce";
    case CommFunc::SendRecv:
      return "SendRecv";
    case CommFunc::Send:
      return "Send";
    case CommFunc::Recv:
      return "Recv";
    default:
      return "Unknown";
  }
}

CommFunc stringToCommFunc(std::string_view str) {
  if (str == "Broadcast") {
    return CommFunc::Broadcast;
  }
  if (str == "Reduce") {
    return CommFunc::Reduce;
  }
  if (str == "AllGather") {
    return CommFunc::AllGather;
  }
  if (str == "ReduceScatter") {
    return CommFunc::ReduceScatter;
  }
  if (str == "AllReduce") {
    return CommFunc::AllReduce;
  }
  if (str == "SendRecv") {
    return CommFunc::SendRecv;
  }
  if (str == "Send") {
    return CommFunc::Send;
  }
  if (str == "Recv") {
    return CommFunc::Recv;
  }

  // Default value if not found
  return CommFunc::NumFuncs;
}

std::string_view commRedOpToString(commRedOp_t op) {
  switch (op) {
    case commRedOp_t::commSum:
      return "Sum";
    case commRedOp_t::commProd:
      return "Prod";
    case commRedOp_t::commMax:
      return "Max";
    case commRedOp_t::commMin:
      return "Min";
    case commRedOp_t::commAvg:
      return "Avg";
    default:
      return "Unknown";
  }
}

commRedOp_t stringToCommRedOp(std::string_view str) {
  if (str == "Sum") {
    return commRedOp_t::commSum;
  }
  if (str == "Prod") {
    return commRedOp_t::commProd;
  }
  if (str == "Max") {
    return commRedOp_t::commMax;
  }
  if (str == "Min") {
    return commRedOp_t::commMin;
  }
  if (str == "Avg") {
    return commRedOp_t::commAvg;
  }

  // Default value if not found
  return commRedOp_t::commNumOps;
}

std::string_view commAlgoToString(CommAlgo algo) {
  switch (algo) {
    case CommAlgo::Tree:
      return "Tree";
    case CommAlgo::Ring:
      return "Ring";
    case CommAlgo::CollNetDirect:
      return "CollNetDirect";
    case CommAlgo::CollNetChain:
      return "CollNetChain";
    case CommAlgo::NVLS:
      return "NVLS";
    case CommAlgo::NVLSTree:
      return "NVLSTree";
    case CommAlgo::PAT:
      return "PAT";
    default:
      return "Unknown";
  }
}

CommAlgo stringToCommAlgo(std::string_view str) {
  if (str == "Tree") {
    return CommAlgo::Tree;
  }
  if (str == "Ring") {
    return CommAlgo::Ring;
  }
  if (str == "CollNetDirect") {
    return CommAlgo::CollNetDirect;
  }
  if (str == "CollNetChain") {
    return CommAlgo::CollNetChain;
  }
  if (str == "NVLS") {
    return CommAlgo::NVLS;
  }
  if (str == "NVLSTree") {
    return CommAlgo::NVLSTree;
  }
  if (str == "PAT") {
    return CommAlgo::PAT;
  }

  // Default value if not found
  return CommAlgo::NumAlgorithms;
}

std::string_view commProtocolToString(CommProtocol protocol) {
  switch (protocol) {
    case CommProtocol::LL:
      return "LL";
    case CommProtocol::LL128:
      return "LL128";
    case CommProtocol::Simple:
      return "Simple";
    default:
      return "Unknown";
  }
}

CommProtocol stringToCommProtocol(std::string_view str) {
  if (str == "LL") {
    return CommProtocol::LL;
  }
  if (str == "LL128") {
    return CommProtocol::LL128;
  }
  if (str == "Simple") {
    return CommProtocol::Simple;
  }

  // Default value if not found
  return CommProtocol::NumProtocols;
}

} // namespace meta::comms
