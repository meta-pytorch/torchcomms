// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "meta/algoconf/AlgoStrConv.h"

using ncclx::algoconf::algoStrToVal;
using ncclx::algoconf::algoValToStr;

void checkAlgoStrToVal(enum NCCL_SENDRECV_ALGO algo) {
  std::string str = "UNHANDLED_ENUM_VALUE";
  switch (algo) {
    case NCCL_SENDRECV_ALGO::orig:
      str = "orig";
      break;
    case NCCL_SENDRECV_ALGO::ctran:
      str = "ctran";
      break;
    case NCCL_SENDRECV_ALGO::ctzcopy:
      str = "ctzcopy";
      break;
    case NCCL_SENDRECV_ALGO::ctp2p:
      str = "ctp2p";
      break;
    case NCCL_SENDRECV_ALGO::ctgraph:
      str = "ctgraph";
      break;
  }
  enum NCCL_SENDRECV_ALGO result;
  algoStrToVal(str, result);
  EXPECT_EQ(result, algo) << "algoStrToVal(\"" << str
                          << "\") returned wrong value";
  EXPECT_EQ(algoValToStr(algo), str) << "algoValToStr round-trip failed";
}

void checkAlgoStrToVal(enum NCCL_ALLGATHER_ALGO algo) {
  std::string str = "UNHANDLED_ENUM_VALUE";
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::orig:
      str = "orig";
      break;
    case NCCL_ALLGATHER_ALGO::ctran:
      str = "ctran";
      break;
    case NCCL_ALLGATHER_ALGO::ctdirect:
      str = "ctdirect";
      break;
    case NCCL_ALLGATHER_ALGO::ctring:
      str = "ctring";
      break;
    case NCCL_ALLGATHER_ALGO::ctsrd:
      str = "ctsrd";
      break;
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      str = "ctbrucks";
      break;
    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
      str = "cthierarchical_ring";
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph:
      str = "ctgraph";
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
      str = "ctgraph_pipeline";
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline:
      str = "ctgraph_rdpipeline";
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
      str = "ctgraph_ring";
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      str = "ctgraph_rd";
      break;
  }
  enum NCCL_ALLGATHER_ALGO result;
  algoStrToVal(str, result);
  EXPECT_EQ(result, algo) << "algoStrToVal(\"" << str
                          << "\") returned wrong value";
  EXPECT_EQ(algoValToStr(algo), str) << "algoValToStr round-trip failed";
}

void checkAlgoStrToVal(enum NCCL_ALLREDUCE_ALGO algo) {
  std::string str = "UNHANDLED_ENUM_VALUE";
  switch (algo) {
    case NCCL_ALLREDUCE_ALGO::orig:
      str = "orig";
      break;
    case NCCL_ALLREDUCE_ALGO::ctran:
      str = "ctran";
      break;
    case NCCL_ALLREDUCE_ALGO::ctdirect:
      str = "ctdirect";
      break;
    case NCCL_ALLREDUCE_ALGO::ctring:
      str = "ctring";
      break;
    case NCCL_ALLREDUCE_ALGO::ctree:
      str = "ctree";
      break;
    case NCCL_ALLREDUCE_ALGO::cthierarchical_ring:
      str = "cthierarchical_ring";
      break;
  }
  enum NCCL_ALLREDUCE_ALGO result;
  algoStrToVal(str, result);
  EXPECT_EQ(result, algo) << "algoStrToVal(\"" << str
                          << "\") returned wrong value";
  EXPECT_EQ(algoValToStr(algo), str) << "algoValToStr round-trip failed";
}

void checkAlgoStrToVal(enum NCCL_ALLTOALLV_ALGO algo) {
  std::string str = "UNHANDLED_ENUM_VALUE";
  switch (algo) {
    case NCCL_ALLTOALLV_ALGO::orig:
      str = "orig";
      break;
    case NCCL_ALLTOALLV_ALGO::ctran:
      str = "ctran";
      break;
    case NCCL_ALLTOALLV_ALGO::compCtran:
      str = "compCtran";
      break;
    case NCCL_ALLTOALLV_ALGO::bsCompCtran:
      str = "bsCompCtran";
      break;
  }
  enum NCCL_ALLTOALLV_ALGO result;
  algoStrToVal(str, result);
  EXPECT_EQ(result, algo) << "algoStrToVal(\"" << str
                          << "\") returned wrong value";
  EXPECT_EQ(algoValToStr(algo), str) << "algoValToStr round-trip failed";
}

void checkAlgoStrToVal(enum NCCL_RMA_ALGO algo) {
  std::string str = "UNHANDLED_ENUM_VALUE";
  switch (algo) {
    case NCCL_RMA_ALGO::orig:
      str = "orig";
      break;
    case NCCL_RMA_ALGO::ctran:
      str = "ctran";
      break;
  }
  enum NCCL_RMA_ALGO result;
  algoStrToVal(str, result);
  EXPECT_EQ(result, algo) << "algoStrToVal(\"" << str
                          << "\") returned wrong value";
  EXPECT_EQ(algoValToStr(algo), str) << "algoValToStr round-trip failed";
}

TEST(AlgoStrConvTest, SendRecvCompleteness) {
  checkAlgoStrToVal(NCCL_SENDRECV_ALGO::orig);
  checkAlgoStrToVal(NCCL_SENDRECV_ALGO::ctran);
  checkAlgoStrToVal(NCCL_SENDRECV_ALGO::ctzcopy);
  checkAlgoStrToVal(NCCL_SENDRECV_ALGO::ctp2p);
  checkAlgoStrToVal(NCCL_SENDRECV_ALGO::ctgraph);
}

TEST(AlgoStrConvTest, AllGatherCompleteness) {
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::orig);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctran);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctdirect);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctring);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctsrd);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctbrucks);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::cthierarchical_ring);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctgraph);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctgraph_pipeline);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctgraph_ring);
  checkAlgoStrToVal(NCCL_ALLGATHER_ALGO::ctgraph_rd);
}

TEST(AlgoStrConvTest, AllReduceCompleteness) {
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::orig);
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::ctran);
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::ctdirect);
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::ctring);
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::ctree);
  checkAlgoStrToVal(NCCL_ALLREDUCE_ALGO::cthierarchical_ring);
}

TEST(AlgoStrConvTest, AllToAllVCompleteness) {
  checkAlgoStrToVal(NCCL_ALLTOALLV_ALGO::orig);
  checkAlgoStrToVal(NCCL_ALLTOALLV_ALGO::ctran);
  checkAlgoStrToVal(NCCL_ALLTOALLV_ALGO::compCtran);
  checkAlgoStrToVal(NCCL_ALLTOALLV_ALGO::bsCompCtran);
}

TEST(AlgoStrConvTest, RmaCompleteness) {
  checkAlgoStrToVal(NCCL_RMA_ALGO::orig);
  checkAlgoStrToVal(NCCL_RMA_ALGO::ctran);
}

TEST(AlgoStrConvTest, UnknownStringFallback) {
  enum NCCL_SENDRECV_ALGO sendrecv;
  algoStrToVal("nonexistent", sendrecv);
  EXPECT_EQ(sendrecv, NCCL_SENDRECV_ALGO::orig);

  enum NCCL_ALLGATHER_ALGO allgather;
  algoStrToVal("nonexistent", allgather);
  EXPECT_EQ(allgather, NCCL_ALLGATHER_ALGO::orig);

  enum NCCL_ALLREDUCE_ALGO allreduce;
  algoStrToVal("nonexistent", allreduce);
  EXPECT_EQ(allreduce, NCCL_ALLREDUCE_ALGO::orig);

  enum NCCL_ALLTOALLV_ALGO alltoallv;
  algoStrToVal("nonexistent", alltoallv);
  EXPECT_EQ(alltoallv, NCCL_ALLTOALLV_ALGO::orig);

  enum NCCL_RMA_ALGO rma;
  algoStrToVal("nonexistent", rma);
  EXPECT_EQ(rma, NCCL_RMA_ALGO::orig);
}
