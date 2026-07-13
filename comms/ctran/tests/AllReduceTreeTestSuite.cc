// Copyright (c) Meta Platforms, Inc. and affiliates.

// AllReduce tree (ctree) correctness leaf on the shared
// CollectiveTestSuite<AllReduceTraits> harness. All parameterization (the
// consolidated size sweep, byte-offset sweep, 512 MiB large payload, and the
// NCCL_CTRAN_MAX_NBLOCKS block-cap sweep) plus main() come from the shared
// REGISTER_COLLECTIVE_TESTS / COLLECTIVE_TEST_MAIN macros in
// CollectiveTestSuite.h; this leaf only binds the tree algo dispatch and its
// topology-specific env.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <optional>

#include "comms/ctran/tests/AllReduceTestSuiteBase.h"

class AllReduceTreeTestSuite : public AllReduceTestSuiteBase {
 protected:
  commResult_t run(
      enum NCCL_ALLREDUCE_ALGO algo,
      const void* send,
      void* recv,
      size_t count,
      commDataType_t datatype,
      commRedOp_t op,
      CtranComm* comm,
      cudaStream_t stream,
      std::optional<std::chrono::milliseconds> timeout) override {
    (void)algo;
    return ctranAllReduceTree(
        send, recv, count, datatype, op, comm, stream, timeout);
  }

  bool algo_supports(commDataType_t /*dt*/, commRedOp_t op) const override {
    return op == commSum;
  }

  ctran::CtranEnvs envOverrides() const override {
    auto envs = AllReduceTestSuiteBase::envOverrides();
    envs.emplace_back("NCCL_ALLREDUCE_ALGO", "ctree");
    envs.emplace_back("NCCL_CTRAN_IBGDA_SENDRECV_ENABLE", "1");
    envs.emplace_back("NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE", "33554432");
    envs.emplace_back("NCCL_CTRAN_IB_MAX_GROUPS", "16");
#if defined(CTRAN_ALLREDUCE_TEST_IB_ONLY)
    envs.emplace_back("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal");
    envs.emplace_back("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1");
    envs.emplace_back("NCCL_P2P_DISABLE", "1");
#endif
    return envs;
  }
};

#if defined(CTRAN_ALLREDUCE_TEST_NVL_ONLY) || \
    defined(CTRAN_ALLREDUCE_TEST_IB_ONLY) ||  \
    defined(CTRAN_ALLREDUCE_TEST_HYBRID)
REGISTER_COLLECTIVE_TESTS(
    AllReduceTreeTestSuite,
    AllReduceTraits,
    NCCL_ALLREDUCE_ALGO::ctree,
    allReduceTreeDataTypes)
#else
#error "Define one CTRAN AllReduce topology: NVL_ONLY, IB_ONLY, or HYBRID"
#endif

COLLECTIVE_TEST_MAIN()
