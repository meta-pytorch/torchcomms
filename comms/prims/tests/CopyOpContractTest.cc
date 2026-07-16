// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

namespace comms::prims {

// The CopyOp fixed-size wire-bytes contract is enforced entirely at compile
// time by the static_asserts in CopyOpContractTest.cu (compiled via the
// :copy_op_contract_test_kernels dependency). Building this target IS the
// check: a policy whose send()/recv() does not return std::size_t fails to
// compile. This runtime body only makes the guard a discoverable test target.
TEST(CopyOpContractTest, FixedSizePoliciesReturnWireBytes) {
  SUCCEED();
}

} // namespace comms::prims
