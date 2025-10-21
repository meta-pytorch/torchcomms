// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommTypes.hpp"

namespace torch {
namespace comms {

const ReduceOp ReduceOp::SUM = ReduceOp(ReduceOp::RedOpType::SUM);
const ReduceOp ReduceOp::PRODUCT = ReduceOp(ReduceOp::RedOpType::PRODUCT);
const ReduceOp ReduceOp::MIN = ReduceOp(ReduceOp::RedOpType::MIN);
const ReduceOp ReduceOp::MAX = ReduceOp(ReduceOp::RedOpType::MAX);
const ReduceOp ReduceOp::BAND = ReduceOp(ReduceOp::RedOpType::BAND);
const ReduceOp ReduceOp::BOR = ReduceOp(ReduceOp::RedOpType::BOR);
const ReduceOp ReduceOp::BXOR = ReduceOp(ReduceOp::RedOpType::BXOR);
const ReduceOp ReduceOp::AVG = ReduceOp(ReduceOp::RedOpType::AVG);

} // namespace comms
} // namespace torch
