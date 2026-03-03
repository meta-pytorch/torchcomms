// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/utils/commSpecs.h"
#include "nccl.h"

// Convert commResult_t to ncclResult_t
ncclResult_t metaCommToNccl(commResult_t result);

// Convert ncclResult_t to commResult_t
commResult_t ncclToMetaComm(ncclResult_t result);

// Convert commRedOp_t to ncclRedOp_t
ncclRedOp_t metaCommToNccl(commRedOp_t op);

// Convert ncclRedOp_t to commRedOp_t
commRedOp_t ncclToMetaComm(ncclRedOp_t op);

// Convert ncclCmpOp_t to commCmpOp_t
commCmpOp_t ncclToMetaComm(ncclCmpOp_t op);

// Convert commDataType_t to ncclDataType_t
ncclDataType_t metaCommToNccl(commDataType_t datatype);

// Convert ncclDataType_t to commDataType_t
commDataType_t ncclToMetaComm(ncclDataType_t dataType);

// Convert ncclx::Hints to meta::comms::Hints
meta::comms::Hints ncclToMetaComm(const ncclx::Hints& datatype);

// Those are temorarly functions to initialized ctranComm from ncclComm
// TODO: remove this factory methods once we have proper CtranComm
// initialization
ctranConfig makeCtranConfigFrom(ncclComm* comm);
commResult_t setCtranCommBase(ncclComm* ncclCommVal);
