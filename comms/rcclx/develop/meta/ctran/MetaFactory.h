// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include "comms/ctran/CtranComm.h"
#include "comms/utils/commSpecs.h"
#include "nccl.h"

// Convert commResult_t to ncclResult_t
ncclResult_t metaCommToNccl(commResult_t result);

// Convert ncclResult_t to commResult_t
commResult_t ncclToMetaComm(ncclResult_t result);

// Convert ncclDataType_t to commDataType_t
commDataType_t ncclToMetaComm(ncclDataType_t dataType);

commResult_t initNcclCommCtran(ncclComm* ncclCommVal);

std::unique_ptr<ncclx::CommStateX> createCtranCommStateXFromNcclComm(
    ncclComm* ncclComm,
    CtranComm* ctranComm);
