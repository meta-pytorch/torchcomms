// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ncclx/headers/helpers.h"

#ifdef HEADER_NAMESPACE
#include NAMESPACED_HEADER_PATH(HEADER_NAMESPACE, nccl.h)
#else
#include HEADER_PATH(nccl.h)
#endif
