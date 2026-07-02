// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/ctran/CtranComm.h"

// Build a splitShare CtranComm: a "virtual" child that defines a subset of
// parent ranks while sharing the parent's ctran_ resources (mapper/algo/gpe).
// The child has ctran_ == nullptr by design — it must not be used for direct
// collectives or RMA. It is intended for internal persistent-collective
// machinery that needs to register a window over a subset of parent ranks.
// The parent must outlive the child.
//
// MPI-style split: parent ranks with the same color end up in the same child
// comm, ordered by key (ties broken by parent rank). Every parent rank must
// participate.
commResult_t ctranCommSplitShare(
    CtranComm* parent,
    int color,
    int key,
    std::shared_ptr<CtranComm>* childOut);

// Convenience: splitShare for the caller's local NVL domain.
commResult_t ctranCommSplitLocalNvl(
    CtranComm* parent,
    std::shared_ptr<CtranComm>* childOut);
