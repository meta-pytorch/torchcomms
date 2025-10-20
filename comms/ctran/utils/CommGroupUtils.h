// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Sibling variable of ncclGroupDepth, managed by NCCLX to ensure match.
// Users should not modify this variable directly; instead, rely on NCCLX for
// updates.
extern __thread int commGroupDepth;
