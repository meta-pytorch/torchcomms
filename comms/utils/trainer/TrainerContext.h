// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

/*
TrainerContext contains information that passed from xlformer using a python
interface. Any functions that needs to be exposed to python should have "nccl"
in the signature, see src/version.script for details.
*/

__attribute__((noinline, visibility("default"))) void ncclxSetIteration(
    int64_t);
__attribute__((noinline, visibility("default"))) int64_t ncclxGetIteration();
