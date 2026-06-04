// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#define CTRAN_CONCAT_IMPL(a, b) a##b
#define CTRAN_CONCAT(a, b) CTRAN_CONCAT_IMPL(a, b)

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#if !defined(__DRIVER_TYPES_H__)
struct ihipStream_t;
using cudaStream_t = ihipStream_t*;
using CTRAN_CONCAT(cuda, Stream_t) = ihipStream_t*;
struct ihipEvent_t;
using CTRAN_CONCAT(cuda, Event_t) = ihipEvent_t*;
struct ihipGraph;
using CTRAN_CONCAT(cuda, Graph_t) = ihipGraph*;
struct hipGraphNode;
using CTRAN_CONCAT(cuda, GraphNode_t) = hipGraphNode*;
#endif
#else
#include <cuda_runtime_api.h>
#endif
