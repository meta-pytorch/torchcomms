// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Self-contained mlx5 direct verbs type definitions.
// Copied from ncclx mlx5dvcore.h — only the subset needed by UniFlow.

#ifndef UNIFLOW_MLX5_DIRECT
#define UNIFLOW_MLX5_DIRECT 0
#endif

#if UNIFLOW_MLX5_DIRECT
#include <infiniband/mlx5dv.h>
#else

enum mlx5dv_reg_dmabuf_access {
  MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT = (1 << 0),
};

#endif // UNIFLOW_MLX5_DIRECT
