# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX link flags for the OSS `make` build (added to the upstream link of
# libnccl.so). Upstream ships a single minimal `LDFLAGS +=` line; the fork
# adds --no-undefined, extra -L search dirs (build/, conda), -lm, the prims
# libs (doca/ibverbs/mlx5), the version script, and the cudarthook Bsymbolic
# resolution. Split out of src/Makefile so the forked upstream makefile
# carries only a one-line `include` hook.
#
# Included by src/Makefile at the same point the block was inline (after
# DEPFILES, before the link recipe), so the expanded LDFLAGS is unchanged.

LDFLAGS    += -Wl,--no-undefined
LDFLAGS    += -L${CUDA_LIB} -L${BUILDDIR} -L${CONDA_LIB_DIR} -l$(CUDARTLIB) -lpthread -lm -lrt -ldl
LDFLAGS    += ${THIRD_PARTY_LDFLAGS}
ifeq ($(ENABLE_PRIMS),1)
LDFLAGS    += -ldoca_gpunetio -libverbs -lmlx5
endif
LDFLAGS    += -Wl,--version-script=version.script
ifeq ($(CUDARTLIB), cudarthook)
    # Ensure that cudart symbols are resolved within the module itself.
	LDFLAGS += -Wl,-Bsymbolic
endif
