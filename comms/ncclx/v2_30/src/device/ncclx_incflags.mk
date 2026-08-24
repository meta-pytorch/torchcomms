# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX device include-path additions for the OSS `make` build, appended to
# the upstream INCFLAGS: the repo-root / shared trees plus BASE_DIR and the
# conda include dir. Split out of device/Makefile so the forked upstream
# device makefile carries only a one-line `include` hook.
#
# Included by device/Makefile right after upstream sets INCFLAGS and before
# INCFLAGS is consumed (NVCUFLAGS/CXXFLAGS), so INCFLAGS expands unchanged.

INCFLAGS += -I../../
INCFLAGS += -I${SHAREDDIR}/

INCFLAGS  += -I$(BASE_DIR)
INCFLAGS  += -I$(CONDA_INCLUDE_DIR)
