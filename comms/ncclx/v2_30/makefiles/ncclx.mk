# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX build customizations (compile flags and include paths) for the OSS
# `make` build. Split out of src/Makefile so the upstream NCCL makefile carries
# only a one-line `include` hook: this keeps the forked upstream Makefile close
# to pristine and confines the NCCLX edits -- and the rebase conflicts they
# cause -- to this NCCLX-owned file. The NCCLX source-file lists live in the
# sibling ncclx_sources.mk.
#
# Included by src/Makefile immediately after ../makefiles/{common,version}.mk,
# so CXXFLAGS / NVCUFLAGS (from common.mk) already exist, and the
# SRCDIR/NCCLDIR/SHAREDDIR paths (from the top of src/Makefile) are available.
# These variables are consumed only inside recipes, so their definitions are
# position-independent within the makefile.

##### Mock Scuba Data in CMake build
CXXFLAGS += -DMOCK_SCUBA_DATA -DFOLLY_XLOG_STRIP_PREFIXES=\"${BASE_DIR}\"
# Use header-only fmt to avoid runtime dependency on libfmt.so
CXXFLAGS += -DFMT_HEADER_ONLY=1

# Enable JSON parsing in the NCCLX built-in CSV/JSON tuner (meta/tuner) when
# folly is available. The conda feedstock build sets NCCLX_TUNER_WITH_FOLLY_JSON=1
# (folly is already linked there); a bare OSS `make` without folly leaves it off,
# so meta/tuner compiles only its CSV parser -- matching def_build.bzl.
NCCLX_TUNER_WITH_FOLLY_JSON ?= 0
ifeq ($(NCCLX_TUNER_WITH_FOLLY_JSON),1)
CXXFLAGS += -DNCCLX_TUNER_WITH_FOLLY_JSON
endif

# NCCLX v2.30 uses tuner API v6, which adds the getChunkSize hook. Gate the
# v6-only getChunkSize callback in meta/tuner on this flag -- matching
# def_build.bzl. The conda feedstock build sets NCCLX_TUNER_HAS_GETCHUNKSIZE=1;
# a bare OSS `make` leaves it off so chunkSize overrides compile out (proto/algo
# and channel rules still work).
# TODO: Remove this flag once v2.29 is deleted (all remaining versions use tuner
# API v6+, so getChunkSize can be enabled unconditionally).
NCCLX_TUNER_HAS_GETCHUNKSIZE ?= 0
ifeq ($(NCCLX_TUNER_HAS_GETCHUNKSIZE),1)
CXXFLAGS += -DNCCLX_TUNER_HAS_GETCHUNKSIZE
endif

ifeq ($(ENABLE_TCPDM),0)
CXXFLAGS += -DCTRAN_DISABLE_TCPDM
endif

ifeq ($(ENABLE_PRIMS),1)
CXXFLAGS += -DENABLE_PRIMS
NVCUFLAGS += -DENABLE_PRIMS
endif

##### NCCLX include paths (added to the upstream compile rules via $(INCLUDES))
INCLUDES := -Iinclude
INCLUDES += -Iinclude/plugin
INCLUDES += -I${NCCLDIR}/
INCLUDES += -I${SHAREDDIR}/
INCLUDES += -I${NCCLDIR}/meta
INCLUDES += -I${SHAREDDIR}/meta
INCLUDES += -I${SHAREDDIR}/meta/wrapper
INCLUDES += -Idevice
INCLUDES += -I$(BASE_DIR)
INCLUDES += -I$(CONDA_INCLUDE_DIR)
