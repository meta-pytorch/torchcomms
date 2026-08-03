# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX NVCC compile-flag customizations for the device OSS `make` build:
# the TCPDM (CTRAN_DISABLE_TCPDM) and prims (ENABLE_PRIMS) preprocessor
# defines added to NVCUFLAGS / NVCUFLAGS_SYM. Split out of device/Makefile
# so the forked upstream device makefile carries only a one-line `include`.
#
# Included by device/Makefile at the same point the blocks were inline
# (after NVCUFLAGS / NVCUFLAGS_SYM are set up), so their expansion is
# unchanged.

ifneq ($(ENABLE_TCPDM),1)
NVCUFLAGS += --compiler-options "-DCTRAN_DISABLE_TCPDM"
NVCUFLAGS_SYM += --compiler-options "-DCTRAN_DISABLE_TCPDM"
endif

ifeq ($(ENABLE_PRIMS),1)
NVCUFLAGS += --compiler-options "-DENABLE_PRIMS"
NVCUFLAGS_SYM += --compiler-options "-DENABLE_PRIMS"
endif
