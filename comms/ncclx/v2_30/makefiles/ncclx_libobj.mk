# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX object-list derivation for the OSS `make` build. LIBSRCFILES mixes
# .cc, .cpp, .c and .cu sources (ynl, ctran/prims, colltrace CUDA) whereas
# upstream NCCL assumes only .cc. Split it per-language into object lists so
# each suffix maps to the right compile rule (see ncclx_rules.mk). Split out
# of src/Makefile so the forked upstream makefile carries only an `include`.
#
# Included by src/Makefile after LIBSRCFILES is fully assembled and OBJDIR
# is defined, and before DEPFILES/LIBOBJ are consumed.

LIBSRCFILES_CC  := $(filter %.cc,$(LIBSRCFILES))
LIBSRCFILES_CPP := $(filter %.cpp,$(LIBSRCFILES))
LIBSRCFILES_C   := $(filter %.c,$(LIBSRCFILES))
LIBSRCFILES_CU  := $(filter %.cu,$(LIBSRCFILES))
LIBOBJ_CC       := $(LIBSRCFILES_CC:%.cc=$(OBJDIR)/%.o)
LIBOBJ_CPP      := $(LIBSRCFILES_CPP:%.cpp=$(OBJDIR)/%.o)
LIBOBJ_C        := $(LIBSRCFILES_C:%.c=$(OBJDIR)/%.o)
LIBOBJ_CU       := $(LIBSRCFILES_CU:%.cu=$(OBJDIR)/%.cu.o)
LIBOBJ          := $(LIBOBJ_CC) $(LIBOBJ_CPP) $(LIBOBJ_C) $(LIBOBJ_CU)
