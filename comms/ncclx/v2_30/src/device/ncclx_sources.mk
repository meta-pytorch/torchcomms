# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX device (.cu) source list for the OSS `make` build. Split out of
# device/Makefile so the upstream NCCL device makefile carries only a one-line
# `include` hook, confining the NCCLX ctran kernel list (which churns as ctran
# algorithms are added) and the generated-ctran rules to this NCCLX-owned file.
#
# Included right after the upstream `SRCS = common.cu onerank.cu` line, so these
# `+=` appends extend it and $(OBJDIR) (from device/Makefile) is defined.

SRCS += ${BASE_DIR}/comms/ctran/algos/AllToAll/AllToAllPImpl.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllToAll/DeviceAllToAllvPipes.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllGather/AllGatherRing.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllGather/StreamedRd/Impl.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllGather/AllGatherBrucksFF.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllReduce/AllReduceShm.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllGatherP/DirectImpl.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/AllGatherP/PipelineImpl.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/Broadcast/Broadcast.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/SendRecv/SendRecv.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/SendRecv/SendRecvP2p.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/Barrier.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/Checksum.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/RMA/PutSignal.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/RMA/Get.cu
SRCS += ${BASE_DIR}/comms/ctran/algos/DevShmState.cu
SRCS += ${BASE_DIR}/comms/ncclx/meta/device/all_reduce_sparse_block.cu

-include $(OBJDIR)/gensrc/ctran_rules.mk
SRCS += $(CTRAN_GEN_SRCS)
