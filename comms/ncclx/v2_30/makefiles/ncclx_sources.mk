# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX source-file lists for the OSS `make` build. Split out of src/Makefile so
# the upstream NCCL makefile carries only a one-line `include` hook, confining
# the (frequently-churned) NCCLX source enumeration -- and the rebase conflicts
# it causes against upstream NCCL -- to this NCCLX-owned file. Compile flags and
# include paths live in the sibling ncclx.mk.
#
# Included by src/Makefile right after the upstream `LIBSRCFILES :=` definition
# (so these `+=` appends extend it) and before LIBOBJ is computed from it. The
# TCPDM block below also appends to CXXFLAGS / INCLUDES, which ncclx.mk (included
# earlier) has already defined.

LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/commDump.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/DeviceRackSerial.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/NcclxConfig.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/commHash.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/RankUtil.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/colltrace/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/logger/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/nvls/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/transport/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/socket_ext/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/utilx/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/commstate/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/rma/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/collectives/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/comms-monitor/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/ctran-integration/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/hints/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/algoconf/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/wrapper/*.cc)
LIBSRCFILES += $(wildcard ${SHAREDDIR}/meta/tuner/*.cc)

#### Start of fbcode source files
## Trainer Context for getting/setting trainer steps.
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/trainer/*.cc)
## Logger for Scuba & Basic logging
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/logger/*.cc)
## Common Utils shared between comm libraries
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/*.cc)
## Memory tracing
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/memtrace/*.cc)
## CollTrace source files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/colltrace/*.cc)
## HRDWRingBuffer + GpuClockCalibration host sources
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/hrdw_ring_buffer/*.cc)
## CollTrace CUDA source files (host-side kernel launchers)
LIBSRCFILES += ${BASE_DIR}/comms/utils/hrdw_ring_buffer/GpuClockCalibration.cu
LIBSRCFILES += ${BASE_DIR}/comms/utils/colltrace/HRDWRingBufferInstantiations.cu
## CollTrace plugin source files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/colltrace/plugins/*.cc)
## Include ibverbx files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/ibverbx/*.cc)
## Include cvars files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/cvars/*.cc)
## Include common fault-tolerance files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/common/fault_tolerance/*.cc)
## Include ctran files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/ib/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/nvl/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/socket/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/transport/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/transport/ib/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/bootstrap/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/gpe/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/hints/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/regcache/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/mapper/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/memory/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/window/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/profiler/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/utils/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/utilx/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/colltrace/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/commstate/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/perftrace/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/common/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllGather/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllGather/StreamedRd/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllGatherP/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/ReduceScatter/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllReduce/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllToAll/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/SendRecv/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/Broadcast/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/RMA/*.cc)

ifeq ($(ENABLE_PRIMS),1)
LIBSRCFILES += ${BASE_DIR}/comms/prims/platform/CudaDriverLazy.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/CuMemAllocation.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/CuMemMapping.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/CuMulticastAllocation.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/GpuMemHandler.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/MultimemHandler.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/memory/NvlMemExchange.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/nvl/MultimemNvlTransportConfig.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/nvl/MultimemNvlTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/nvl/MultiPeerNvlTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/MultiPeerTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/trace/PipesTrace.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/MultiPeerIbTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/ibgda/MultipeerIbgdaTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/ibrc/MultipeerIbrcTransport.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/ibrc/MultipeerIbrcTransportCuda.cu
LIBSRCFILES += ${BASE_DIR}/comms/prims/topology/TopologyDiscovery.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/window/HostWindow.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/rdma/IbHcaParser.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/rdma/NicDiscovery.cc
LIBSRCFILES += ${BASE_DIR}/comms/prims/collectives/AllGatherLauncher.cu
LIBSRCFILES += ${BASE_DIR}/comms/prims/collectives/AllToAllv.cu
LIBSRCFILES += ${BASE_DIR}/comms/prims/collectives/ReduceScatterDirectIb.cu
LIBSRCFILES += ${BASE_DIR}/comms/prims/collectives/ReduceScatterDirectIbLauncher.cu
LIBSRCFILES += ${BASE_DIR}/comms/prims/transport/ibgda/MultipeerIbgdaTransportCuda.cu
endif

ifeq ($(ENABLE_TCPDM),1)
## TCP DevMem transport
## Kept in sync with comms/tcp_devmem/CMakeLists.txt (and BUCK).
## The device-manager client uses a header-only "nanothrift" backend
## (comms/tcp_devmem/devmgr/nanothrift/gen/*.h), so no thrift codegen step is
## needed here. FBThrift (gen-cpp2) is only pulled in when -DUSE_FBTHRIFT is set.
CXXFLAGS    += -DTCP_DEVMEM_AGENT
DEVMEM_DIR  := $(BASE_DIR)/comms/tcp_devmem

## ctran <-> tcp_devmem backend glue
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/tcpdevmem/*.cc)
## Top-level transport sources: affinity_mgr, bond_transport, communicator,
## devmem, request_pool, transport, unpack_manager, worker
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/*.cc)
## Device manager, buffer manager, (nanothrift) client and NIC affinity
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/devmgr.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/bufmgr.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/devmgr_client.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_inventory.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_planner.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_probe.cc
## Shared utilities and logging
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/common/*.cc)
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/logger/*.cc)
## Unpack pipeline (host sources + CUDA kernel); tests excluded
LIBSRCFILES += $(DEVMEM_DIR)/unpack/gpu_backend.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_state.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_producer.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_consumer.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_kernel.cu
## YNL is a library that wraps Linux Netlink API and needed for TCPDM:
## https://docs.kernel.org/userspace-api/netlink/intro-specs.html
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/ethtool-user.cpp)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/netdev-user.cpp)

CXXFLAGS += -DYNL_CPP
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/lib/ynl.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/lib/ynl-cpp.cpp)

INCLUDES += -I$(BASE_DIR)/ynl/lib
INCLUDES += -I$(BASE_DIR)/ynl/generated
INCLUDES += -isystem $(BASE_DIR)/ynl
endif
