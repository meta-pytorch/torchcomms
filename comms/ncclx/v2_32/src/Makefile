#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
ENABLE_TCPDM ?= 0

SRCDIR = $(abspath ./)
NCCLDIR = $(abspath ../)
SHAREDDIR = $(abspath ${NCCLDIR}/..)

include ../makefiles/common.mk
include ../makefiles/version.mk

# Shared logging sources use the standalone telemetry stub and header-only fmt.
CXXFLAGS += -DMOCK_SCUBA_DATA -DFOLLY_XLOG_STRIP_PREFIXES=\"${BASE_DIR}\"
CXXFLAGS += -DFMT_HEADER_ONLY=1
CXXFLAGS += -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO -DSPDLOG_FMT_EXTERNAL
NVCUFLAGS += -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO -DSPDLOG_FMT_EXTERNAL

# Enable JSON parsing in the built-in tuner when folly is available.
NCCLX_TUNER_WITH_FOLLY_JSON ?= 0
ifeq ($(NCCLX_TUNER_WITH_FOLLY_JSON),1)
CXXFLAGS += -DNCCLX_TUNER_WITH_FOLLY_JSON
endif

# NCCLX v2.32 uses tuner API v6, which includes getChunkSize.
NCCLX_TUNER_HAS_GETCHUNKSIZE ?= 0
ifeq ($(NCCLX_TUNER_HAS_GETCHUNKSIZE),1)
CXXFLAGS += -DNCCLX_TUNER_HAS_GETCHUNKSIZE
endif

ifeq ($(ENABLE_TCPDM),0)
CXXFLAGS += -DCTRAN_DISABLE_TCPDM
endif

##### src files
# The only public headers are nccl.h and nccl_device.h. Any changes to this list should be discussed widely.
INCEXPORTS  := nccl.h nccl_device.h \
	$(patsubst include/%,%,$(wildcard include/nccl_device/*.h include/nccl_device/*/*.h include/nccl_device/*/*/*.h)) \
	$(patsubst include/%,%,$(wildcard include/param/*.h))

# On Linux use real net_socket and net_ib; on Windows use stubs (avoid multiple definition)
TRANSPORT_CC = $(wildcard transport/*.cc)
ifeq ($(NCCL_OS_LINUX), 1)
TRANSPORT_CC := $(filter-out transport/net_socket_stub.cc transport/net_ib_stub.cc,$(TRANSPORT_CC))
endif

LIBSRCFILES := \
	bootstrap.cc channel.cc collectives.cc debug.cc group.cc \
	init.cc proxy.cc transport.cc mnnvl.cc allocator.cc dev_runtime.cc cft_dev_runtime.cc dev_runtime_segments.cc sym_kernels.cc ce_coll.cc mem_manager.cc \
	$(wildcard config/*.cc) \
	$(wildcard graph/*.cc) \
	$(wildcard enqueue/*.cc) \
	$(wildcard enqueue/task_prep/*.cc) \
	$(wildcard enqueue/task_sched/*.cc) \
	$(wildcard misc/*.cc) \
	$(TRANSPORT_CC) \
	$(wildcard transport/net_socket/*.cc) \
	$(wildcard transport/net_ib/*.cc) \
	$(wildcard transport/net_ib/gdaki/*.cc) \
	$(wildcard transport/net_ib/gdaki/gpucontext/*.cc) \
	$(wildcard register/*.cc) \
	$(wildcard plugin/*.cc) \
	$(wildcard plugin/net/*.cc) \
	$(wildcard plugin/gin/*.cc) \
	$(wildcard plugin/rma/*.cc) \
	$(wildcard plugin/tuner/*.cc) \
	$(wildcard plugin/profiler/*.cc) \
	$(wildcard plugin/env/*.cc) \
	$(wildcard nccl_device/*.cc) \
	$(wildcard scheduler/*.cc) \
	$(wildcard gin/*.cc) \
	$(wildcard gin/proxy_gpucontext/*.cc) \
	$(wildcard rma/*.cc) \
	$(wildcard devcomm/*.cc) \
	$(wildcard tuning/*.cc) \
	$(wildcard tuning/sym_model/*.cc) \
	$(filter-out param/ncclparam.cc,$(wildcard param/*.cc)) \
	$(filter-out ras/client.cc,$(wildcard ras/*.cc))

# Keep this allowlist synchronized with the subset in def_build.bzl. Shared
# meta sources are added only after the baseline patches they depend on land.
LIBSRCFILES += \
	${SHAREDDIR}/meta/RankUtil.cc \
	${SHAREDDIR}/meta/DeviceRackSerial.cc \
	${SHAREDDIR}/meta/collectives/AllReduceSparseBlock.cc \
	${SHAREDDIR}/meta/colltrace/CollTraceWrapper.cc \
	${SHAREDDIR}/meta/commHash.cc \
	${SHAREDDIR}/meta/ctran-integration/BaselineBootstrap.cc \
	${SHAREDDIR}/meta/hints/GlobalHints.cc \
	${SHAREDDIR}/meta/hints/Hints.cc \
	${SHAREDDIR}/meta/logger/DebugExt.cc \
	${SHAREDDIR}/meta/NcclxConfig.cc \
	${SHAREDDIR}/meta/tcpstore/Backoff.cc \
	${SHAREDDIR}/meta/tcpstore/TCPSocket.cc \
	${SHAREDDIR}/meta/tcpstore/TCPStore.cc \
	${SHAREDDIR}/meta/transport/NcclxNetPluginHelper.cc \
	${SHAREDDIR}/meta/transport/transportConnect.cc \
	${SHAREDDIR}/meta/transport/transportExt.cc \
	${SHAREDDIR}/meta/transport/transportProxy.cc \
	${SHAREDDIR}/meta/tuner/Int64Range.cc \
	${SHAREDDIR}/meta/tuner/MetaTuner.cc \
	${SHAREDDIR}/meta/commstate/FactoryCommStateX.cc \
	${SHAREDDIR}/meta/wrapper/MetaFactory.cc

# HintsConversion.cc arrives with the 2.32 RMA port and marks when its baseline
# APIs are available. Keep the pre-existing shared RMA sources out until then.
ifneq ($(wildcard ${SHAREDDIR}/meta/wrapper/HintsConversion.cc),)
LIBSRCFILES += \
	$(wildcard ${SHAREDDIR}/meta/rma/*.cc) \
	${SHAREDDIR}/meta/wrapper/HintsConversion.cc
endif

# StuckWatchdog.h arrives with the 2.32 NVLS watchdog port and marks when its
# call sites and shared implementation are compatible with this baseline.
ifneq ($(wildcard ${SHAREDDIR}/meta/nvls/StuckWatchdog.h),)
LIBSRCFILES += ${SHAREDDIR}/meta/nvls/NvlsBindWatchdog.cc
endif

#### Shared support and CTran source files
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

INCLUDES := -Iinclude
INCLUDES += -Iinclude/plugin
INCLUDES += -I${NCCLDIR}/
INCLUDES += -I${SHAREDDIR}/
INCLUDES += -I${SHAREDDIR}/meta
INCLUDES += -I${SHAREDDIR}/meta/wrapper
INCLUDES += -Idevice
INCLUDES += -I$(BASE_DIR)
INCLUDES += -I$(CONDA_INCLUDE_DIR)

ifeq ($(ENABLE_TCPDM),1)
## TCP DevMem transport
## Kept in sync with comms/tcp_devmem/CMakeLists.txt (and BUCK).
CXXFLAGS    += -DTCP_DEVMEM_AGENT
DEVMEM_DIR  := $(BASE_DIR)/comms/tcp_devmem

## ctran <-> tcp_devmem backend glue
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/tcpdevmem/*.cc)
## Top-level transport sources
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/*.cc)
## Device manager, buffer manager, client, and NIC affinity
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/devmgr.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/bufmgr.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/devmgr_client.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_inventory.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_planner.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/affinity_probe.cc
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/local_addr.cc
## Shared utilities and logging
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/common/*.cc)
LIBSRCFILES += $(wildcard $(DEVMEM_DIR)/logger/*.cc)
## Unpack pipeline (host sources + CUDA kernel); tests excluded
LIBSRCFILES += $(DEVMEM_DIR)/unpack/cpu_backend.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/gpu_backend.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_state.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_producer.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_consumer.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/external_unpack.cc
LIBSRCFILES += $(DEVMEM_DIR)/unpack/batch_unpack_kernel.cu
## YNL wraps the Linux Netlink API used by TCPDM.
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/ethtool-user.cpp)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/netdev-user.cpp)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/rt-addr-user.cpp)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/generated/rt-link-user.cpp)

CXXFLAGS += -DYNL_CPP
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/lib/ynl.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/ynl/lib/ynl-cpp.cpp)

INCLUDES += -I$(BASE_DIR)/ynl/lib
INCLUDES += -I$(BASE_DIR)/ynl/generated
INCLUDES += -isystem $(BASE_DIR)/ynl
endif

BINSRCFILES := ras/client.cc
PARAMBINSRCFILES := param/ncclparam.cc

ifeq ($(NCCL_OS_LINUX), 1)
	LIBSRCFILES += os/linux.cc os/linux_ipcsocket.cc os/linux_socket_pair.cc diagnostics.cc $(wildcard diagnostics/*.cc)
else ifeq ($(NCCL_OS_WINDOWS), 1)
	LIBSRCFILES += os/windows.cc os/windows_ipcsocket.cc os/windows_socket_pair.cc
endif

ifneq ($(NVTX), 0)
	LIBSRCFILES += init_nvtx.cc
endif

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### binaries
BINNAME := ncclras
PARAMBIN := ncclparam
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
BINDIR := $(BUILDDIR)/bin

##### target files
CUDARTLIB  ?= cudart_static

# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
ifeq ($(CUDARTLIB), cudart_static)
	LIBSRCFILES += enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBSRCFILES_CC  := $(filter %.cc,$(LIBSRCFILES))
LIBSRCFILES_CPP := $(filter %.cpp,$(LIBSRCFILES))
LIBSRCFILES_C   := $(filter %.c,$(LIBSRCFILES))
LIBSRCFILES_CU  := $(filter %.cu,$(LIBSRCFILES))
LIBOBJ_CC       := $(LIBSRCFILES_CC:%.cc=$(OBJDIR)/%.o)
LIBOBJ_CPP      := $(LIBSRCFILES_CPP:%.cpp=$(OBJDIR)/%.o)
LIBOBJ_C        := $(LIBSRCFILES_C:%.c=$(OBJDIR)/%.o)
LIBOBJ_CU       := $(LIBSRCFILES_CU:%.cu=$(OBJDIR)/%.cu.o)
LIBOBJ          := $(LIBOBJ_CC) $(LIBOBJ_CPP) $(LIBOBJ_C) $(LIBOBJ_CU)
BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o) $(OBJDIR)/ras/ras_param_client.o
PARAMBINOBJ     := $(PARAMBINSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d) $(PARAMBINOBJ:%.o=%.d)
LDFLAGS    += -Wl,--no-undefined
LDFLAGS    += -L${CUDA_LIB} -L${BUILDDIR} -L${CONDA_LIB_DIR} -l$(CUDARTLIB) -lpthread -lm -lrt -ldl
LDFLAGS    += ${THIRD_PARTY_LDFLAGS}
LDFLAGS    += -L$(LIBDIR) -lobservatory
LDFLAGS    += -Wl,--version-script=version.script
ifeq ($(CUDARTLIB), cudarthook)
	# Ensure that cudart symbols are resolved within the module itself.
	LDFLAGS += -Wl,-Bsymbolic
endif
INCPLUGIN  := include/plugin

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest
DIAG_DEVMANIFEST := $(BUILDDIR)/obj/diagnostics/device/manifest
DEVMANIFESTS := $(DEVMANIFEST)
ifeq ($(NCCL_OS_LINUX), 1)
DEVMANIFESTS += $(DIAG_DEVMANIFEST)
endif


GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h
$(GIT_VERSION_FILE): ALWAYS_REBUILD
	@mkdir -p $(dir $@)
	@./misc/generate_git_version.py $@

# DOCA GPUNetIO definitions
DOCA_HOME        ?= transport/net_ib/gdaki/doca-gpunetio
DOCA_INC_INSTALL := $(INCDIR)/nccl_device/gin/gdaki/doca_gpunetio
DOCA_OBJDIR      := $(OBJDIR)/transport/net_ib/gdaki/doca-gpunetio
DOCA_INCLUDES    := $(DOCA_HOME)/include/doca_gpunetio_device.h $(wildcard $(DOCA_HOME)/include/common/*.h) $(wildcard $(DOCA_HOME)/include/device/*.cuh)
DOCA_INCTARGETS  := $(DOCA_INCLUDES:$(DOCA_HOME)/include/%=$(DOCA_INC_INSTALL)/%)
INCTARGETS       += $(DOCA_INCTARGETS)
DOCA_LIBSRC      := doca_verbs_qp.cpp doca_verbs_cq.cpp doca_verbs_device_attr.cpp doca_verbs_umem.cpp doca_verbs_srq.cpp doca_verbs_uar.cpp doca_verbs_dev.cpp doca_gpunetio.cpp doca_gpunetio_log.cpp doca_gpunetio_high_level.cpp doca_gpunetio_cuda_wrapper.cpp doca_verbs_mlx5dv_wrapper.cpp doca_verbs_ibv_wrapper.cpp doca_gpunetio_gdrcopy.cpp doca_gpunetio_sdk_wrapper.cpp doca_verbs_dev_sdk_wrapper.cpp doca_verbs_umem_sdk_wrapper.cpp doca_verbs_uar_sdk_wrapper.cpp doca_verbs_cq_sdk_wrapper.cpp doca_verbs_qp_sdk_wrapper.cpp
DOCA_LIBOBJ      := $(DOCA_LIBSRC:%.cpp=$(DOCA_OBJDIR)/%.o)
LIBOBJ           += $(DOCA_LIBOBJ)

# EFA-DP-Direct device header installation (mirrors DOCA pattern above)
EFA_DP_DIRECT_HOME        := transport/net_efa_gda/efa-dp-direct
EFA_DP_DIRECT_INC_INSTALL := $(INCDIR)/nccl_device/transport/net_efa_gda/efa-dp-direct
EFA_DP_DIRECT_INCLUDES    := $(wildcard $(EFA_DP_DIRECT_HOME)/include/device/*.cuh) $(wildcard $(EFA_DP_DIRECT_HOME)/include/device/*.h) $(wildcard $(EFA_DP_DIRECT_HOME)/include/common/*.h)
EFA_DP_DIRECT_INCTARGETS  := $(EFA_DP_DIRECT_INCLUDES:$(EFA_DP_DIRECT_HOME)/include/%=$(EFA_DP_DIRECT_INC_INSTALL)/include/%)
INCTARGETS                += $(EFA_DP_DIRECT_INCTARGETS)

##### rules
BUILD_NCCL_CLI ?= 1
ifeq ($(BUILD_NCCL_CLI),1)
build : lib staticlib binary
else
build : lib staticlib
endif

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

binary : $(BINDIR)/$(BINNAME) $(BINDIR)/$(PARAMBIN)

headers : $(INCTARGETS)

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device BASE_DIR=$(BASE_DIR) CONDA_INCLUDE_DIR=$(CONDA_INCLUDE_DIR)

$(DIAG_DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./diagnostics/device

# Empty target to force rebuild
.PHONY: ALWAYS_REBUILD

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ../makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

# libobservatory.so. See comms/observatory/README.md.
#
# This library must retain default visibility and remain outside libnccl's
# version script so all users resolve the process-wide registry here.
LIBOBSERVATORY_SRC    := ${BASE_DIR}/comms/observatory/colltrace/LifecycleFeedRegistry.cc
LIBOBSERVATORY_NAME   := libobservatory.so
LIBOBSERVATORY_SONAME := $(LIBOBSERVATORY_NAME).1
LIBOBSERVATORY_HDRS   := $(wildcard ${BASE_DIR}/comms/observatory/colltrace/*.h)

$(LIBDIR)/$(LIBOBSERVATORY_SONAME): $(LIBOBSERVATORY_SRC) $(LIBOBSERVATORY_HDRS)
	@printf "Linking    %-35s > %s\n" $(LIBOBSERVATORY_SONAME) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -fvisibility=default -I${BASE_DIR} -shared \
	    -Wl,--no-undefined -Wl,-soname,$(LIBOBSERVATORY_SONAME) -o $@ $<
	ln -sf $(LIBOBSERVATORY_SONAME) $(LIBDIR)/$(LIBOBSERVATORY_NAME)

# Device objects are linked into libnccl.so so cross-translation-unit device
# symbols resolve from the shared library.
$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVMANIFESTS) $(LIBDIR)/$(LIBOBSERVATORY_SONAME)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $$(cat $(DEVMANIFESTS)) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVMANIFESTS)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ) $$(cat $(DEVMANIFESTS))

$(BINDIR)/$(BINNAME): $(BINOBJ)
	@printf "Linking    %-35s > %s\n" $(BINNAME) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BINDIR)/$(PARAMBIN): $(PARAMBINOBJ) $(LIBDIR)/$(LIBTARGET)
	@printf "Linking    %-35s > %s\n" $(PARAMBIN) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIBDIR) -L${CONDA_LIB_DIR} -lnccl -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl -latomic

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/param/%.h: include/param/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/param
	install -m 644 $< $@

$(INCDIR)/nccl_device/%.h: include/nccl_device/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device
	install -m 644 $< $@

$(INCDIR)/nccl_device/impl/%.h: include/nccl_device/impl/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/impl
	install -m 644 $< $@

$(INCDIR)/nccl_device/gin/%.h: include/nccl_device/gin/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin
	install -m 644 $< $@

$(INCDIR)/nccl_device/gin/gdaki/%.h: include/nccl_device/gin/gdaki/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin/gdaki
	install -m 644 $< $@

$(INCDIR)/nccl_device/gin/gpi/%.h: include/nccl_device/gin/gpi/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin/gpi
	install -m 644 $< $@

$(INCDIR)/nccl_device/gin/proxy/%.h: include/nccl_device/gin/proxy/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin/proxy
	install -m 644 $< $@

$(INCDIR)/nccl_device/gin/efa_gda/%.h: include/nccl_device/gin/efa_gda/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin/efa_gda
	install -m 644 $< $@

$(EFA_DP_DIRECT_INC_INSTALL)/include/device/%.cuh: $(EFA_DP_DIRECT_HOME)/include/device/%.cuh
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(EFA_DP_DIRECT_INC_INSTALL)/include/device
	install -m 644 $< $@

$(EFA_DP_DIRECT_INC_INSTALL)/include/device/%.h: $(EFA_DP_DIRECT_HOME)/include/device/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(EFA_DP_DIRECT_INC_INSTALL)/include/device
	install -m 644 $< $@

$(EFA_DP_DIRECT_INC_INSTALL)/include/common/%.h: $(EFA_DP_DIRECT_HOME)/include/common/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(EFA_DP_DIRECT_INC_INSTALL)/include/common
	install -m 644 $< $@

$(DOCA_INC_INSTALL)/%.h: $(DOCA_HOME)/include/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DOCA_INC_INSTALL)
	install -m 644 $< $@

$(DOCA_INC_INSTALL)/common/%.h: $(DOCA_HOME)/include/common/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DOCA_INC_INSTALL)/common
	install -m 644 $< $@

$(DOCA_INC_INSTALL)/device/%.cuh: $(DOCA_HOME)/include/device/%.cuh
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DOCA_INC_INSTALL)/device
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/ras/ras_param_client.o: ras/ras_param.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) -DNCCL_RAS_CLIENT -Iinclude -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
	                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.cpp $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
	                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.c $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CC) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CC) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
	                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.cu.o : %.cu $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -I. -I$(INCDIR) -I$(OBJDIR)/include $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(NVCC) $(NVCUFLAGS) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -I. -I$(INCDIR) -I$(OBJDIR)/include $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
	                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/misc/git_version.o: $(GIT_VERSION_FILE)

$(DOCA_OBJDIR)/%.o : $(DOCA_HOME)/src/%.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -Iinclude -I$(DOCA_HOME)/src -I$(DOCA_HOME)/include $(CXXFLAGS) -c $< -o $@
	@$(CXX) -Iinclude -I$(DOCA_HOME)/src -I$(DOCA_HOME)/include $(CXXFLAGS) -MM $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C device clean
	$(MAKE) -C diagnostics/device clean
	rm -rf ${BINDIR} ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v -r $(BUILDDIR)/include/* $(PREFIX)/include/
ifeq ($(BUILD_NCCL_CLI),1)
	cp -v $(BUILDDIR)/bin/ncclras $(PREFIX)/bin/
	cp -v $(BUILDDIR)/bin/ncclparam $(PREFIX)/bin/
endif

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
