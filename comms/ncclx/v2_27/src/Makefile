#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

ENABLE_TCPDM ?= 0

# Assuming calling from src directory
SRCDIR = $(abspath ./)
NCCLDIR = $(abspath ../)

include ../makefiles/common.mk
include ../makefiles/version.mk

##### Mock Scuba Data in CMake build
CXXFLAGS += -DMOCK_SCUBA_DATA -DFOLLY_XLOG_STRIP_PREFIXES=\"${BASE_DIR}\"

ifeq ($(ENABLE_TCPDM),0)
CXXFLAGS += -DCTRAN_DISABLE_TCPDM
endif

##### src files
INCEXPORTS  := nccl.h CtranEx.h CtranExComm.h
LIBSRCFILES := \
 	$(wildcard ${SRCDIR}/*.cc) \
	$(wildcard ${SRCDIR}/graph/*.cc) \
	$(wildcard ${SRCDIR}/misc/*.cc) \
	$(wildcard ${SRCDIR}/transport/*.cc) \
	$(wildcard ${SRCDIR}/register/*.cc) \
	$(wildcard ${SRCDIR}/plugin/*.cc) \
	$(wildcard ${SRCDIR}/plugin/net/*.cc) \
	$(wildcard ${SRCDIR}/plugin/tuner/*.cc) \
	$(wildcard ${SRCDIR}/plugin/profiler/*.cc) \
	$(filter-out ${SRCDIR}/ras/client.cc,$(wildcard ${SRCDIR}/ras/*.cc))

LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/analyzer/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/colltrace/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/logger/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/tcpstore/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/transport/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/utilx/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/commstate/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/rma/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/collectives/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/comms-monitor/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/ctran-integration/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/hints/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/algoconf/*.cc)
LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/wrapper/*.cc)

#### Start of fbcode source files
## Trainer Context for getting/setting trainer steps.
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/trainer/*.cc)
## Logger for Scuba & Basic logging
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/logger/*.cc)
## Common Utils shared between comm libraries
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/*.cc)
## CollTrace source files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/colltrace/*.cc)
## CollTrace plugin source files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/colltrace/plugins/*.cc)
## Include ibverbx files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/ibverbx/*.cc)
## Include cvars files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/utils/cvars/*.cc)
## Include ctran files
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/ib/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/nvl/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/socket/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/bootstrap/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/gpe/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/hints/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/mapper/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/memory/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/window/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/profiler/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/utils/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/utilx/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/colltrace/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/commstate/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/common/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllGather/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllGatherP/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/ReduceScatter/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllReduce/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllToAll/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/AllToAllvDedup/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/SendRecv/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/Broadcast/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/algos/RMA/*.cc)

INCLUDES := -Iinclude
INCLUDES += -Iinclude/plugin
INCLUDES += -I${NCCLDIR}/
INCLUDES += -I${NCCLDIR}/meta
INCLUDES += -I${NCCLDIR}/meta/wrapper
INCLUDES += -I${NCCLDIR}/meta/hints
INCLUDES += -I${NCCLDIR}/meta/algoconf
INCLUDES += -I${NCCLDIR}/meta/logger -Idevice
INCLUDES += -I$(BASE_DIR)
INCLUDES += -I$(CONDA_INCLUDE_DIR)

ifeq ($(ENABLE_TCPDM),1)
## Thrift buffer manager client
CXXFLAGS    += -DTCP_DEVMEM_AGENT
THRIFT      ?= thrift1
DEVMEM_DIR  := $(BASE_DIR)/comms/tcp_devmem
THRIFT_FILE := $(DEVMEM_DIR)/devmgr/devmgr.thrift
THRIFT_H    := $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_client.h

LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgrAsyncClient.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr_processmap_binary.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr_processmap_compact.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_constants.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_data.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_metadata.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_binary.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_compact.cpp
LIBSRCFILES += $(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_serialization.cpp

$(THRIFT_H): $(THRIFT_FILE)
	@printf "Generating  %-35s > %s\n" $(THRIFT_FILE) $(DEVMEM_DIR)/devmgr
	$(THRIFT) --gen mstch_cpp2:include_prefix=$(DEVMEM_DIR)/devmgr -o $(DEVMEM_DIR)/devmgr $(THRIFT_FILE)

$(DEVMEM_DIR)/devmgr/devmgr_client.cc: $(THRIFT_H)
$(DEVMEM_DIR)/transport.cc: $(THRIFT_H)

$(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgrAsyncClient.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr_processmap_binary.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/DevMgr_processmap_compact.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_constants.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_data.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_metadata.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_binary.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_compact.cpp: $(THRIFT_H)
$(DEVMEM_DIR)/devmgr/gen-cpp2/devmgr_types_serialization.cpp: $(THRIFT_H)

## Include TCP Devmem transport files and dependencies
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/ctran/backends/tcpdevmem/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/affinity_mgr.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/bond_transport.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/communicator.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/devmem.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/request_pool.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/transport.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/worker.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/devmgr/devmgr.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/devmgr/devmgr_client.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/common/*.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/unpack/external_unpack.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/unpack/internal_unpack.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/unpack/batch_unpack_state.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/unpack/batch_unpack_producer.cc)
LIBSRCFILES += $(wildcard ${BASE_DIR}/comms/tcp_devmem/unpack/batch_unpack_consumer.cc)
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

BINSRCFILES := ras/client.cc

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### binaries
BINNAME := ncclras
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ${NCCLDIR}/build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
BINDIR := $(BUILDDIR)/bin
##### target files
CUDARTLIB  ?= cudart_static

# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
ifeq ($(CUDARTLIB), cudart_static)
	#LIBSRCFILES += enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBSRCFILES_CC  := $(filter %.cc,$(LIBSRCFILES))
LIBSRCFILES_CPP := $(filter %.cpp,$(LIBSRCFILES))
LIBSRCFILES_C   := $(filter %.c,$(LIBSRCFILES))
LIBOBJ_CC       := $(LIBSRCFILES_CC:%.cc=$(OBJDIR)/%.o)
LIBOBJ_CPP      := $(LIBSRCFILES_CPP:%.cpp=$(OBJDIR)/%.o)
LIBOBJ_C        := $(LIBSRCFILES_C:%.c=$(OBJDIR)/%.o)
LIBOBJ          := $(LIBOBJ_CC) $(LIBOBJ_CPP) $(LIBOBJ_C)
BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d)
LDFLAGS    += -Wl,--no-undefined
LDFLAGS    += -L${CUDA_LIB} -L${BUILDDIR} -L${CONDA_LIB_DIR} -l$(CUDARTLIB) -lpthread -lm -lrt -ldl
LDFLAGS    += ${THIRD_PARTY_LDFLAGS}
LDFLAGS    += -Wl,--version-script=version.script
ifeq ($(CUDARTLIB), cudarthook)
    # Ensure that cudart symbols are resolved within the module itself.
	LDFLAGS += -Wl,-Bsymbolic
endif

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest

##### rules
build : lib staticlib binary

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

binary : $(BINDIR)/$(BINNAME)

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device BASE_DIR=$(BASE_DIR) CONDA_INCLUDE_DIR=$(CONDA_INCLUDE_DIR)

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ${NCCLDIR}/makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
		-e "s/\$${nccl:DevSignature}/$(DEV_SIGNATURE)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $$(cat $(DEVMANIFEST)) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ) $$(cat $(DEVMANIFEST))

$(BINDIR)/$(BINNAME): $(BINOBJ)
	@printf "Linking    %-35s > %s\n" $(BINNAME) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

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

$(INCDIR)/CtranEx.h : ${BASE_DIR}/comms/ctran/CtranEx.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/CtranExComm.h : ${NCCLDIR}/meta/wrapper/CtranExComm.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.cpp $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.c $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CC) -I. -I$(INCDIR) $(CFLAGS) -fPIC $(INCLUDES) -c $< -o $@
	@$(CC) -I. -I$(INCDIR) $(CFLAGS) -fPIC $(INCLUDES) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C device clean
	rm -rf ${BINDIR} ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/
	cp -v $(BUILDDIR)/bin/ncclras $(PREFIX)/bin/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ${NCCLDIR}/makefiles/formatting.mk
