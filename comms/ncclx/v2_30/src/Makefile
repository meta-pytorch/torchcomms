#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

ENABLE_TCPDM ?= 0
ENABLE_PRIMS ?= 1

# Assuming calling from src directory
SRCDIR = $(abspath ./)
NCCLDIR = $(abspath ../)
SHAREDDIR = $(abspath ${NCCLDIR}/..)

# fbcode/prims CUDA sources include Folly headers that require C++20.
CXXSTD ?= -std=c++20

include ../makefiles/common.mk
include ../makefiles/version.mk

# NCCLX compile-flag and include-path customizations (see makefiles/ncclx.mk).
include ../makefiles/ncclx.mk

##### src files
INCEXPORTS  := nccl.h ncclx.h nccl_device.h \
	$(patsubst ${SRCDIR}/include/%,%,$(wildcard ${SRCDIR}/include/nccl_device/*.h ${SRCDIR}/include/nccl_device/*/*.h ${SRCDIR}/include/nccl_device/*/*/*.h)) \
	$(patsubst ${SRCDIR}/include/%,%,$(wildcard ${SRCDIR}/include/param/*.h))

# On Linux use real net_socket and net_ib; on Windows use stubs (avoid multiple definition)
TRANSPORT_CC = $(wildcard transport/*.cc)
ifeq ($(NCCL_OS_LINUX), 1)
TRANSPORT_CC := $(filter-out transport/net_socket_stub.cc transport/net_ib_stub.cc,$(TRANSPORT_CC))
endif

LIBSRCFILES := \
	$(filter-out enhcompat.cc init_nvtx.cc,$(wildcard *.cc)) \
	$(wildcard graph/*.cc) \
	$(wildcard misc/*.cc) \
	$(TRANSPORT_CC) \
	$(wildcard transport/net_ib/*.cc) \
	$(wildcard transport/net_ib/gdaki/*.cc) \
	$(wildcard register/*.cc) \
	$(wildcard plugin/*.cc) \
	$(wildcard plugin/net/*.cc) \
	$(wildcard plugin/gin/*.cc) \
	$(wildcard plugin/tuner/*.cc) \
	$(wildcard plugin/profiler/*.cc) \
	$(wildcard plugin/env/*.cc) \
	$(wildcard nccl_device/*.cc) \
	$(wildcard scheduler/*.cc) \
	$(wildcard gin/*.cc) \
	$(wildcard rma/*.cc) \
	$(wildcard devcomm/*.cc) \
	$(filter-out param/ncclparam.cc,$(wildcard param/*.cc)) \
	$(filter-out ras/client.cc,$(wildcard ras/*.cc))
# NCCLX: re-root the (otherwise pristine-upstream) source list under ${SRCDIR}
# for the out-of-tree build, so the list above stays byte-for-byte upstream.
LIBSRCFILES := $(addprefix ${SRCDIR}/,$(LIBSRCFILES))

# NCCLX source-file lists (see makefiles/ncclx_sources.mk).
include ../makefiles/ncclx_sources.mk

BINSRCFILES := ${SRCDIR}/ras/client.cc
PARAMBINSRCFILES := ${SRCDIR}/param/ncclparam.cc

ifeq ($(NCCL_OS_LINUX), 1)
	LIBSRCFILES += ${SRCDIR}/os/linux.cc ${SRCDIR}/os/linux_ipcsocket.cc
else ifeq ($(NCCL_OS_WINDOWS), 1)
	LIBSRCFILES += ${SRCDIR}/os/windows.cc ${SRCDIR}/os/windows_ipcsocket.cc
endif

ifneq ($(NVTX), 0)
	LIBSRCFILES += ${SRCDIR}/init_nvtx.cc
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
	#LIBSRCFILES += ${SRCDIR}/enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
# NCCLX: LIBSRCFILES mixes .cc/.cpp/.c/.cu (ynl, ctran/prims) whereas
# upstream assumes only .cc; split it per-language into object lists.
include ../makefiles/ncclx_libobj.mk
BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o)
PARAMBINOBJ     := $(PARAMBINSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d) $(PARAMBINOBJ:%.o=%.d)
# NCCLX link flags (extra -L dirs, prims libs, version script, cudarthook
# Bsymbolic); see makefiles/ncclx_ldflags.mk. Kept as a seam so the forked
# upstream makefile carries only this include.
include ../makefiles/ncclx_ldflags.mk
INCPLUGIN  := ${SRCDIR}/include/plugin

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest


GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h
$(GIT_VERSION_FILE): ALWAYS_REBUILD
	@mkdir -p $(dir $@)
	@./misc/generate_git_version.py $@

# DOCA GPUNetIO definitions
DOCA_HOME        ?= ${SRCDIR}/transport/net_ib/gdaki/doca-gpunetio
DOCA_INC_INSTALL := $(INCDIR)/nccl_device/gin/gdaki/doca_gpunetio
DOCA_OBJDIR      := $(OBJDIR)/transport/net_ib/gdaki/doca-gpunetio
DOCA_INCLUDES    := $(DOCA_HOME)/include/doca_gpunetio_device.h $(wildcard $(DOCA_HOME)/include/common/*.h) $(wildcard $(DOCA_HOME)/include/device/*.cuh)
DOCA_INCTARGETS  := $(DOCA_INCLUDES:$(DOCA_HOME)/include/%=$(DOCA_INC_INSTALL)/%)
INCTARGETS       += $(DOCA_INCTARGETS)
DOCA_LIBSRC      := doca_verbs_qp.cpp doca_verbs_cq.cpp doca_verbs_device_attr.cpp doca_verbs_umem.cpp doca_verbs_srq.cpp doca_verbs_uar.cpp doca_gpunetio.cpp doca_gpunetio_log.cpp doca_gpunetio_high_level.cpp doca_verbs_cuda_wrapper.cpp doca_verbs_mlx5dv_wrapper.cpp doca_verbs_ibv_wrapper.cpp doca_gpunetio_gdrcopy.cpp
DOCA_LIBOBJ      := $(DOCA_LIBSRC:%.cpp=$(DOCA_OBJDIR)/%.o)
LIBOBJ           += $(DOCA_LIBOBJ)

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

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device BASE_DIR=$(BASE_DIR) CONDA_INCLUDE_DIR=$(CONDA_INCLUDE_DIR)

# Empty target to force rebuild
.PHONY: ALWAYS_REBUILD

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
	    $< > $@

# Device objs (from device/Makefile manifest, incl. common.cu.o) are linked in so libnccl.so contains
# device code (e.g. ncclGin_BackendMask<...>) that tests resolve when linking with -lnccl.
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

$(INCDIR)/nccl_device/gin/proxy/%.h: include/nccl_device/gin/proxy/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)/nccl_device/gin/proxy
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

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

# NCCLX-only object rules (.cpp/.c/.cu) for sources pulled in via
# ncclx_sources.mk; kept in a seam so this forked makefile carries only
# the include. Placed here (after OBJDIR is defined) so the $(OBJDIR)/%.o
# pattern targets expand exactly as when these rules were inline.
include ../makefiles/ncclx_rules.mk

$(OBJDIR)/misc/git_version.o: $(GIT_VERSION_FILE)

$(DOCA_OBJDIR)/%.o : $(DOCA_HOME)/src/%.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I$(DOCA_HOME)/src -I$(DOCA_HOME)/include $(CXXFLAGS) -c $< -o $@
	@$(CXX) -I$(DOCA_HOME)/src -I$(DOCA_HOME)/include $(CXXFLAGS) -M $< > $(@:%.o=%.d.tmp)
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
	cp -v -r $(BUILDDIR)/include/* $(PREFIX)/include/
ifeq ($(BUILD_NCCL_CLI),1)
	cp -v $(BUILDDIR)/bin/ncclras $(PREFIX)/bin/
	cp -v $(BUILDDIR)/bin/ncclparam $(PREFIX)/bin/
endif

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ${NCCLDIR}/makefiles/formatting.mk
