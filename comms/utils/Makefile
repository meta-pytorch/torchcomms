# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# The commsutils package: libcommsutils.so.1 and libobservatory.so.1, the
# one-copy-per-process libraries shared by NCCLX and MCCL.
#
# libcommsutils.so.1 is comms/utils -- NCCL's configuration (cvars) table, the
# logger, the trainer context, memtrace, colltrace and the shared helpers.
# libobservatory.so.1 is comms/observatory (see its README for why it must be
# a single instance). libnccl.so, libctran.so and libmccl.so link both as
# prebuilt libraries; nothing else builds them.
#
# Inputs: BASE_DIR (the directory containing comms/), BUILDDIR (outputs go to
# $(BUILDDIR)/lib and $(BUILDDIR)/obj/commsutils), PREFIX (where `install`
# puts the libraries, $(PREFIX)/lib), CUDA_HOME, NVCC_GENCODE,
# CONDA_INCLUDE_DIR / CONDA_LIB_DIR, THIRD_PARTY_LDFLAGS (the folly/abseil link
# line, with gflags and glog as static archives -- see
# comms/github/build_commsutils.sh, which derives it), CUDARTLIB, PYTHON (runs
# cvars/extractcvars.py). Only the libraries are installed: consumers compile
# against the headers of the source tree they were built from.

BASE_DIR  ?= $(abspath ../..)
BUILDDIR  ?= $(BASE_DIR)/build/commsutils
PREFIX    ?= $(CONDA_PREFIX)
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  ?= $(CUDA_HOME)/include
CUDA_LIB  ?= $(CUDA_HOME)/lib64
NVCC      ?= $(CUDA_HOME)/bin/nvcc
NVCC_GENCODE ?=
CUDARTLIB ?= cudart_static
CXX       ?= g++
PYTHON    ?= python3
# The sources need C++20 (defaulted comparisons, concepts). Callers that
# forward CXXSTD from their own environment may hand over an empty string,
# which `?=` would accept as a value, so test for emptiness instead.
ifeq ($(strip $(CXXSTD)),)
CXXSTD    := -std=c++20
endif
THIRD_PARTY_LDFLAGS ?=

LIBDIR    := $(BUILDDIR)/lib
OBJDIR    := $(BUILDDIR)/obj/commsutils
UTILS_DIR := $(BASE_DIR)/comms/utils
OBSERVATORY_DIR := $(BASE_DIR)/comms/observatory
NAME      := libcommsutils.so
SONAME    := $(NAME).1
OBSERVATORY_NAME   := libobservatory.so
OBSERVATORY_SONAME := $(OBSERVATORY_NAME).1

INCLUDES := -I$(BASE_DIR) -I$(CUDA_INC) -I$(CUDA_INC)/cccl
ifneq ($(CONDA_INCLUDE_DIR),)
INCLUDES += -I$(CONDA_INCLUDE_DIR)
endif
DEFINES  := -DMOCK_SCUBA_DATA -DFMT_HEADER_ONLY=1 \
            -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO -DSPDLOG_FMT_EXTERNAL

# The whole of comms/utils is the API: default visibility, and it goes last
# because the flags inherited from the environment may end in
# -fvisibility=hidden. The third-party code stays private: --exclude-libs for
# the statically linked archives, and $(EXPORT_MAP) for what these objects
# instantiate from folly/fmt/spdlog/glog headers (see the map for why).
CXXFLAGS  := $(CXXFLAGS) -fPIC -O3 -g $(CXXSTD) \
             -Wall -Wno-unused-function -Wno-sign-compare \
             $(DEFINES) $(INCLUDES) -fvisibility=default
NVCUFLAGS := -ccbin $(CXX) $(NVCC_GENCODE) $(CXXSTD) --expt-extended-lambda \
             -Xptxas -maxrregcount=96 -Xfatbin -compress-all -O3 \
             $(DEFINES) $(INCLUDES) \
             -Xcompiler -fPIC -Xcompiler -fvisibility=default
# STB_GNU_UNIQUE symbols (function-local statics of inline functions and
# templates) are merged process-wide regardless of visibility or RTLD_LOCAL,
# so emit them as ordinary weak symbols where the compiler has the flag (GCC)
# and let $(EXPORT_MAP) localize them like the rest.
NO_GNU_UNIQUE := $(shell $(CXX) -fno-gnu-unique -fsyntax-only -x c++ /dev/null >/dev/null 2>&1 && echo -fno-gnu-unique)
CXXFLAGS  += $(NO_GNU_UNIQUE)
NVCUFLAGS += $(if $(NO_GNU_UNIQUE),-Xcompiler $(NO_GNU_UNIQUE))
EXPORT_MAP := $(UTILS_DIR)/libcommsutils.map

# cvars/nccl_cvars.{cc,h} are generated from nccl_cvars.yaml; the .cc is named
# explicitly because $(wildcard) runs at parse time, before the rule below has
# produced it on a clean tree. One generator run produces both files, so the
# rule is keyed on a stamp: a plain two-target rule would run the generator
# once per target, concurrently under -j, with both writing the same files.
CVARS_DIR   := $(UTILS_DIR)/cvars
CVARS_CC    := $(CVARS_DIR)/nccl_cvars.cc
CVARS_H     := $(CVARS_DIR)/nccl_cvars.h
CVARS_STAMP := $(OBJDIR)/nccl_cvars.stamp

SRC := $(wildcard $(UTILS_DIR)/*.cc)
SRC += $(wildcard $(UTILS_DIR)/trainer/*.cc)
SRC += $(wildcard $(UTILS_DIR)/logger/*.cc)
SRC += $(wildcard $(UTILS_DIR)/memtrace/*.cc)
SRC += $(wildcard $(UTILS_DIR)/colltrace/*.cc)
SRC += $(wildcard $(UTILS_DIR)/colltrace/plugins/*.cc)
SRC += $(wildcard $(UTILS_DIR)/hrdw_ring_buffer/*.cc)
SRC += $(filter-out $(CVARS_CC),$(wildcard $(CVARS_DIR)/*.cc)) $(CVARS_CC)
CUSRC := $(UTILS_DIR)/hrdw_ring_buffer/GpuClockCalibration.cu
CUSRC += $(UTILS_DIR)/colltrace/HRDWRingBufferInstantiations.cu

OBJ := $(SRC:$(BASE_DIR)/%.cc=$(OBJDIR)/%.o) $(CUSRC:$(BASE_DIR)/%.cu=$(OBJDIR)/%.cu.o)

OBSERVATORY_SRC  := $(OBSERVATORY_DIR)/colltrace/LifecycleFeedRegistry.cc
OBSERVATORY_HDRS := $(wildcard $(OBSERVATORY_DIR)/colltrace/*.h)

all: $(LIBDIR)/$(SONAME) $(LIBDIR)/$(OBSERVATORY_SONAME)

$(CVARS_STAMP): $(CVARS_DIR)/nccl_cvars.yaml $(CVARS_DIR)/nccl_cvars.cc.in $(CVARS_DIR)/extractcvars.py
	@printf "Generating %-35s > %s\n" nccl_cvars.yaml $(CVARS_CC)
	@mkdir -p $(dir $@)
	cd $(CVARS_DIR) && NCCL_CVARS_OUTPUT_DIR=$(CVARS_DIR) $(PYTHON) extractcvars.py
	touch $@

$(CVARS_CC) $(CVARS_H): $(CVARS_STAMP) ;

# Order-only on the generated header: many comms/utils headers include it, and
# the .d files that would otherwise record that do not exist on a first build.
$(OBJDIR)/%.o: $(BASE_DIR)/%.cc | $(CVARS_H)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJDIR)/%.cu.o: $(BASE_DIR)/%.cu | $(CVARS_H)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) -c $< -o $@

# Default visibility and no -Bsymbolic: consumers must resolve these symbols
# here rather than locally. The export map only hides third-party code.
$(LIBDIR)/$(OBSERVATORY_SONAME): $(OBSERVATORY_SRC) $(OBSERVATORY_HDRS) $(EXPORT_MAP)
	@printf "Linking    %-35s > %s\n" $(OBSERVATORY_SONAME) $@
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -I$(BASE_DIR) -shared \
	    -Wl,--no-undefined -Wl,--version-script=$(EXPORT_MAP) \
	    -Wl,-soname,$(OBSERVATORY_SONAME) -o $@ $<
	ln -sf $(OBSERVATORY_SONAME) $(LIBDIR)/$(OBSERVATORY_NAME)

# -lstdc++ ahead of the third-party archives: libfolly.a carries folly's
# exception tracer, which defines __cxa_throw and friends and finds the real
# ones with dlsym(RTLD_NEXT). The first archive member satisfying a `throw`
# wins at link time, so without this the tracer is pulled in as a local
# interposer; RTLD_NEXT then misses libstdc++ (already loaded, e.g. by
# Python) and the first exception thrown in here crashes the process.
$(LIBDIR)/$(SONAME): $(OBJ) $(LIBDIR)/$(OBSERVATORY_SONAME) $(EXPORT_MAP)
	@printf "Linking    %-35s > %s\n" $(SONAME) $@
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-undefined -Wl,--exclude-libs,ALL \
	    -Wl,--version-script=$(EXPORT_MAP) -Wl,-soname,$(SONAME) -o $@ $(OBJ) -lstdc++ \
	    -L$(CUDA_LIB) $(if $(CONDA_LIB_DIR),-L$(CONDA_LIB_DIR)) -L$(LIBDIR) -lobservatory \
	    -l$(CUDARTLIB) -latomic -lpthread -lm -lrt -ldl $(THIRD_PARTY_LDFLAGS) -Wl,-rpath,'$$ORIGIN'
	ln -sf $(SONAME) $(LIBDIR)/$(NAME)

install: all
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 $(LIBDIR)/$(SONAME) $(LIBDIR)/$(OBSERVATORY_SONAME) $(DESTDIR)$(PREFIX)/lib/
	ln -sf $(SONAME) $(DESTDIR)$(PREFIX)/lib/$(NAME)
	ln -sf $(OBSERVATORY_SONAME) $(DESTDIR)$(PREFIX)/lib/$(OBSERVATORY_NAME)

clean:
	rm -rf $(OBJDIR) $(LIBDIR)/$(NAME) $(LIBDIR)/$(SONAME) \
	    $(LIBDIR)/$(OBSERVATORY_NAME) $(LIBDIR)/$(OBSERVATORY_SONAME)

-include $(filter %.d,$(OBJ:.o=.d))

.PHONY: all install clean
