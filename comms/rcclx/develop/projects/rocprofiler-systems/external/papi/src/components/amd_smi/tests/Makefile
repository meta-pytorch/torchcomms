NAME=amd_smi
include ../../Makefile_comp_tests.target

# ROCm + PAPI build settings
# Set PAPI_AMDSMI_ROOT to your ROCm install prefix (e.g., /opt/rocm or /opt/rocm-6.4.0).
# Used to find AMD SMI / HIP / ROCm headers and libraries. You can override on the command line.
PAPI_AMDSMI_ROOT ?= /opt/rocm-6.4.0
HIP_PATH := $(PAPI_AMDSMI_ROOT)
HIPCC := $(HIP_PATH)/bin/hipcc
CC ?= cc

# Toolchain / flags
CFLAGS   ?= -DPAPI_NUM_COMP=3
CSTD     ?= -std=c11
HIPFLAGS ?= $(CFLAGS) -std=c++17
OPTFLAGS ?= -O2

# Libraries in the PAPI tree (override if your layout differs)
UTILOBJS ?= ../../../testlib/libtestlib.a
PAPILIB  ?= ../../../libpapi.a

# Include paths (adjust if your PAPI install lives elsewhere)
BASE_INCLUDES_FROM_LOG ?= -I. -I../../.. -I../../../testlib -I../../../validation_tests
PAPI_DIR_INCLUDE ?= ../../../include

INCLUDE_PAPI := $(BASE_INCLUDES_FROM_LOG) -I$(PAPI_DIR_INCLUDE)
INCLUDE_AMDSMI := -I$(PAPI_AMDSMI_ROOT)/include -I$(PAPI_AMDSMI_ROOT)/include/rocm_smi
INCLUDE_HIP := -I$(PAPI_AMDSMI_ROOT)/include/hip \
               -I$(PAPI_AMDSMI_ROOT)/include/hsa \
               -I$(PAPI_AMDSMI_ROOT)/include/rocprofiler \
               -I$(PAPI_AMDSMI_ROOT)/include/rocblas

EFFECTIVE_INCLUDE_C := $(INCLUDE_PAPI) $(INCLUDE_AMDSMI)
EFFECTIVE_INCLUDE_HIP := $(EFFECTIVE_INCLUDE_C) $(INCLUDE_HIP)

# Linker flags
COMMON_LDFLAGS  = $(UTILOBJS) $(PAPILIB) -L$(PAPI_AMDSMI_ROOT)/lib -L$(PAPI_AMDSMI_ROOT)/lib64 $(LDFLAGS) -pthread
LDFLAGS_AMDSMI  = $(COMMON_LDFLAGS) -lamd_smi
LDFLAGS_GEMM    = $(COMMON_LDFLAGS) -lrocblas

# AMD SMI tests to build with 'make tests'
TESTS_AMDSMI := amdsmi_energy_monotonic amdsmi_ctx_conflict amdsmi_set_test

# Default target
TESTS = amdsmi_hello amdsmi_basics amdsmi_gemm $(TESTS_AMDSMI)
amd_smi_tests: $(TESTS)

# ------------------------
# Compile rules
# ------------------------

# .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -c $< -o $@

# amdsmi_gemm.c is compiled as HIP
amdsmi_gemm.o: amdsmi_gemm.c
	$(HIPCC) $(HIPFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_HIP) -x hip -c $< -o $@

# ------------------------
# Link rules
# ------------------------

# amdsmi_hello replaces the older amdsmi_example
amdsmi_hello: amdsmi_hello.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -o $@ amdsmi_hello.o $(LDFLAGS_AMDSMI)

# Back-compat alias: allow 'make amdsmi_example' to build amdsmi_hello
amdsmi_example: amdsmi_hello
	@true

amdsmi_basics: amdsmi_basics.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -o $@ amdsmi_basics.o $(LDFLAGS_AMDSMI)

# GEMM sample linked against rocBLAS
amdsmi_gemm: amdsmi_gemm.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(HIPFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_HIP) -o $@ amdsmi_gemm.o $(LDFLAGS_GEMM)

# Tests
amdsmi_energy_monotonic: amdsmi_energy_monotonic.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -o $@ amdsmi_energy_monotonic.o $(LDFLAGS_AMDSMI)

amdsmi_ctx_conflict: amdsmi_ctx_conflict.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -o $@ amdsmi_ctx_conflict.o $(LDFLAGS_AMDSMI)

amdsmi_set_test: amdsmi_set_test.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE_C) -o $@ amdsmi_set_test.o $(LDFLAGS_AMDSMI)

$(UTILOBJS):
	$(MAKE) -C ../../../testlib libtestlib.a

# Convenience meta-target
tests: $(TESTS_AMDSMI)

# ------------------------
# Cleanup and diagnostics
# ------------------------

clean:
	@echo "Cleaning up..."
	rm -f \
	  amdsmi_hello amdsmi_hello.o \
	  amdsmi_basics amdsmi_basics.o \
	  amdsmi_gemm amdsmi_gemm.o \
	  amdsmi_energy_monotonic amdsmi_energy_monotonic.o \
	  amdsmi_ctx_conflict amdsmi_ctx_conflict.o \
	  amdsmi_set_test amdsmi_set_test.o \
	  *.exe

checkpath:
	@echo "--- Variables ---"
	@echo "PAPI_AMDSMI_ROOT = $(PAPI_AMDSMI_ROOT)"
	@echo "CC = $(CC)"
	@echo "HIPCC = $(HIPCC)"
	@echo "INCLUDE_PAPI = $(INCLUDE_PAPI)"
	@echo "INCLUDE_AMDSMI = $(INCLUDE_AMDSMI)"
	@echo "INCLUDE_HIP = $(INCLUDE_HIP)"
	@echo "EFFECTIVE_INCLUDE_C = $(EFFECTIVE_INCLUDE_C)"
	@echo "EFFECTIVE_INCLUDE_HIP = $(EFFECTIVE_INCLUDE_HIP)"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "CSTD = $(CSTD)"
	@echo "HIPFLAGS = $(HIPFLAGS)"
	@echo "OPTFLAGS = $(OPTFLAGS)"
	@echo "UTILOBJS = $(UTILOBJS)"
	@echo "PAPILIB = $(PAPILIB)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "COMMON_LDFLAGS = $(COMMON_LDFLAGS)"
	@echo "LDFLAGS_AMDSMI = $(LDFLAGS_AMDSMI)"
	@echo "LDFLAGS_GEMM = $(LDFLAGS_GEMM)"
	@echo "TESTS_AMDSMI = $(TESTS_AMDSMI)"

.PHONY: all clean checkpath tests \
	amdsmi_hello amdsmi_example amdsmi_basics amdsmi_gemm \
	amdsmi_energy_monotonic amdsmi_ctx_conflict amdsmi_set_test
