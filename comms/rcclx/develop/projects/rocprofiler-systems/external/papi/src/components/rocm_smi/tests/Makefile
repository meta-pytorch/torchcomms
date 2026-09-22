NAME=rocm_smi
include ../../Makefile_comp_tests.target

path_to_rocblas =
rocm_smi_included_in_path := $(shell echo "$(PAPI_ROCMSMI_ROOT)" | grep --ignore-case --fixed-strings rocm_smi)
# Per the README, for ROCm versions 5.2 and newer users must set PAPI_ROCMSMI_ROOT to /opt/rocm
ifeq ($(rocm_smi_included_in_path),)
    HIPCC ?= $(PAPI_ROCMSMI_ROOT)/bin/hipcc
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include/rocm_smi
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include/hip
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include/hsa
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include/rocprofiler
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include/rocblas
    path_to_rocblas := $(PAPI_ROCMSMI_ROOT)
# Per the README, for ROCM versions prior to 5.2 users must set PAPI_ROCMSMI_ROOT to /opt/rocm/rocm_smi
else
    HIPCC ?= $(PAPI_ROCMSMI_ROOT)/../bin/hipcc
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/../include
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/../include/hip
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/../include/hsa
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/../include/rocprofiler
    INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/../include/rocblas
    path_to_rocblas := $(PAPI_ROCMSMI_ROOT)/..
endif
ROCM_SMI_LDFLAGS := $(LDFLAGS) -ldl -g -pthread

%.o:%.c
	@echo "INCLUDE=" $(INCLUDE)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

%.o:%.cpp
	@echo "INCLUDE=" $(INCLUDE)
	@echo "CFLAGS=" $(CFLAGS)
	g++ $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = rocm_command_line rocm_smi_all power_monitor_rocm rocm_smi_writeTests
TESTS_LONG = rocmsmi_example

rocm_smi_tests: $(TESTS)
rocm_smi_tests_long: $(TESTS_LONG)

# Note: We compile .o separately from the executable link; some versions of hipcc
#       have trouble managing libraries if we try to do both in a single step.

rocm_command_line.o: rocm_command_line.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@ 

rocm_command_line: rocm_command_line.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(ROCM_SMI_LDFLAGS)

rocm_smi_all.o: rocm_smi_all.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocm_smi_all: rocm_smi_all.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(ROCM_SMI_LDFLAGS)

power_monitor_rocm.o: power_monitor_rocm.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

power_monitor_rocm: power_monitor_rocm.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(ROCM_SMI_LDFLAGS)

rocmsmi_example.o: rocmsmi_example.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocmsmi_example: rocmsmi_example.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(ROCM_SMI_LDFLAGS) -L$(PAPI_ROCM_ROOT)/lib/rocblas -lrocblas

rocm_smi_writeTests.o: rocm_smi_writeTests.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocm_smi_writeTests: rocm_smi_writeTests.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(ROCM_SMI_LDFLAGS)

clean:
	rm -f $(TESTS) $(TESTS_LONG) *.o

checkpath: 
	echo PAPI_ROCM_ROOT = $(PAPI_ROCM_ROOT)
	echo HIP_PATH = $(HIP_PATH)
	echo HIPCC = $(HIPCC)
	echo INCLUDE = $(INCLUDE)
	echo ROCM_SMI_LDFLAGS = $(ROCM_SMI_LDFLAGS)
	echo CFLAGS = $(CFLAGS)

