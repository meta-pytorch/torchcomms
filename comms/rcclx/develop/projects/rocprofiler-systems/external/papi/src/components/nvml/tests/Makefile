NAME=nvml
include ../../Makefile_comp_tests.target
# -fverbose-asm -Wa,-adhln=benchSANVML.s

PAPI_NVML_ROOT ?= $(PAPI_CUDA_ROOT)
PAPI_NVML_INC ?= $(PAPI_CUDA_ROOT)/include
PAPI_NVML_LIB ?= $(PAPI_CUDA_ROOT)/lib64/stubs

NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc
NVCFLAGS = -I$(PAPI_NVML_INC) -ccbin='$(CC)' $(NVPPC64LEFLAGS)
CUDALIBS = -L$(PAPI_NVML_LIB) -L$(PAPI_CUDA_ROOT)/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++
PAPILIB := ../../../libpapi.a  -ldl

%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = HelloWorld nvml_power_limiting_test

nvml_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 

nvml_power_limiting_test: nvml_power_limiting_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limiting_test nvml_power_limiting_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

nvml_power_limit_read_test: nvml_power_limit_read_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limit_read_test nvml_power_limit_read_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

benchSANVML: benchSANVML.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o benchSANVML benchSANVML.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 

clean:
	rm -f $(TESTS) *.o
	rm -f benchSANVML
	rm -f nvmlcap_plot results*

FORCE:
