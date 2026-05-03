NAME=nvml
include ../../Makefile_comp_tests
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

nvmlcap_plot: nvmlcap_plot.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvmlcap_plot nvmlcap_plot.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)


clean:
	rm -f nvmlcap_plot nvmlcap_plot.o results*

