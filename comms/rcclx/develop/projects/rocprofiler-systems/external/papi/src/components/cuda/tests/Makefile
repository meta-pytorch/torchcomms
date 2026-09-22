NAME=cuda
include ../../Makefile_comp_tests.target

PAPI_CUDA_ROOT ?= $(shell dirname $(shell dirname $(shell which nvcc)))

TESTS = HelloWorld simpleMultiGPU \
		pthreads cudaOpenMP concurrent_profiling \
		test_multi_read_and_reset test_multipass_event_fail \
		test_2thr_1gpu_not_allowed

TESTS_NOCTX = concurrent_profiling_noCuCtx pthreads_noCuCtx \
			  cudaOpenMP_noCuCtx HelloWorld_noCuCtx \
			  simpleMultiGPU_noCuCtx

NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc
NVCC_VERSION := $(shell $(NVCC) --version | grep -oP '(?<=release )\d+\.\d+')

PAPI_FLAG = -DPAPI    # Comment this line for tests to run without PAPI profiling
NVCFLAGS = -g -ccbin='$(CC)' $(PAPI_FLAG)

# Check to see if the Cuda Toolkit version is either 12.8 or 12.9 such that
# we can add -Wno-deprecated-gpu-targets to avoid compilation warning
CUDA_TOOLKIT_EQUAL_TO_12_8_OR_12_9 := $(shell echo "$(NVCC_VERSION)" | awk '{print ($$1 == 12.8 || $$1 == 12.9)}')
ifeq ($(CUDA_TOOLKIT_EQUAL_TO_12_8_OR_12_9), 1)
    NVCFLAGS += -Wno-deprecated-gpu-targets
endif

# Check to see if we are using a Cuda Toolkit version greater than or equal to 13
# as the API call for cuCtxCreate changed at this version
CUDA_TOOLKIT_GE_13 := $(shell echo "$(NVCC_VERSION) 13.0" | awk '{print $$1 >= $$2}')
CUDA_CPPFLAGS :=
ifeq ($(CUDA_TOOLKIT_GE_13), 1)
    CUDA_CPPFLAGS += -DCUDA_TOOLKIT_GE_13
endif

ifeq ($(BUILD_SHARED_LIB),yes)
	NVCFLAGS += -Xcompiler -fpic
endif
CFLAGS += -g $(PAPI_FLAG)
INCLUDE += -I$(PAPI_CUDA_ROOT)/include
CUDALIBS = -L$(PAPI_CUDA_ROOT)/lib64 -lcudart -lcuda

cuda_tests: $(TESTS) $(TESTS_NOCTX)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) $(CUDA_CPPFLAGS) -E -c -o $@ $<

test_multi_read_and_reset: test_multi_read_and_reset.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o test_multi_read_and_reset test_multi_read_and_reset.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling: concurrent_profiling.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -pthread -o concurrent_profiling concurrent_profiling.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling_noCuCtx: concurrent_profiling_noCuCtx.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -pthread -o concurrent_profiling_noCuCtx concurrent_profiling_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads: pthreads.o
	$(CXX) $(CFLAGS) -pthread -o pthreads pthreads.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads_noCuCtx: pthreads_noCuCtx.o
	$(CXX) $(CFLAGS) -pthread -o pthreads_noCuCtx pthreads_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP: cudaOpenMP.o
	$(CXX) $(CFLAGS) -o cudaOpenMP cudaOpenMP.o -lgomp -fopenmp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP_noCuCtx: cudaOpenMP_noCuCtx.o
	$(CXX) $(CFLAGS) -o cudaOpenMP_noCuCtx cudaOpenMP_noCuCtx.o -lgomp -fopenmp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_multipass_event_fail: test_multipass_event_fail.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o test_multipass_event_fail test_multipass_event_fail.o $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(LDFLAGS) $(CUDALIBS)

test_2thr_1gpu_not_allowed: test_2thr_1gpu_not_allowed.o
	$(CXX) $(CFLAGS) -pthread -o test_2thr_1gpu_not_allowed test_2thr_1gpu_not_allowed.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_noCuCtx: HelloWorld_noCuCtx.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o HelloWorld_noCuCtx HelloWorld_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o simpleMultiGPU simpleMultiGPU.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_noCuCtx: simpleMultiGPU_noCuCtx.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o simpleMultiGPU_noCuCtx simpleMultiGPU_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f *.o $(TESTS) $(TESTS_NOCTX)
