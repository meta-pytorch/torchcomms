NAME=intel_gpu
include ../../Makefile_comp_tests.target
CFLAGS += -DDEBUG  -I../../../testlib

INTEL_L0_HEADERS ?=/usr/include
INTEL_L0_LIB64 ?=/usr/lib
CPPLDFLAGS+=-ldl
GPULIB=-L$(INTEL_L0_LIB64) -lze_loader

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -g -fPIC -c -o $@ $<

%.o:%.cc
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(OPTFLAGS) $(INCLUDE) -I$(INTEL_L0_HEADERS) -DENABLE_PAPI -g -fPIC -c -o $@ $<

TESTS = gpu_metric_read gpu_metric_list gpu_query_gemm gpu_gemm

intel_gpu_tests: $(TESTS)

gpu_metric_list: gpu_metric_list.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o gpu_metric_list gpu_metric_list.o $(PAPILIB) $(LDFLAGS)

gpu_metric_read: gpu_metric_read.o gpu_common_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o gpu_metric_read gpu_metric_read.o gpu_common_utils.o $(PAPILIB) $(LDFLAGS)

gpu_gemm.o: gpu_query_gemm.cc
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(OPTFLAGS) $(INCLUDE) -I$(INTEL_L0_HEADERS) -g -fPIC -c -o gpu_gemm.o gpu_query_gemm.cc

gpu_gemm: gpu_gemm.o gpu_common_utils.o $(PAPILIB)
	$(CXX) $(CFLAGS) -o gpu_gemm gpu_gemm.o gpu_common_utils.o $(GPULIB) $(PAPILIB) $(CPPLDFLAGS) $(LDFLAGS)

gpu_query_gemm: gpu_query_gemm.o gpu_common_utils.o $(PAPILIB)
	$(CXX) $(CFLAGS) -o gpu_query_gemm gpu_query_gemm.o gpu_common_utils.o $(GPULIB) $(PAPILIB) $(CPPLDFLAGS) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
