NAME=template
include ../../Makefile_comp_tests.target

CFLAGS    = $(OPTFLAGS)
CPPFLAGS += $(INCLUDE)
LDFLAGS  += $(PAPILIB) $(TESTLIB) $(UTILOBJS)

GPUARCH = $(shell rocm_agent_enumerator 2>/dev/null | head -1)
ifeq ($(GPUARCH),)
    GPUARCH = native
endif
GPUFLAGS=--offload-arch=$(GPUARCH) --hip-link --rtlib=compiler-rt -unwindlib=libgcc

TESTS = simple advanced two_eventsets simple_sampling
template_tests: $(TESTS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPTFLAGS) -c -o $@ $<

kernel.o: kernel.cpp
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -D__HIP_ROCclr__=1 -O2 -g -DNDEBUG --offload-arch=$(GPUARCH) -W -Wall -Wextra -Wshadow -o kernel.o -x hip -c kernel.cpp

simple: simple.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG $(GPUFLAGS) simple.o kernel.o -o simple $(LDFLAGS)

advanced: advanced.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG $(GPUFLAGS) advanced.o kernel.o -o advanced $(LDFLAGS)

two_eventsets: two_eventsets.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG $(GPUFLAGS) two_eventsets.o kernel.o -o two_eventsets $(LDFLAGS)

simple_sampling: simple_sampling.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG $(GPUFLAGS) simple_sampling.o kernel.o -o simple_sampling $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
