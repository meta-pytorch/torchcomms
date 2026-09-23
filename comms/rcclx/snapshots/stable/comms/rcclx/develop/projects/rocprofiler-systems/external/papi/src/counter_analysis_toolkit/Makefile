PAPIDIR?=${PAPI_DIR}
USEMPI?=false
LDFLAGS=-L$(PAPIDIR)/lib -lpapi -lm -lpthread -ldl -lrt
INCFLAGS=-I$(PAPIDIR)/include
CFLAGS+=-g -Wall -Wextra
OPT0=-O0
OPT1=-O1
OPT2=-O2
OPT3=-O3
CC=gcc

## Check to see if MPI is used.
ifeq ($(USEMPI),true)
    CC=mpicc
    CFLAGS+=-DUSE_MPI
endif

## Try to auto-detect architecture if it is not set.
ifndef ARCH
    UNAME_M:=$(shell uname -m)

    ARCH=POWER
    ifeq ($(UNAME_M),x86_64)
        ARCH=X86
    endif
    ifeq ($(UNAME_M),aarch64)
        ARCH=ARM
    endif

    $(info    Detected ARCH=$(ARCH))
endif

## Architecture determines vector instruction set.
ifeq ($(ARCH),X86)
    FLOP+=-mfma -DX86
    VECSRC=vec_fma_hp vec_fma_sp vec_fma_dp vec_nonfma_hp vec_nonfma_sp vec_nonfma_dp
    VEC_META=-DAVX128_AVAIL -DAVX256_AVAIL -DAVX512_AVAIL
    VEC128=-mavx -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_128B
    VEC256=-mavx -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_256B
    VEC512=-mavx512f -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_512B
    VEC128_FMA=-mfma4 -mfma -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_128B
    VEC256_FMA=-mfma4 -mfma -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_256B
    VEC512_FMA=-mfma4 -mfma -mavx512f -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_512B
    VEC=-mavx -O0 -DX86
    VEC_FMA=-mfma4 -mfma -O0 -DX86
    VEC_ALL=$(VEC) $(VEC_FMA) -O0 -DX86
    INSTR=-mavx512vl
endif
ifeq ($(ARCH),POWER)
    FLOP+=-maltivec -DPOWER
    VECSRC=vec_fma_hp.o vec_fma_sp.o vec_fma_dp.o vec_nonfma_hp.o vec_nonfma_sp.o vec_nonfma_dp.o
    VEC=-maltivec -DPOWER
    VEC_FMA=-maltivec -DPOWER
    VEC_ALL=$(VEC) -DPOWER
endif
ifeq ($(ARCH),ARM)
    FLOP+=-march=armv8.2-a+fp16 -DARM
    VECSRC=vec_fma_hp.o vec_fma_sp.o vec_fma_dp.o vec_nonfma_hp.o vec_nonfma_sp.o vec_nonfma_dp.o
    VEC=-march=armv8.2-a+fp16 -O0 -DARM
    VEC_FMA=-march=armv8.2-a+fp16 -O0 -DARM
    VEC_ALL=$(VEC) -O0 -DARM
endif

all: branch.o d_cache eventstock.o flops i_cache instr vector
	make cat_collect PAPIDIR=$(PAPIDIR)

d_cache: timing_kernels.o prepareArray.o compar.o dcache.o

i_cache: icache.o icache_seq_kernel_0.o

vector: weak_symbols.o vec.o vec_scalar_verify.o $(VECSRC)

branch.o: branch.c branch.h
	$(CC) $(OPT0) $(CFLAGS) $(INCFLAGS) -c branch.c -o branch.o

timing_kernels.o: timing_kernels.c timing_kernels.h
	$(CC) $(OPT2) $(CFLAGS) -fopenmp $(INCFLAGS) -c timing_kernels.c -o timing_kernels.o

prepareArray.o: prepareArray.c prepareArray.h
	$(CC) $(OPT2) $(CFLAGS) -c prepareArray.c -o prepareArray.o

compar.o: compar.c
	$(CC) $(CFLAGS) $(OPT2) -c compar.c -o compar.o

dcache.o: dcache.c dcache.h
	$(CC) $(CFLAGS) $(OPT2) -fopenmp $(INCFLAGS) -c dcache.c -o dcache.o

eventstock.o: eventstock.c eventstock.h
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c eventstock.c -o eventstock.o

flops: flops.c flops.h cat_arch.h
	$(CC) $(CFLAGS) $(FLOP) $(OPT1) $(INCFLAGS) -c flops.c -o flops.o

icache.o: icache.c icache.h
	bash gen_seq_dlopen.sh
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache.c -o icache.o

icache_seq_kernel_0.o: icache_seq.c icache_seq.h
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache_seq.c -o icache_seq.o
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -fPIC -c icache_seq_kernel.c -o icache_seq_kernel_0.o
	$(CC) $(CFLAGS) $(OPT0) -shared -o icache_seq_kernel_0.so icache_seq_kernel_0.o
	bash replicate.sh
	rm icache_seq_kernel_0.o

instr: instructions.c instr.h
	-$(CC) -c $(CFLAGS) $(OPT2) -ftree-vectorize $(FLOP) $(INSTR) $(INCFLAGS) instructions.c -o instructions.o

weak_symbols.o: weak_symbols.c vec.h
	-$(CC) -c $(CFLAGS) weak_symbols.c

vec.o: vec.c vec.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) -D$(ARCH) $(VEC_META) vec.c

vec_scalar_verify.o: vec_scalar_verify.c vec_scalar_verify.h cat_arch.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_scalar_verify.c

vec_fma_hp.o: vec_fma_hp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_FMA) vec_fma_hp.c

vec_fma_hp: vec_fma_hp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_hp.c -o vec_fma_hp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_hp.c -o vec_fma_hp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_hp.c -o vec_fma_hp-512B.o

vec_fma_sp.o: vec_fma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_FMA) vec_fma_sp.c

vec_fma_sp: vec_fma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_sp.c -o vec_fma_sp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_sp.c -o vec_fma_sp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_sp.c -o vec_fma_sp-512B.o

vec_fma_dp.o: vec_fma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_FMA) vec_fma_dp.c

vec_fma_dp: vec_fma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_dp.c -o vec_fma_dp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_dp.c -o vec_fma_dp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_dp.c -o vec_fma_dp-512B.o

vec_nonfma_hp.o: vec_nonfma_hp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC) vec_nonfma_hp.c

vec_nonfma_hp: vec_nonfma_hp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_hp.c -o vec_nonfma_hp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_hp.c -o vec_nonfma_hp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_hp.c -o vec_nonfma_hp-512B.o

vec_nonfma_sp.o: vec_nonfma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC) vec_nonfma_sp.c

vec_nonfma_sp: vec_nonfma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_sp.c -o vec_nonfma_sp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_sp.c -o vec_nonfma_sp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_sp.c -o vec_nonfma_sp-512B.o

vec_nonfma_dp.o: vec_nonfma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC) vec_nonfma_dp.c

vec_nonfma_dp: vec_nonfma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_dp.c -o vec_nonfma_dp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_dp.c -o vec_nonfma_dp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_dp.c -o vec_nonfma_dp-512B.o

cat_collect:
	$(CC) $(CFLAGS) -fopenmp $(INCFLAGS) main.c $(wildcard *.o) -o cat_collect $(LDFLAGS)

clean:
	rm -f *.o

realclean:
	rm -f cat_collect *.o *.so icache_seq.c icache_seq.h icache_seq_kernel.c

