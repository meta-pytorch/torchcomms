# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
NVCCFLAGS  = -O2 -std=c++17 -arch=$(CUDA_ARCH)

BLOCKED_SRCS  = memcpyasync_blocked.cu busy_kernel.cu first_time_kernel.cu
DEADLOCK_SRCS = memcpyasync_deadlock.cu persistent_kernel.cu first_time_kernel.cu

all: memcpyasync_blocked memcpyasync_deadlock

memcpyasync_blocked: $(BLOCKED_SRCS)
	$(NVCC) $(NVCCFLAGS) $(BLOCKED_SRCS) -o $@

memcpyasync_deadlock: $(DEADLOCK_SRCS)
	$(NVCC) $(NVCCFLAGS) $(DEADLOCK_SRCS) -o $@

# Demo 1: bounded host-side block. Lock contention demo.
run-blocked: memcpyasync_blocked
	CUDA_MODULE_LOADING=LAZY ./memcpyasync_blocked

# Same binary, eager loading: no lock contention -> no host-side stall.
run-blocked-eager: memcpyasync_blocked
	CUDA_MODULE_LOADING=EAGER ./memcpyasync_blocked

# Demo 2: true deadlock. Process wedges; watchdog kills it after 30s.
run-deadlock: memcpyasync_deadlock
	CUDA_MODULE_LOADING=LAZY ./memcpyasync_deadlock

# Same binary, eager loading: no deadlock -> process completes in <1s.
run-deadlock-eager: memcpyasync_deadlock
	CUDA_MODULE_LOADING=EAGER ./memcpyasync_deadlock

clean:
	rm -f memcpyasync_blocked memcpyasync_deadlock

.PHONY: all run-blocked run-blocked-eager run-deadlock run-deadlock-eager clean
