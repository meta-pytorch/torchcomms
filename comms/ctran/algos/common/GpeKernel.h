// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef GPE_KERNEL_H_
#define GPE_KERNEL_H_

// Indicate a free flag
#define KERNEL_UNSET (0)
// Indicate the kernel has been scheduled and the flag is inuse
#define KERNEL_SCHEDULED (1)
// Indicate kernel has started, so that GPE thread can start
#define KERNEL_STARTED (2)
// Indicate GPE thread has finished, so that kernel can terminate
#define KERNEL_TERMINATE (3)
// Indicate kernel has started and will exit without waiting on GPE completion,
// so that GPE thread can free the flag immediately after consumed the start
// signal from kernel.
#define KERNEL_STARTED_AND_EXIT (4)
// host abort flag
#define KERNEL_HOST_ABORT (5)

#endif
