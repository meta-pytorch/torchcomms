#ifndef _TIMING_KERNELS_
#define _TIMING_KERNELS_

#include <stdint.h>

#include "caches.h"

#define N_1   p = (uintptr_t *)*p;
#define N_2   N_1 N_1
#define N_16  N_2 N_2 N_2 N_2 N_2 N_2 N_2 N_2
#define N_128 N_16 N_16 N_16 N_16 N_16 N_16 N_16 N_16

// NW_1 reads a pointer from an array and then modifies the pointer
// and stores it back to the same place in the array. This way it
// causes exactly one read and one write operation in memory, assuming
// that the variable "p_prime" resides in a register. The exact steps
// are the following:
// 1. Reads the element pointed to by "p" into "p_prime". 
//    This element is almost a pointer to the next element in the chain, but the least significant bit might be set to 1.
// 2. Flip the least significant bit of "p_prime" and store it back into the buffer.
// 3. Clear the least significant bit of "p_prime" and store the result in "p".
#define NW_1   {p_prime = *p; *p = p_prime ^ 0x1; p = (uintptr_t *)(p_prime & (~0x1));}
#define NW_2   NW_1 NW_1
#define NW_16  NW_2 NW_2 NW_2 NW_2 NW_2 NW_2 NW_2 NW_2
#define NW_128 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16

#define CACHE_READ_ONLY  0x0
#define CACHE_READ_WRITE 0x1

run_output_t probeBufferSize(long long active_buf_len, long long line_size, float pageCountPerBlock, int pattern, long long llc_size, uintptr_t **v, uintptr_t *rslt, int detect_size, int mode, int ONT);
void error_handler(int e, int line);

#endif
