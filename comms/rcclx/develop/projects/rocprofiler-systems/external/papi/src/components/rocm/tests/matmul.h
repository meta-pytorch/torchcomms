/**
 * @file   matmul.h
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <hip/hip_runtime.h>

#define BLOCK_DIM_X (16)
#define BLOCK_DIM_Y (16)
#define ROWS        (4096)
#define COLS        (ROWS)

void hip_do_matmul_init(void **handle);
void hip_do_matmul_work(void *handle, hipStream_t stream);
void hip_do_matmul_cleanup(void **handle);

#endif /* End of __MATMUL_H__ */
