//===------ test_allocate_allocators.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Tests the allocate directive with allocator clause, based on the OpenMP
// 5.0 example for allocators. The allocator testing first creates
// an allocator, with 64-byte alignment and the default memory space,
// then checks that 64-byte alignment is correct and that the memory can
// be written to in the parallel region. The tests checks that the values
// were written correctly, and then frees the memory and deletes the
// allocator.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ompvv.h"

#define N 1024

int test_allocate_allocator() {
  int errors = 0;

  int n = N;
  omp_memspace_handle_t x_memspace = omp_default_mem_space;
  omp_alloctrait_t x_traits[1] = {{omp_atk_alignment, 64}};
  omp_allocator_handle_t x_alloc = omp_init_allocator(x_memspace, 1, x_traits);
  {
    // x and y must be in an inner scope as their lifetime must not exceed the
    // lifetime of the allocator associated with x_alloc; i.e. their lifetime
    // must have ended before omp_destroy_allocator is called.
    //
    // The following two lines allocate the pointer 'x' itself
    // (i.e. sizeof(void*) bytes) and 'y'. The omp_alloc allocates then
    // space for the actual data stored in 'x'.
    int* x, y[n];
#pragma omp allocate(x,y) allocator(x_alloc)

    x = (int *)omp_alloc(N * sizeof(int), x_alloc);

    // OMPVV_TEST_AND_SET_VERBOSE diagnostic uses printf, which
    // causes compiler warnigns for '% '. Hence:
    OMPVV_TEST_AND_SET(errors, ((intptr_t) &x) % 64 != 0);
    OMPVV_TEST_AND_SET(errors, ((intptr_t) &y) % 64 != 0);
    OMPVV_TEST_AND_SET(errors, ((intptr_t) x) % 64 != 0);
    OMPVV_ERROR_IF(((intptr_t) &x) % 64 != 0,
                   "Condition (intptr_t) &x) %% 64 != 0 failed")
    OMPVV_ERROR_IF(((intptr_t) &y) % 64 != 0,
                   "Condition (intptr_t) &y) %% 64 != 0 failed")
    OMPVV_ERROR_IF(((intptr_t) x) % 64 != 0,
                   "Condition (intptr_t) x) %% 64 != 0 failed")

#pragma omp parallel for simd simdlen(16) aligned(x, y: 64)
    for (int i = 0; i < N; i++) {
      x[i] = i;
      y[i] = 3*i;
    }

    for (int i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i);
      OMPVV_TEST_AND_SET_VERBOSE(errors, y[i] != 3*i);
    }

    omp_free(x, x_alloc);
  }
  omp_destroy_allocator(x_alloc);

  return errors;
}

int main() {

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_allocator() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
