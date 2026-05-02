//===------ test_allocate_allocator_align.c ------------------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Tests the allocate directive with allocator and align clause.
// The declarative allocator statement uses the omp_default_mem_alloc handle
// for default memory allocation for "x", aligned to 64-byte alignment via the 
// align clause. Parallel region checks that 64-byte alignment is correct 
// and that the memory can be written to in and the values
// were written correctly, and then frees the memory.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_allocate_allocator_align() {
  int errors = 0;

  int x[N];

#pragma omp allocate(x) allocator(omp_default_mem_alloc) align(64)


#pragma omp target map(from:x[:N])
{	
#pragma omp parallel for simd simdlen(16) aligned(x: 64)
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }
}
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i);
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_allocator_align() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
