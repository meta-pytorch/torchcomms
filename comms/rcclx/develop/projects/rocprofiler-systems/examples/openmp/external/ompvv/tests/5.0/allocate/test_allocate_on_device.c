//===----------------- test_allocate_on_device.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Tests the allocate directive with allocator clause in the target region.
// A variable 'x' is allocated using and omp_default_mem_alloc allocator. 
// According to the spec omp_alloc invocations that appear in target regions 
// must not pass omp_null_allocator as the allocator argument, which must be 
// a constant expression that evaluates to one of the predefined memory allocator 
// values. Predefined allocators appearing in a uses_allocators clause cannot 
// have traits specified. The test checks that the values were written correctly, 
// and then frees the memory.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_allocate_on_device() {

  int A[N], errors = 0;

#pragma omp target map(tofrom: errors, A) uses_allocators(omp_default_mem_alloc)
  {
   int* x;
   #pragma omp allocate(x) allocator(omp_default_mem_alloc)
   x = (int *) omp_alloc(N*sizeof(int), omp_default_mem_alloc);

   #pragma omp parallel for 
    for (int i = 0; i < N; i++) {
      x[i] = 2*i;
    }
    for (int i = 0; i < N; i++) {
      A[i] = x[i] + i;
    }
    omp_free(x, omp_default_mem_alloc);
  }
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != 3*i);
  }
  return errors;
}

int main() {

  int errors = 0;

  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_on_device() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
