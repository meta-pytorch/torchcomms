//===--------------------- test_target_aligned_calloc.c -------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Tests the omp_aligned_calloc routine. Requests a memory allocation
// aligned to the passed byte parameter. Starts by initializing some memory
// in the omp_default_mem_space and then checks if this memory is aligned
// and can be properly written to and read from. Checks on the device.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ompvv.h"

#define N 1024

int test_aligned_calloc_on_device() {
  int errors = 0;

  #pragma omp target map(tofrom: errors) uses_allocators(omp_default_mem_alloc) 
  {
    int *x;
    int not_correct_array_values = 0;
  
    x = (int *)omp_aligned_calloc(64, N, N*sizeof(int), omp_default_mem_alloc);
    
    if (x == NULL) { 
      OMPVV_ERROR("omp_aligned_calloc returned null"); 
      errors++;
    } else {
      OMPVV_TEST_AND_SET(errors, ((intptr_t)(x))%64 != 0);
      OMPVV_ERROR_IF(((intptr_t)(x))%64 != 0, " Condition ((intptr_t)(x))%%64 != 0 failed. The memory does not seem to be properly aligned.");

      #pragma omp parallel for simd simdlen(16) aligned(x: 64)
      for (int i = 0; i < N; i++) {
        x[i] = i;
      }

      #pragma omp parallel for simd simdlen(16) aligned(x: 64)
      for (int i = 0; i < N; i++) {
        if (x[i] != i) {
          #pragma omp atomic write
          not_correct_array_values = 1; 
        }
      }

      if (not_correct_array_values) {
        OMPVV_ERROR("Values in the array did NOT match the expected values. Changes may not have persisted.");
        errors++;
      }

      omp_free(x, omp_default_mem_alloc);
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_aligned_calloc_on_device() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}


