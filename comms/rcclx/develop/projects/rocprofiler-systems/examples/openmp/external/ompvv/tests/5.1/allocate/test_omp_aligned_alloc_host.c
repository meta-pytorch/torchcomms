//===------ test_omp_aligned_alloc_host.c ------------------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Tests the omp_aligned_alloc routine properly requests a memory allocation
// aligned to the passed byte parameter. Starts by initializing some memory
// in the omp_default_mem_space and then checks if this memory is aligned
// and can be properly written to and read from. Checks on host.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ompvv.h"

#define N 1024

int test_omp_aligned_alloc_on_host() {
  int errors = 0;
  int *x;

  omp_memspace_handle_t  memspace = omp_default_mem_space;
  omp_alloctrait_t       traits[1] = {{omp_atk_alignment, 64}};
  omp_allocator_handle_t alloc = omp_init_allocator(memspace,1,traits);

  x = (int *)omp_aligned_alloc(64, N*sizeof(int), alloc);
    
  OMPVV_TEST_AND_SET_VERBOSE(errors, ((intptr_t)(x))%64 != 0);

  int values_did_not_match_expected_changes = 0;

  #pragma omp parallel for simd simdlen(16) aligned(x: 64)
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  #pragma omp parallel for simd simdlen(16) aligned(x: 64)
  for (int i = 0; i < N; i++) {
    if (x[i] != i) {
      values_did_not_match_expected_changes = 1;
    }
  }


  if (values_did_not_match_expected_changes) {
    OMPVV_ERROR("Array values did not match the expected values");
    errors++;
  }

  omp_free(x, alloc);
  omp_destroy_allocator(alloc);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_omp_aligned_alloc_on_host() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}


