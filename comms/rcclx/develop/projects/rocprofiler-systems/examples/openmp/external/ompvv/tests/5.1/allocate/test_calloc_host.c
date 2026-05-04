//===------------------------ test_calloc_host.c --------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Tests the omp_calloc routine to make sure memory is properly allocated
// on the host and is zero initialized. Allocates an array and makes sure
// each value is set to 0 to start. Then adds i to each index and validates the
// changes persisted and were written correctly.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ompvv.h"

#define N 1024

int test_omp_calloc_host() {
  int errors = 0;
  int *x;

  x = (int *)omp_calloc(64, N*sizeof(int), omp_default_mem_alloc);

  if (x == NULL) {
    OMPVV_ERROR("omp_calloc returned null"); 
    return (1); 
  }

  int not_init_to_zero = 0;
  int not_correct_updated_values = 0;

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    if (x[i] != 0) {
      #pragma omp atomic write
      not_init_to_zero = 1;
    }  
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }
  
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    if (x[i] != i) {
      #pragma omp atomic write
      not_correct_updated_values = 1;
    }
  }

  if (not_init_to_zero) {
    OMPVV_ERROR("Values were not initialized to 0");
    errors++;
  }

  if (not_correct_updated_values) {
    OMPVV_ERROR("Values in the array did NOT match the expected values. Changes may not have persisted.");
    errors++;
  }

  omp_free(x, omp_default_mem_alloc);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_omp_calloc_host() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
