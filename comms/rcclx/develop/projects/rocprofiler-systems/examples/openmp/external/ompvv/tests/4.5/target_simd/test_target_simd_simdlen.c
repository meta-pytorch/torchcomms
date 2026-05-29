//===---- test_target_simd_simdlen.c - simd directive clause simdlen   -===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test checks for the use of the simdlen clause which provides a hint to the
// compiler to the appropriate number of simd lanes when generating the simd
// region of code. However, different to safelen, this clause does not guarantee 
// that the number of lanes will actually be the one specified. The number of lanes
// is implementation defined 
// ===--------------------------------------------------------------------------===//
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define ARRAY_SIZE 1024

int test_target_simd_simdlen() {
  OMPVV_INFOMSG("test_target_simd_simdlen");
  OMPVV_WARNING("This test cannot check if actual SIMD extensions at the hardware level" \
                 " were used, or of the generated code is different in any way");

  // Variable for errors counting
  int errors = 0;

  int A[ARRAY_SIZE];
  int i, len;

  // a and b array initialization
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] = 1;
  }

  // Test simdlen of 1 5 8 13 16 100 128
#pragma omp target simd simdlen(1) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(5) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(8) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(13) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(16) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(100) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

#pragma omp target simd simdlen(128) map(tofrom: A[0:ARRAY_SIZE])
  for (i = 0; i < ARRAY_SIZE; ++i) {
      A[i] += A[i]; 
  }

  // Check the results
  for (i = 0; i < ARRAY_SIZE; ++i) {
      OMPVV_TEST_AND_SET(errors, A[i] != 1<<7);
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_simd_simdlen());

  OMPVV_REPORT_AND_RETURN(errors);
}
