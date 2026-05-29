//===---- test_target_data_map_array_translation.c - pointer to already mapped array-===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// According to the OpenMP Specs, if a pointer is used in the mapping of a variable
// and the pointer points to a host array that is already available in the device,
// this pointer's address has to be updated with the device address. 
//
// See page 105, lines 24 through 32.
//
// This test check these conditions are valid
//
//===----------------------------------------------------------------------===//

#include "ompvv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000

int test_map_same_function() {

  OMPVV_INFOMSG("Testing map same function definition")

  int sum = 0, sum2 = 0, errors = 0;

  // host arrays: heap and stack
  int *h_array_h = (int *)malloc(N * sizeof(int));
  int h_array_s[N];

  // Pointers to be used for the translation
  int *ptr_h_array_h;
  int *ptr_h_array_s;

#pragma omp target data map(h_array_h[0:N]) map(h_array_s[0:N])
  {
    // Multiple tests at once.

    ptr_h_array_h = h_array_h;
    ptr_h_array_s = h_array_s;

    OMPVV_INFOMSG("map(ptr) specified full-length array section")
#pragma omp target map(ptr_h_array_h[0:N], ptr_h_array_s[0:N])
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] = 1;
        ptr_h_array_s[i] = 2;
      }
    } // end target

    OMPVV_INFOMSG("map(ptr) specified zero-length array section")
#pragma omp target map(ptr_h_array_h[:0], ptr_h_array_s[:0])
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] += 1;
        ptr_h_array_s[i] += 2;
      }
    } // end target

    OMPVV_INFOMSG("no map(ptr) Specified")
#pragma omp target 
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] += 1;
        ptr_h_array_s[i] += 2;
      }
    } // end target

  } // end target data

  // checking results
  for (int i = 0; i < N; ++i) {
    sum += h_array_h[i];
    sum2 += h_array_s[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != sum) || (6*N != sum2));

  free(h_array_h);
  return errors;
}

void helper_function(int *ptr_h_array_h, int *ptr_h_array_s) {

    OMPVV_INFOMSG("map(ptr) specified full-length array section")
#pragma omp target map(ptr_h_array_h[0:N], ptr_h_array_s[0:N])
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] = 1;
        ptr_h_array_s[i] = 2;
      }
    } // end target

    OMPVV_INFOMSG("map(ptr) specified zero-length array section")
#pragma omp target map(ptr_h_array_h[:0], ptr_h_array_s[:0])
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] += 1;
        ptr_h_array_s[i] += 2;
      }
    } // end target

    OMPVV_INFOMSG("no map(ptr) Specified")
#pragma omp target 
    {
      for (int i = 0; i < N; ++i) {
        ptr_h_array_h[i] += 1;
        ptr_h_array_s[i] += 2;
      }
    } // end target
}

int test_map_different_function() {

  OMPVV_INFOMSG("Testing map different function definition")

  int sum = 0, sum2 = 0, errors = 0;

  // host arrays: heap and stack
  int *h_array_h = (int *)malloc(N * sizeof(int));
  int h_array_s[N];

#pragma omp target data map(h_array_h[0:N]) map(h_array_s[0:N])
  {
    // Multiple tests at once.
    helper_function(h_array_h, h_array_s);

  } // end target data

  // checking results
  for (int i = 0; i < N; ++i) {
    sum += h_array_h[i];
    sum2 += h_array_s[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != sum) || (6*N != sum2));

  free(h_array_h);
  return errors;
}
int main() {
  OMPVV_TEST_OFFLOADING
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_same_function());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_different_function());
  OMPVV_REPORT_AND_RETURN(errors);
}
