//===---- test_target_data_map_to.c ---------------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// The test_target_data_map group of tests checks all the possible map-type-modifiers
// for the target data map clauses. These are: from, to, fromto, alloc, release and 
// delete. There is a a separate c file for each test with the prefix test_target_data_map.
// This specific test is for the 'to' clause. 
//
//===----------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000
// Test for OpenMP 4.5 target data map(to: ) 
int test_map_to() {

  OMPVV_INFOMSG("test_map_to");

  int sum = 0, sum2 = 0, errors = 0;
  
  // host arrays: heap and stack
  int *h_array_h = (int *)malloc(N*sizeof(int));
  int *h_array2_h = (int *)malloc(N*sizeof(int));
  int h_array_s[N];
  int h_array2_s[N];

  // initializing arrays 
  for (int i = 0; i < N; ++i) {
    h_array_h[i] = 1;
    h_array_s[i] = 1;
    h_array2_h[i] = 0;
    h_array2_s[i] = 0;
  }

  // device arrays to get the data from the device
  // pointer arithmetic is not supported on the devices for
  // the device address returned by omp_target_alloc
  // section 3.5.1 omp_target_alloc. OpenMP API Version 4.5 Nov 2015
  int *d_array =
      (int *)omp_target_alloc(N*sizeof(int), omp_get_default_device());
  int *d_array2 =
      (int *)omp_target_alloc(N*sizeof(int), omp_get_default_device());

  if (d_array == NULL || d_array2 == NULL ) {
    errors = 1;
    OMPVV_ERROR("omp_target_alloc returns NULL, this test is running on host, cannot properly test mapping of variables to device.");
    return errors;
  }

#pragma omp target data map(to: h_array_h[0:N])  \
        map(to: h_array_s[0:N]) 
  {
#pragma omp target is_device_ptr(d_array, d_array2)
    {
      for (int i = 0; i < N; ++i) {
        d_array[i] = h_array_h[i];
        d_array2[i] = h_array_s[i];
      }
    } // end target
  } // end target data

  // copy from d to h
  omp_target_memcpy(h_array2_h, d_array, N*sizeof(int), 0, 0,
                    omp_get_initial_device(), omp_get_default_device());
  omp_target_memcpy(h_array2_s, d_array2, N*sizeof(int), 0, 0,
                    omp_get_initial_device(), omp_get_default_device());
  // deallocating device arrays 
  omp_target_free(d_array, omp_get_default_device());
  omp_target_free(d_array2, omp_get_default_device());

  // checking errors
  for (int i = 0; i < N; ++i) {
    sum += h_array2_h[i];
    sum2 += h_array2_s[i];
  }

  free(h_array_h);
  free(h_array2_h);
  OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != sum2));

  return errors;
}

int main() {

  int errors = 0;
  
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_to());
  OMPVV_REPORT_AND_RETURN(errors);
}

