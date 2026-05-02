//===---- test_target_data_map_alloc.c ---------------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// The test_target_data_map group of tests checks all the possible map-type-modifiers
// for the target data map clauses. These are: from, to, tofrom, alloc, release and 
// delete. There is a a separate c file for each test with the prefix test_target_data_map.
// This specific test is for the 'alloc' clause. 
//
//===----------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

// Test for OpenMP 4.5 target data map(alloc:)
int test_map_alloc() {

  OMPVV_INFOMSG("test_map_alloc");

  int sum = 0, errors = 0;
  int *h_array_h = (int *)malloc(N*sizeof(int));

  // pointer arithmetic is not supported on the devices for
  // the device address returned by omp_target_alloc
  // section 3.5.1 omp_target_alloc. OpenMP API Version 4.5 Nov 2015
  int *d_sum = (int *)omp_target_alloc(sizeof(int), omp_get_default_device());

  if (d_sum == NULL) {
    errors = 1;
    OMPVV_ERROR("omp_target_alloc returns NULL, this test is running on host, cannot properly test mapping of variables to device.");
    return errors;
  }

#pragma omp target data map(alloc: h_array_h[0:N])
  {
#pragma omp target is_device_ptr(d_sum)
    {
      for (int i = 0; i < N; ++i) 
        h_array_h[i] = 1;
      
      // checking errors
      d_sum[0] = 0; 
      for (int i = 0; i < N; ++i)
        d_sum[0] += h_array_h[i];
    } // end target
    omp_target_memcpy(&sum, d_sum, sizeof(int), 0, 0,
                                  omp_get_initial_device(),
                                  omp_get_default_device());
  } // end target data
  omp_target_free(d_sum, omp_get_default_device());

  free(h_array_h);
  OMPVV_TEST_AND_SET_VERBOSE(errors, (N - sum) != 0);

  return errors;
}

int main() {

  int errors = 0;
  
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_alloc());
  OMPVV_REPORT_AND_RETURN(errors);
}
  
