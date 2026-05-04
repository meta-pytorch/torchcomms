//===---- test_target_data_map_from.c -----------------------------------------------------===//
//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// The test_target_data_map group of tests checks all the possible map-type-modifiers 
// for the target data map clauses. These are: from, to, fromto, alloc, release and delete.
// There is a separate c file for each test with the prefix test_target_data_map.
// This specific test is for mapping 'from'.
//
//===--------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

// Test for OpenMP 4.5 target data map(from: )
int test_map_from() {

  OMPVV_INFOMSG("test_map_from");

  int sum = 0, sum2 = 0, errors = 0;

  // host arrays: heap and stack
  int *h_array_h = (int *)malloc(N * sizeof(int));
  int h_array_s[N];

#pragma omp target data map(from: h_array_h[0:N])  \
        map(from: h_array_s[0:N])
  {
#pragma omp target
    {
      for (int i = 0; i < N; ++i) {
        h_array_h[i] = 1;
        h_array_s[i] = 2;
      }
    } // end target
  } // end target data

  // checking results
  for (int i = 0; i < N; ++i) {
    sum += h_array_h[i];
    sum2 += h_array_s[i];
  }
  
  free(h_array_h);
  OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (2*N != sum2));

  return errors;
}


int main() {

  int errors = 0;
  
  //Check for offloading
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_from());
  OMPVV_REPORT_AND_RETURN(errors);
}
