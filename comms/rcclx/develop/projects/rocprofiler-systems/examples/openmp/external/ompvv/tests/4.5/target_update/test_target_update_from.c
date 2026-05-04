//===---- test_target_update_from.c -------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test checks the target update motion clause 'from' by mapping an array 
// to the device with map-type 'to', changing the values of array on the device,
// and finally using the update 'from' motion clause to assign the value of the 
// list item. Back on the host, measures are taken to ensure the value was properly
// updated.  
//===--------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

#define N 100

int a[N];
int b[N];
int c[N];


// Test for OpenMP 4.5 target update with to
int main() {
  int errors = 0, i = 0, change_flag = 0;

  for (i = 0; i < N; i++) {
    a[i] = 10;
    b[i] = 2; 
  }

  // We test for offloading
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);


#pragma omp target data map(to: a[:N], b[:N]) 
{
  #pragma omp target
  {
    int j = 0;
    for (j = 0; j < N; j++) {
      b[j] = (a[j] + b[j]);// b = 12 
    }
  } // end target

  #pragma omp target update from(b[:N]) // update b = 12 on host 

} //end target-data
 
  // Checking values of b[N] 
  for (i = 0; i < N; i++) 
    OMPVV_TEST_AND_SET_VERBOSE(errors, (b[i] != 12)); 

  #pragma omp target 
  {
    int j = 0;
    for (j = 0; j < N; j++) {
      c[j] = (2* b[j]);// c = 24 
    }
  } // end target

  // Checking values of c[N] 
  for (i = 0; i < N; i++) 
    OMPVV_TEST_AND_SET_VERBOSE(errors, (c[i] != 24)); 
    
  OMPVV_REPORT_AND_RETURN(errors);
}
