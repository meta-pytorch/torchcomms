//===--- qmcpack_target_math.c --- math lib invocation inside target---------===//
//
// OpenMP API Version 4.5 Nov 2015
//
//This is a QMCPACK specific test that looks at math library support 
//from within the target region. The array is initialized to FP_ZERO
//and subsequently pow math function is invoked form the target region.
//
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ompvv.h"

#define N 1000

int test_math_lib_inside_target() {
  OMPVV_INFOMSG("test_math_lib_inside_target");

  double array[N];
  int errors = 0;
  
  // Array initialization
  for (int i = 0; i < N; ++i) {
    array[i] = 0.99;
  }
  // This is intentional
  int c99_zero = FP_ZERO;
  
#pragma omp target map(tofrom: array[0:N]) 
  for (int i = 0; i < N; ++i) {
    array[i] = pow((double)i,2.0);
  }

  for (int i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, (array[i] - pow((double)i,2)) > 0.000009);
  }
  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_math_lib_inside_target());

  OMPVV_REPORT_AND_RETURN(errors);
}
