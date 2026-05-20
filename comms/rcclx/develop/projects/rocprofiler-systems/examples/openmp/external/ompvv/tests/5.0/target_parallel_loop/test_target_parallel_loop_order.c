//===---------------test_target_parallel_loop_order_concurrent.c-----------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Description - This is a test program to demonstrate how order clause is
// used with target parallel loop construct.
//
////===--------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

#define SIZE 1024

int test_target_parallel_loop_order_concurrent() {
    OMPVV_INFOMSG("test_target_parallel_loop_order_concurrent");
    int a[SIZE];
    int b[SIZE];
    int c[SIZE];
    int errors = 0;

    // Initialization
    for (int i = 0; i < SIZE; i++) {
        a[i] = 1;
        b[i] = i + 1;
        c[i] = 2*(i + 1);
    }

#pragma omp target parallel loop order(concurrent)
    for (int i = 0; i < SIZE; i++) {
      a[i] += b[i]*c[i];
    }

    // Validation
    for (int i = 0; i < SIZE; i++) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != (1 + (b[i]*c[i])));
    }

    return errors;
}


int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors,
                            test_target_parallel_loop_order_concurrent());

  OMPVV_REPORT_AND_RETURN(errors);
}
