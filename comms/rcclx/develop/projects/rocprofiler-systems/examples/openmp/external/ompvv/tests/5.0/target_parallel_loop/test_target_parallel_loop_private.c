//===-------------------test_target_parallel_loop_private.c-------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Description - This is a test program to demonstrate how private clause is
// used with target parallel loop construct.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024

int main() {
    OMPVV_TEST_OFFLOADING;
    int a[SIZE];
    int b[SIZE];
    int priv_val = 12345;
    int errors = 0;
    int num_threads = -1;

  // Initialization
    for (int x = 0; x < SIZE; ++x) {
        a[x] = 5*x;
        b[x] = 0;
    }

    #pragma omp target parallel loop private(priv_val) \
                       num_threads(OMPVV_NUM_THREADS_DEVICE)
        for (int x = 0; x < SIZE; ++x) {
            priv_val = 0;
            for (int y = 0; y < x+1; ++y) {
                       priv_val++;
            }

                b[x] = a[x] * priv_val;
        }

    OMPVV_TEST_AND_SET_VERBOSE(errors, priv_val != 12345);

    for (int x = 0; x < SIZE; ++x) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, b[x] != (1 + x)*5*x);
    }

    OMPVV_REPORT_AND_RETURN(errors);
}
