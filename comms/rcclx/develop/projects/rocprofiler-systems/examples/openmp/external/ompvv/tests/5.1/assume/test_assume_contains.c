//===--- test_assume_contains.c ---------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  The specification states that "when the contains clause appears on an assume
//  directive, the application proviedes a hint that constructs that match the
//  listed directive names are likely to be encountered in the scope of the
//  assume directive." In this test, we use "target parallel for" as the
//  directive-name for the contains clause.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int errors, i;

int test_assume_contains() {
    int arr[N];
    for(i = 0; i < N; i++){
        arr[i] = i;
    }
    #pragma omp assume contains(target, parallel, for)
    {
        #pragma omp target map(tofrom: arr)
        #pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE)
        #pragma omp for
        for(i = 0; i < N; i++){
            arr[i] = arr[i]*2;
        }
    }
    // Test that no issues were caused from using assume directive
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != i*2);
    }
    return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_assume_contains() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
