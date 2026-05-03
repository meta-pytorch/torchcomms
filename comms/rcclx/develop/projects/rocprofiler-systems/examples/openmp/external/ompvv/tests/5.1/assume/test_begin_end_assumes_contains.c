//===--- test_begin_end_assumes.c -------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  Test for the "begin_end" version of assumes. Spec states that
//  "declaration-definition-seq" goes between the begin & end assumes directives.
//  In this test, the function declaration is listed between these w/ the 
//  contain(target parallel for) clause, telling the compiler that this 
//  directive will be used in this function.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

#pragma omp begin assumes contains(target, parallel, for)
int test_begin_end_assumes();
#pragma omp end assumes

int errors, i;

int test_begin_end_assumes() {
    int arr[N];
    for(i = 0; i < N; i++){
        arr[i] = i;
    }

    #pragma omp target map(tofrom: arr)
    #pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE)
    #pragma omp for
    for(i = 0; i < N; i++){
        arr[i] = arr[i]*2;
    }
    // Test that no issues were caused from using assumes directive
    for(i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, arr[i] != i*2);
    }
   return errors;
}
int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_begin_end_assumes() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
