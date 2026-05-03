//===--- test_assumes_contains.c --------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  The purpose of this test in particular is to test the syntax listed as:
//  #pragma omp assumes clause[ [ [ ,]clause]...]new-line.
//
//  The scope of the assumes directive is the "the code executed and reached 
//  from the current compilation unit". In this test, we use assumes 
//  contains(target parallel for) to tell the compiler that this directive
//  will show up somewhere in our program, which it does.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

#pragma omp assumes contains(target, parallel, for)

int errors, i;

int test_assumes_contains() {
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
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_assumes_contains() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
